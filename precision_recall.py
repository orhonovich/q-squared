# Copyright 2020 The Q2 Authors
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import pandas as pd
import numpy as np

import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

from collections import namedtuple
from sklearn import metrics
from baselines import add_baselines

sns.set_theme(style="darkgrid", font_scale=1.1)

PrecisionRecallValues = namedtuple('PrecisionRecallValues', ['precision', 'recall', 'thresholds'])

GROUNDED_LABEL = 1
UNGROUNDED_LABEL = 0
DEFAULT_THRESHOLD = 0.5


def compute_precision_recall_single_threshold(scores, labels, threshold):
    """Returns a dictionary containing the grounded and ungrounded Precision-Recall values, for a given threshold."""
    predictions = []
    for score in scores:
        if score <= threshold:
            predictions.append(UNGROUNDED_LABEL)
        else:
            predictions.append(GROUNDED_LABEL)
    grounded_precision = metrics.precision_score(y_true=labels, y_pred=predictions, pos_label=GROUNDED_LABEL)
    grounded_recall = metrics.recall_score(y_true=labels, y_pred=predictions, pos_label=GROUNDED_LABEL)
    ungrounded_precision = metrics.precision_score(y_true=labels, y_pred=predictions, pos_label=UNGROUNDED_LABEL)
    ungrounded_recall = metrics.recall_score(y_true=labels, y_pred=predictions, pos_label=UNGROUNDED_LABEL)
    accuracy = metrics.accuracy_score(y_true=labels, y_pred=predictions)
    result_dict = {
        'grounded_precision': grounded_precision,
        'grounded_recall': grounded_recall,
        'ungrounded_precision': ungrounded_precision,
        'ungrounded_recall': ungrounded_recall,
        'accuracy': accuracy
    }
    return result_dict


def compute_precision_recall_various_thresholds(scores, labels, grounded_detection=False):
    """Computes the Precision-Recall values for different thresholds and returns three arrays: Precision values,
    Recall values, and the corresponding thresholds."""
    if not grounded_detection:
        # In the ungrounded case, each example will be predicted as ungrounded if
        # its score is greater than, or equal to, the threshold. We therefore take
        # 1-score for ungrounded text detection.
        scores = 1 - scores

    precision, recall, thresholds = metrics.precision_recall_curve(y_true=labels, probas_pred=scores,
                                                                   pos_label=int(grounded_detection))
    return PrecisionRecallValues(precision=precision, recall=recall, thresholds=thresholds)


def plot_precision_recall_vs_thresholds(precisions, recalls, thresholds, plot_grounded, fig_name):
    """Plots the Precision and Recall values for various thresholds."""
    if not plot_grounded:
        # The Precision and Recall were calculated by classifying as ungrounded any
        # example for which (1-score) is larger than the threshold. We would like
        # the plot to show the Precision and Recall for classifying as ungrounded any
        # example for which the score in smaller than the threshold.
        thresholds = 1 - thresholds
    plt.figure()
    plt.plot(thresholds, precisions[:-1], 'b--', label='precision')
    plt.plot(thresholds, recalls[:-1], 'g--', label='recall')
    plt.xlabel('Threshold')
    plt.legend()
    if plot_grounded:
        plot_type = 'grounded'
    else:
        plot_type = 'ungrounded'
    plot_title = f'Precision and Recall vs. various thresholds, {plot_type} detection'
    plt.title(plot_title)
    plt.savefig(fig_name)


def create_single_metric_plots(scores, labels, metric_name, fig_name):
    """Plot the Grounded and Ungrounded Precision and Recall for a given metric, for various thresholds."""
    grounded_precision_recall = compute_precision_recall_various_thresholds(scores, labels, grounded_detection=True)
    plot_precision_recall_vs_thresholds(grounded_precision_recall.precision, grounded_precision_recall.recall,
                                        grounded_precision_recall.thresholds,
                                        True, f'{fig_name}_{metric_name}_grounded.png')
    ungrounded_precision_recall = compute_precision_recall_various_thresholds(scores, labels, grounded_detection=False)
    plot_precision_recall_vs_thresholds(ungrounded_precision_recall.precision, ungrounded_precision_recall.recall,
                                        ungrounded_precision_recall.thresholds, False,
                                        f'{fig_name}_{metric_name}_ungrounded.png')


def create_multiple_metrics_comparison_plots(metrics_scores, fig_name):
    """
    Plot the Precision-Recall curves for several metrics.

    The plot shows the Precision-Recall trade-off for several input metrics, allowing comparison between the different
     metrics.
    """
    precision = np.array([])
    recall = np.array([])
    metric_type = []
    for metric, scores in metrics_scores.items():
        inconsistent = scores[0]
        consistent = scores[1]
        metric_scores = np.append(inconsistent, consistent)

        gold_score_inconsistent = np.zeros(shape=(len(inconsistent)))
        gold_score_consistent = np.ones(shape=(len(consistent)))
        gold_scores = np.append(gold_score_inconsistent, gold_score_consistent)

        metric_precision, metric_recall, _ = metrics.precision_recall_curve(y_true=gold_scores, probas_pred=metric_scores)

        precision = np.append(precision, metric_precision)
        recall = np.append(recall, metric_recall)
        metric_type.extend([metric] * len(metric_precision))

    plt.figure()
    for_plt = pd.DataFrame({"Recall": recall, "Precision": precision, "Metric": metric_type})

    sns_plot = sns.lineplot(x='Recall', y='Precision', hue="Metric",
                            data=for_plt).set_title("Precision vs. Recall, consistent and inconsistent scores")
    sns_plot.figure.savefig(fig_name)


def plot_hist(metrics_scores, fig_name):
    for metric, scores in metrics_scores.items():
        inconsistent = scores[0]
        consistent = scores[1]

        plt.figure()
        df_inconsistent = pd.DataFrame({"Score": inconsistent})
        sns_plot = sns.histplot(df_inconsistent, x="Score", bins=10).set_title("Histogram of response scores, {0}, "
                                                                  "inconsistent data".format(metric))
        sns_plot.figure.savefig("{0}_inconsistent_{1}".format(fig_name, metric))

        plt.figure()
        df_consistent = pd.DataFrame({"Score": consistent})
        sns_plot = sns.histplot(df_consistent, x="Score", bins=10).set_title("Histogram of response scores, {0}, "
                                                                             "consistent data".format(metric))
        sns_plot.figure.savefig("{0}_consistent_{1}".format(fig_name, metric))


def get_metric_scores(incons_dodeca, cons_dodeca, incons_memnet, cons_memnet, metric_type):
    incons_scores_dodeca = incons_dodeca[metric_type].to_numpy(dtype=np.float64)
    cons_scores_dodeca = cons_dodeca[metric_type].to_numpy(dtype=np.float64)

    incons_scores_memnet = incons_memnet[metric_type].to_numpy(dtype=np.float64)
    cons_scores_memnet = cons_memnet[metric_type].to_numpy(dtype=np.float64)

    inconsistent_scores = np.append(incons_scores_dodeca, incons_scores_memnet)
    consistent_scores = np.append(cons_scores_dodeca, cons_scores_memnet)

    return inconsistent_scores, consistent_scores


def response_level_evaluation(incons_dodeca, cons_dodeca, incons_memnet, cons_memnet, metrics_names, metric_to_threshold):
    scores_dict = {}

    for metric_name in metrics_names:
        inconsistent_metric_scores, consistent_metric_scores = get_metric_scores(incons_dodeca, cons_dodeca,
                                                                 incons_memnet, cons_memnet, metric_name)
        # Normalize BLEU scores to be in [0,1]
        if metric_name == 'bleu':
            inconsistent_metric_scores = inconsistent_metric_scores / 100
            consistent_metric_scores = consistent_metric_scores / 100

        # Normalize BERTScore to be in [0,1]
        elif metric_name == 'bertscore':
            min_bertscore = np.amin(np.append(inconsistent_metric_scores, consistent_metric_scores))
            inconsistent_metric_scores = inconsistent_metric_scores - min_bertscore
            consistent_metric_scores = consistent_metric_scores - min_bertscore

            max_bertscore = np.amax(np.append(inconsistent_metric_scores, consistent_metric_scores))
            inconsistent_metric_scores = inconsistent_metric_scores / max_bertscore
            consistent_metric_scores = consistent_metric_scores / max_bertscore

        metric_scores = np.append(inconsistent_metric_scores, consistent_metric_scores)
        gold_labels_inconsistent = np.zeros(shape=(len(inconsistent_metric_scores)))
        gold_labels_consistent = np.ones(shape=(len(consistent_metric_scores)))
        gold_labels = np.append(gold_labels_inconsistent, gold_labels_consistent)
        create_single_metric_plots(metric_scores, gold_labels, metric_name, 'metric_precision_recall')
        precision_recall_dict = compute_precision_recall_single_threshold(metric_scores, gold_labels,
                                                                          metric_to_threshold[metric_name])
        print(f'For metric {metric_name}:', precision_recall_dict)

        scores_dict[metric_name] = [inconsistent_metric_scores, consistent_metric_scores]

    create_multiple_metrics_comparison_plots(scores_dict, 'precision_recall_comparison_2.png')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--incons_dodeca_f", type=str, required=True)
    parser.add_argument("--cons_dodeca_f", type=str, required=True)
    parser.add_argument("--incons_memnet_f", type=str, required=True)
    parser.add_argument("--cons_memnet_f", type=str, required=True)
    parser.add_argument('--metrics_names', nargs="+", required=True)
    parser.add_argument('--thresholds', nargs="*", required=False)
    parser.add_argument("--add_baselines", default=False, action="store_true",
                        help="Whether to include baseline methods in the meta-evaluation.")
    args = parser.parse_args()

    inconsistent_dodeca = pd.read_csv(args.incons_dodeca_f)
    consistent_dodeca = pd.read_csv(args.cons_dodeca_f)
    inconsistent_memnet = pd.read_csv(args.incons_memnet_f)
    consistent_memnet = pd.read_csv(args.cons_memnet_f)

    input_metrics_names = args.metrics_names
    input_thresholds = args.thresholds
    if args.add_baselines:
        inconsistent_dodeca = add_baselines(inconsistent_dodeca)
        consistent_dodeca = add_baselines(consistent_dodeca)
        inconsistent_memnet = add_baselines(inconsistent_memnet)
        consistent_memnet = add_baselines(consistent_memnet)
        input_metrics_names.extend(['overlap', 'bleu', 'bertscore'])

    thresholds_dict = {}

    for i, name in enumerate(input_metrics_names):
        if input_thresholds:
            thresholds_dict[name] = float(input_thresholds[i])
        else:
            thresholds_dict[name] = DEFAULT_THRESHOLD

    response_level_evaluation(inconsistent_dodeca, consistent_dodeca, inconsistent_memnet, consistent_memnet,
                              args.metrics_names, thresholds_dict)


