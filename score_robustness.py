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

from sklearn.metrics import precision_recall_curve
from baselines import add_baselines

sns.set_theme(style="darkgrid", font_scale=1.1)


def plt_precision_recall(metrics_scores, fig_name):
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

        metric_precision, metric_recall, _ = precision_recall_curve(y_true=gold_scores, probas_pred=metric_scores)

        precision = np.append(precision, metric_precision)
        recall = np.append(recall, metric_recall)
        metric_type.extend([metric] * len(metric_precision))

    plt.figure()
    for_plt = pd.DataFrame({"Recall": recall, "Precision": precision, "Metric": metric_type})

    sns_plot = sns.lineplot(x='Recall', y='Precision', hue="Metric",
                            data=for_plt).set_title("Precision vs. Recall, consistent and inconsistent scores")
    sns_plot.figure.savefig(fig_name)


def plt_hist(metrics_scores, fig_name):
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


def get_metric_scores(incons_dodeca, cons_dodeca, incons_memnet, cons_memnet, metric_type='score'):
    incons_scores_dodeca = incons_dodeca[metric_type].to_numpy(dtype=np.float64)
    cons_scores_dodeca = cons_dodeca[metric_type].to_numpy(dtype=np.float64)

    incons_scores_memnet = incons_memnet[metric_type].to_numpy(dtype=np.float64)
    cons_scores_memnet = cons_memnet[metric_type].to_numpy(dtype=np.float64)

    inconsistent_scores = np.append(incons_scores_dodeca, incons_scores_memnet)
    consistent_scores = np.append(cons_scores_dodeca, cons_scores_memnet)

    return inconsistent_scores, consistent_scores


def response_level(incons_dodeca, cons_dodeca, incons_memnet, cons_memnet):
    scores_dict = {}

    inconsistent_nli, consistent_nli = get_metric_scores(incons_dodeca, cons_dodeca,
                                                       incons_memnet, cons_memnet, 'Q2')
    inconsistent_no_nli, consistent_no_nli = get_metric_scores(incons_dodeca, cons_dodeca,
                                                       incons_memnet, cons_memnet, 'Q2_no_nli')
    # inconsistent_e2e, consistent_e2e = get_metric_scores(incons_dodeca, cons_dodeca,
    #                                                    incons_memnet, cons_memnet, 'E2E_NLI')
    inconsistent_overlap, consistent_overlap = get_metric_scores(incons_dodeca, cons_dodeca,
                                                                 incons_memnet, cons_memnet, 'overlap')
    inconsistent_bleu, consistent_bleu = get_metric_scores(incons_dodeca, cons_dodeca,
                                                                 incons_memnet, cons_memnet, 'bleu')
    inconsistent_bert, consistent_bert = get_metric_scores(incons_dodeca, cons_dodeca,
                                                                 incons_memnet, cons_memnet, 'bertscore')

    # Normalize BLEU scores to be in [0,1]
    inconsistent_bleu = inconsistent_bleu / 100
    consistent_bleu = consistent_bleu / 100

    # Normalize BERTScore to be in [0,1]
    min_bertscore = np.amin(np.append(inconsistent_bert, consistent_bert))
    inconsistent_bert = inconsistent_bert - min_bertscore
    consistent_bert = consistent_bert - min_bertscore

    max_bertscore = np.amax(np.append(inconsistent_bert, consistent_bert))
    inconsistent_bert = inconsistent_bert / max_bertscore
    consistent_bert = consistent_bert / max_bertscore


    scores_dict['Q2'] = [inconsistent_nli, consistent_nli]
    scores_dict[r'Q2 w/o NLI'] = [inconsistent_no_nli, consistent_no_nli]
    # scores_dict[r'E2E NLI'] = [inconsistent_e2e, consistent_e2e]
    scores_dict['Overlap'] = [inconsistent_overlap, consistent_overlap]
    scores_dict['BERTScore'] = [inconsistent_bert, consistent_bert]
    scores_dict['BLEU'] = [inconsistent_bleu, consistent_bleu]

    plt_precision_recall(scores_dict, 'prec_recall_curve.png')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--incons_dodeca_f", type=str, required=True)
    parser.add_argument("--cons_dodeca_f", type=str, required=True)
    parser.add_argument("--incons_memnet_f", type=str, required=True)
    parser.add_argument("--cons_memnet_f", type=str, required=True)
    args = parser.parse_args()

    incons_dodeca_df = add_baselines(pd.read_csv(args.incons_dodeca_f))
    cons_dodeca_df = add_baselines(pd.read_csv(args.cons_dodeca_f))
    incons_memnet_df = add_baselines(pd.read_csv(args.incons_memnet_f))
    cons_memnet_df = add_baselines(pd.read_csv(args.cons_memnet_f))
    response_level(incons_dodeca_df, cons_dodeca_df, incons_memnet_df, cons_memnet_df)


