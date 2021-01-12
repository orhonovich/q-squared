import argparse

import pandas as pd
import numpy as np

import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import precision_recall_curve

sns.set_theme(style="darkgrid", font_scale=1.1)


TOTAL = 400


def system_at_k(inconsistent, consistent, k, n_iter=1000):
    sys_scores = np.zeros(shape=(n_iter))

    num_inconsistent = round(k*TOTAL)
    num_consistent = TOTAL - num_inconsistent

    for i in range(n_iter):
        inconsistent_sample = np.random.choice(inconsistent, size=num_inconsistent, replace=True)
        consistent_sample = np.random.choice(consistent, size=num_consistent, replace=True)

        system = np.concatenate((inconsistent_sample, consistent_sample))
        sys_scores[i] = np.mean(system)

    low = np.mean(sys_scores) - np.percentile(sys_scores, 2.5)
    high = np.percentile(sys_scores, 97.5) - np.mean(sys_scores)
    return np.mean(sys_scores), low, high


def bootstrap_robustness(metrics_scores, qualities, fig_name):
    mean = np.array([])
    intervals = []
    metric_type = []
    for metric, scores in metrics_scores.items():
        metric_type.extend([metric] * len(qualities))
        inconsistent = scores[0]
        consistent = scores[1]
        low = []
        high = []
        for k in qualities:
            m, l, h = system_at_k(inconsistent, consistent, k)
            mean = np.append(mean, m)
            low.append(l)
            high.append(h)
        intervals.append([low, high])

    plt.figure()
    plt.title('Score Robustness')
    for_plt = pd.DataFrame({"Proportion of inconsistent responses": qualities * len(metrics_scores),
                            "Avg. score": mean, "Metric": metric_type})
    sns_plot = sns.lineplot(data=for_plt, x='Proportion of inconsistent responses',
                            y='Avg. score', hue='Metric').set_title("Score robustness")
    mean = mean.reshape((len(metrics_scores), -1))
    for i, interval in enumerate(intervals):
        plt.errorbar(qualities, mean[i], yerr=interval, fmt='none', c='C' + str(i))
    sns_plot.figure.savefig(fig_name)


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--incons_dodeca_f", type=str, required=True)
    parser.add_argument("--cons_dodeca_f", type=str, required=True)
    parser.add_argument("--incons_memnet_f", type=str, required=True)
    parser.add_argument("--cons_memnet_f", type=str, required=True)
    args = parser.parse_args()

    incons_dodeca = pd.read_csv(args.incons_dodeca_f)
    cons_dodeca = pd.read_csv(args.cons_dodeca_f)
    incons_memnet = pd.read_csv(args.incons_memnet_f)
    cons_memnet = pd.read_csv(args.cons_memnet_f)

    scores_dict = {}

    inconsistent_q2, consistent_q2 = get_metric_scores(incons_dodeca, cons_dodeca,
                                                       incons_memnet, cons_memnet, 'Q2')
    inconsistent_overlap, consistent_overlap = get_metric_scores(incons_dodeca, cons_dodeca,
                                                                 incons_memnet, cons_memnet, 'overlap')
    inconsistent_bleu, consistent_bleu = get_metric_scores(incons_dodeca, cons_dodeca,
                                                                 incons_memnet, cons_memnet, 'bleu')

    scores_dict['Q2'] = [inconsistent_q2, consistent_q2]
    scores_dict['overlap'] = [inconsistent_overlap, consistent_overlap]

    bootstrap_robustness(scores_dict, [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0], 'organized_plot_new_large')
    bootstrap_robustness(scores_dict, [1, 0.25, 0.2, 0.15, 0.1, 0.05, 0], 'organized_plot_new_small')

    scores_dict['bleu'] = [inconsistent_bleu, consistent_bleu]
    plt_precision_recall(scores_dict, 'prec_recall_new.png')

    plt_hist(scores_dict, 'hist_new')
