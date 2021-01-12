import argparse

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

TOTAL = 350


def bootstrap(df, qualities, metric, n_iter=1000):

    dodeca_col = "dodeca_{0}".format(metric)
    memnet_col = "memnet_{0}".format(metric)

    correlations = []

    for i in range(n_iter):
        sampled = df.sample(n=TOTAL, replace=True)  # sample sentences
        sys_scores = []

        for q in qualities:
            num_inconsistent = round(q * TOTAL)
            # For each example, we choose whether to take the consistent or inconsistent response
            inconsistent_sample = np.random.choice(np.arange(TOTAL), size=num_inconsistent, replace=False)

            metric_scores = []
            curr_sample = 0
            for _, row in sampled.iterrows():
                if curr_sample in inconsistent_sample:
                    # Select the system that had an inconsistency (labeled as 1)
                    if int(row['dodeca_label']) == 1:
                        metric_scores.append(float(row[dodeca_col]))
                    else:  # memnet label is 1
                        metric_scores.append(float(row[memnet_col]))

                else:  # Else, take the score for the consistent response
                    if int(row['dodeca_label']) == 0:
                        metric_scores.append(float(row[dodeca_col]))
                    else:  # memnet label is 0
                        metric_scores.append(float(row[memnet_col]))
                curr_sample += 1

            sys_scores.append(np.mean(metric_scores))

        ranks = np.arange(1, len(qualities) + 1)[::-1]
        corr = spearmanr(ranks, sys_scores)[0]
        correlations.append(corr)

    return correlations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, required=True,
                        help="Path to a csv file containing metric scores and baselines.")

    args = parser.parse_args()

    df = pd.read_csv(args.infile)

    # Keep examples in which one system was consistent and the other wasn't
    df = df[df['dodeca_label'] + df['memnet_label'] == 1]

    for metric in ['q2', 'overlap', 'bleu']:
        correlations = bootstrap(df, [0.05, 0.1, 0.15, 0.2, 0.25], metric, n_iter=1000)

        print(metric)
        print(np.mean(correlations))
        print(np.percentile(correlations, 2.5))
        print(np.percentile(correlations, 97.5))