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
from scipy.stats import pearsonr, spearmanr

from baselines import cross_add_baselines

TOTAL = 350


def merge_cross_annotations(dodeca_f, memnet_f):
    df_dodeca = pd.read_csv(dodeca_f)
    df_memnet = pd.read_csv(memnet_f)
    merged = df_dodeca.merge(df_memnet, left_on='id', right_on='id')
    return merged


def bootstrap(df, qualities, metric, n_iter=1000):

    dodeca_col = "{0}_x".format(metric)
    memnet_col = "{0}_y".format(metric)

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
                    if int(row['label_x']) == 1:
                        metric_scores.append(float(row[dodeca_col]))
                    else:  # memnet label is 1
                        metric_scores.append(float(row[memnet_col]))

                else:  # Else, take the score for the consistent response
                    if int(row['label_x']) == 0:
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
    parser.add_argument("--dodeca_path", type=str, required=True,
                        help="Path to a csv file containing metric for the dodeca system outputs.")
    parser.add_argument("--memnet_path", type=str, required=True,
                        help="Path to a csv file containing metric for the memnet system outputs.")
    parser.add_argument('--metrics_names', nargs="+", required=True)
    parser.add_argument("--add_baselines", default=False, action="store_true",
                        help="Whether to include baseline methods in the meta-evaluation.")

    args = parser.parse_args()
    merged = merge_cross_annotations(args.dodeca_path, args.memnet_path)

    # Keep examples in which one system was consistent and the other wasn't
    merged = merged[merged['label_x'] + merged['label_y'] == 1]

    input_metrics_names = args.metrics_names
    if args.add_baselines:
        merged = cross_add_baselines(merged)
        input_metrics_names.extend(['overlap', 'bleu', 'bertscore'])

    for metric in input_metrics_names:
        correlations = bootstrap(merged, [0.05, 0.1, 0.15, 0.2, 0.25], metric, n_iter=1000)

        print(metric)
        print(np.mean(correlations))
        print(np.percentile(correlations, 2.5))
        print(np.percentile(correlations, 97.5))