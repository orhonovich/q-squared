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

import numpy as np
import pandas as pd

from allennlp.predictors.predictor import Predictor
import allennlp_models.pair_classification

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/snli_roberta-2020.06.09.tar.gz",
                                predictor_name="textual_entailment")


NO_ANS = '[CLS]'
NO_NLI = 'NO_NLI'
NO_Q = -1
ENTAILMENT_SCORE = 1
CONTRADICTION_SCORE = 0
NEUTRAL_SCORE = 0.5


def fallback(response, knowledge):
    res = predictor.predict(
        premise=knowledge,
        hypothesis=response
    )

    nli_label = res['label']

    if nli_label == 'entailment':  # If entails, the score is 1
        return ENTAILMENT_SCORE
    elif nli_label == 'contradiction':  # If contradicts, the score is 0
        return CONTRADICTION_SCORE
    else:
        return NEUTRAL_SCORE


def get_nli_label(question, cand, evidence_ans):
    premise = question + ' ' + evidence_ans + '.'
    hypothesis = question + ' ' + cand + '.'

    res = predictor.predict(
        premise=premise,
        hypothesis=hypothesis
    )

    return res['label']


def scores_with_nli(in_path):
    nli_scores = []
    f1_scores = []
    responses_idx = []

    df = pd.read_csv(in_path)

    idx = 0
    for _, row in df.iterrows():
        idx += 1
        responses_idx.append(idx)
        f1_score = row['score']

        evidence_answer = str(row['knowledge_ans'])

        nli_score = f1_score

        # Use NLI to determine answer similarity.
        # This is only applicable for responses that had at least one valid question generated

        if 0 <= f1_score < 1 and NO_ANS not in evidence_answer and evidence_answer != '' and evidence_answer != 'nan':
            f1_scores.append(f1_score)
            # If the score is 1, there is a full overlap between the
            # candidate and the predicted answer, so the score is 1
            # If there is no answer - can't run NLI, keep the original score (0)

            nli_label = get_nli_label(str(row['question']), str(row['cand']), evidence_answer)

            if nli_label == 'entailment':  # If entails, the score is 1
                nli_score = ENTAILMENT_SCORE
            elif nli_label == 'contradiction':  # If contradicts, the score is 0
                nli_score = CONTRADICTION_SCORE

        # Add fallback NLI to responses that are not covered by Q2 (no questions generated)
        elif f1_score == NO_Q:
            nli_fallback = fallback(str(row['response']), str(row['knowledge']).lower())
            nli_score = nli_fallback
            f1_scores.append(nli_fallback)
        else:
            f1_scores.append(f1_score)

        nli_scores.append(nli_score)

    df['q2_score'] = nli_scores
    df['q2_no_nli'] = f1_scores
    return df


def aggregate_per_response(df, out_path='', cross=False):
    f1_scores_by_id = dict()
    nli_scores_by_id = dict()
    knowledge_by_id = dict()
    response_by_id = dict()
    label_by_id = dict()

    for _, row in df.iterrows():
        idx = row['id']
        f1_score = row['q2_no_nli']
        nli_score = row['q2_score']

        if idx in f1_scores_by_id:
            f1_scores_by_id[idx].append(f1_score)
            nli_scores_by_id[idx].append(nli_score)
        else:
            f1_scores_by_id[idx] = [f1_score]
            nli_scores_by_id[idx] = [nli_score]
            response_by_id[idx] = row['response']
            knowledge_by_id[idx] = row['knowledge']
            if cross:
                label_by_id[idx] = row['label']

    mean_f1_scores = []
    mean_nli_scores = []
    responses = []
    knowledge = []
    labels = []

    for idx in f1_scores_by_id.keys():
        mean_f1_scores.append(np.mean(f1_scores_by_id[idx]))
        mean_nli_scores.append(np.mean(nli_scores_by_id[idx]))
        responses.append(response_by_id[idx])
        knowledge.append(knowledge_by_id[idx])
        if cross:
            labels.append(label_by_id[idx])

    print('Q2:', np.mean(mean_nli_scores))
    print('Q2, no nli:', np.mean(mean_f1_scores))
    data = {'id': list(f1_scores_by_id.keys()), 'response': responses, 'knowledge': knowledge,
            'Q2_no_nli': mean_f1_scores, 'Q2': mean_nli_scores}

    res_df = pd.DataFrame(data=data)
    if cross:
        res_df['label'] = labels

    if out_path != '':
        res_df.to_csv(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, required=True,
                        help="Path to a csv file containing token-level f1 scores.")
    parser.add_argument("--outfile", type=str, default='', required=False, help="Path to an output file")
    parser.add_argument("--cross", default=False, action="store_true", help="Whether to save all pipeline steps")
    args = parser.parse_args()

    with_nli_df = scores_with_nli(args.infile)
    aggregate_per_response(with_nli_df, args.outfile, args.cross)