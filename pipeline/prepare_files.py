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

from collections import Counter
import numpy as np
import pandas as pd
import spacy

import sacrebleu
from bert_score import score

from run_pipeline import get_response_score

nlp = spacy.load("en_core_web_sm")


def get_tokens(text):
    doc = nlp(text)
    tokens = [tok.text.lower() for tok in doc if not tok.is_stop and not tok.is_punct]
    return tokens


def f1_score(gold, pred):
    gold_toks = get_tokens(gold)
    pred_toks = get_tokens(pred)

    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def add_bertscore(pred, ref):
    P, R, F1 = score(pred, ref, lang="en", verbose=False, rescale_with_baseline=True)
    return F1.detach().numpy()


def add_baselines(df):
    dodeca_bleu = []
    memnet_bleu = []

    dodeca_overlap = []
    memnet_overlap = []

    dodeca_all = []
    memnet_all = []
    knowledge_all = []

    for _, row in df.iterrows():
        dodeca_response = row['dodeca_response']
        memnet_response = row['memnet_response']
        knowledge = row['knowledge'].lower()

        # BLEU
        dodeca_bleu.append(sacrebleu.corpus_bleu([dodeca_response], [[knowledge]]).score)
        memnet_bleu.append(sacrebleu.corpus_bleu([memnet_response], [[knowledge]]).score)

        # Overlap
        dodeca_overlap.append(f1_score(knowledge, dodeca_response))
        memnet_overlap.append(f1_score(knowledge, memnet_response))

        dodeca_all.append(dodeca_response)
        memnet_all.append(memnet_response)
        knowledge_all.append(knowledge)

    df['dodeca_bleu'] = dodeca_bleu
    df['memnet_bleu'] = memnet_bleu

    df['dodeca_overlap'] = dodeca_overlap
    df['memnet_overlap'] = memnet_overlap

    df['dodeca_bertscore'] = add_bertscore(dodeca_all, knowledge_all)
    df['memnet_bertscore'] = add_bertscore(memnet_all, knowledge_all)

    return df


def prepare_validation(in_path, out_path):
    df = pd.read_csv(in_path)

    dodeca_q = []
    memnet_q = []

    for _, row in df.iterrows():
        dodeca_response = row['dodeca_response']
        memnet_response = row['memnet_response']
        knowledge = row['knowledge']

        dodeca_score = get_response_score(dodeca_response, knowledge, 'greedy', single=True, remove_personal=False)
        if dodeca_score >= 0:
            dodeca_q.append(dodeca_score)
        else:
            dodeca_q.append(-1)

        memnet_score = get_response_score(memnet_response, knowledge, 'greedy', single=True, remove_personal=False)
        if memnet_score >= 0:
            memnet_q.append(memnet_score)
        else:
            memnet_q.append(-1)


    df['dodeca_q2'] = dodeca_q
    df['memnet_q2'] = memnet_q
    df = df[df.dodeca_q2 >= 0]
    df = df[df.memnet_q2 >= 0]

    df = add_baselines(df)
    df.to_csv(out_path)