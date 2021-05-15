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


def length_baseline(in_path):
    df = pd.read_csv(in_path)

    sentences_length = []
    num_tokens = []
    num_tokens_no_punct = []

    for _, row in df.iterrows():
        text = row['response']
        sentences_length.append(len(text))

        doc = nlp(text)
        num_tokens.append(len(doc))

        no_punct_tokens = 0
        for tok in doc:
            if tok.pos_ != 'PUNCT':
                no_punct_tokens += 1

        num_tokens_no_punct.append(no_punct_tokens)

    print('Avg. sentence length (all chars):', np.mean(sentences_length))
    print('Avg. number of tokens:', np.mean(num_tokens))
    print('Avg. number of tokens, no punct', np.mean(num_tokens_no_punct))


def add_bertscore(pred, ref):
    P, R, F1 = score(pred, ref, lang="en", verbose=False, rescale_with_baseline=True)
    return F1.detach().numpy()


def add_baselines(df, out_path=''):
    bleu = []
    overlap = []

    all_responses = []
    all_knowledge = []

    for _, row in df.iterrows():
        response = row['response']
        knowledge = row['knowledge'].lower()

        # BLEU
        bleu.append(sacrebleu.corpus_bleu([response], [[knowledge]]).score)

        # Overlap
        overlap.append(f1_score(knowledge, response))

        all_responses.append(response)
        all_knowledge.append(knowledge)

    df['bleu'] = bleu
    df['overlap'] = overlap
    df['bertscore'] = add_bertscore(all_responses, all_knowledge)

    if out_path != '':
        df.to_csv(out_path)
    return df


def cross_add_baselines(df, out_path=''):
    dodeca_bleu = []
    memnet_bleu = []

    dodeca_overlap = []
    memnet_overlap = []

    dodeca_all = []
    memnet_all = []
    knowledge_all = []

    for _, row in df.iterrows():
        dodeca_response = row['response_x']
        memnet_response = row['response_y']
        knowledge = row['knowledge_x'].lower()

        # BLEU
        dodeca_bleu.append(sacrebleu.corpus_bleu([dodeca_response], [[knowledge]]).score)
        memnet_bleu.append(sacrebleu.corpus_bleu([memnet_response], [[knowledge]]).score)

        # Overlap
        dodeca_overlap.append(f1_score(knowledge, dodeca_response))
        memnet_overlap.append(f1_score(knowledge, memnet_response))

        dodeca_all.append(dodeca_response)
        memnet_all.append(memnet_response)
        knowledge_all.append(knowledge)

    df['bleu_x'] = dodeca_bleu
    df['bleu_y'] = memnet_bleu

    df['overlap_x'] = dodeca_overlap
    df['overlap_y'] = memnet_overlap

    df['bertscore_x'] = add_bertscore(dodeca_all, knowledge_all)
    df['bertscore_y'] = add_bertscore(memnet_all, knowledge_all)

    if out_path != '':
        df.to_csv(out_path)
    return df