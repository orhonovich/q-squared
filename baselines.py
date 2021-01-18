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


def add_baselines(in_path, out_path):
    df = pd.read_csv(in_path)
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

    # Bertscore
    P, R, F1 = score(all_responses, all_knowledge, lang="en", verbose=False, rescale_with_baseline=True)

    df['bleu'] = bleu
    df['overlap'] = overlap
    df['bertscore'] = F1.detach().numpy()

    df.to_csv(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, required=True, help="Path to a csv file containing q^2 scores.")
    parser.add_argument("--outfile", type=str, required=True, help="Path to an output file")
    args = parser.parse_args()

    add_baselines(args.infile, args.outfile)