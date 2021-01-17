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
import json
from allennlp.predictors.predictor import Predictor
import allennlp_models.coref
import re

blacklist = ["i", "me", "my", "you", "your", "we", "our"]
det = ["the", "a", "an"]

MARK_REMOVE = '#####'


predictor = \
    Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")


def ordered_sublist(l1, l2):
    """
    Given two lists, l1 and l2, where l1 is a sublist of l2, finds the index in l2 where l1 begins.
    if l1 is not an ordered sublist of l2, returns -1.
    """
    length = len(l1)
    for i in range(len(l2) - length + 1):
        if all(l1[j] == l2[j + i] for j in range(length)):
            return i
    return -1


def get_resolved_turn(previous, text):
    """
    Get dialogue text that possibly contains pronouns and a list of the previous dialogue turns,
    and run coreference resolution on the current dialogue turn.
    """
    if len(text) == 0:
        return text
    concat = ''
    for turn in previous:
        concat += turn
        concat += ' '
    concat += text

    # First, get the tokens of the current dialogue turn.
    try:
        only_curr = predictor.predict(document=text)
    except:
        print("Exception while trying to resolve the current dialogue text")
        return text
    only_curr_tokens = only_curr['document']

    res = predictor.predict(document=concat)
    document = res['document']
    clusters = res['clusters']

    # Find the index in which the current dialogue turn begins
    # in the list of all tokens so far (previous and current turn)
    current_start_idx = ordered_sublist(only_curr_tokens, document)

    for cluster in clusters:
        centroid_start = cluster[0][0]
        centroid_end = cluster[0][1]
        centroid = ' '.join(document[centroid_start:centroid_end+1])

        # Resolve
        if centroid not in blacklist and centroid not in det:
            for span in cluster:
                span_start = span[0]
                if span_start >= current_start_idx != centroid_start and document[span_start].lower() not in blacklist:
                    span_end = span[1]
                    if not (span_end == span_start and document[span_start].lower() in det):
                        document[span_start] = centroid
                        span_start += 1
                        while span_start <= span_end:
                            document[span_start] = MARK_REMOVE
                            span_start += 1

    replaced = document[current_start_idx:]
    replaced = [t for t in replaced if t != MARK_REMOVE]

    resolved_str = ' '.join(replaced)
    resolved_str = re.sub(' +', ' ', resolved_str)
    return resolved_str


def dialogue_coref(conversation):
    """
    Get a full conversation and run coreference resolution on all turns.
    """
    all_resolved = []
    curr_resolved = {}
    turns_so_far = []
    for i, turn in enumerate(conversation['turns']):
        curr_resolved['text'] = get_resolved_turn(turns_so_far, turn['text'])
        turns_so_far.append(turn['text'])
        curr_resolved['model_response'] = get_resolved_turn(turns_so_far, turn['model_response'])
        curr_resolved['gold_response'] = get_resolved_turn(turns_so_far, turn['gold_response'])

        # The context given to the model in each step uses only gold responses
        turns_so_far.append(turn['gold_response'])

        all_resolved.append(curr_resolved)
        curr_resolved = {}

    return all_resolved


def knowledge_coref(conversation):
    all_resolved = []
    curr_resolved = {}
    turns_so_far = []
    for i, turn in enumerate(conversation['turns']):
        curr_resolved['gold_knowledge'] = get_resolved_turn(turns_so_far, turn['gold_knowledge'])
        turns_so_far.append(turn['gold_knowledge'])
        all_resolved.append(curr_resolved)
        curr_resolved = {}

    return all_resolved


def coref_resolve_knowledge(in_path, out_path):
    resolved_episodes = []
    with open(in_path) as json_file:
        episodes = json.load(json_file, encoding='utf-8')['all_episodes']

    for episode in episodes:
        curr_episode = episode
        resolved_turns = knowledge_coref(episode)
        for i, turn in enumerate(resolved_turns):
            curr_episode['turns'][i]['gold_knowledge'] = turn['gold_knowledge']

        resolved_episodes.append(curr_episode)
    all_episodes = {'all_episodes': resolved_episodes}
    with open(out_path, 'w', encoding='utf8') as out_file:
        json.dump(all_episodes, out_file, indent=4, ensure_ascii=False)


def coref_resolve_dialogue(in_path, out_path):
    resolved_episodes = []
    with open(in_path) as json_file:
        episodes = json.load(json_file, encoding='utf-8')['all_episodes']

    for episode in episodes:
        curr_episode = episode
        resolved_turns = dialogue_coref(episode)
        for i, turn in enumerate(resolved_turns):
            curr_episode['turns'][i]['text'] = turn['text']
            curr_episode['turns'][i]['model_response'] = turn['model_response']
            curr_episode['turns'][i]['gold_response'] = turn['gold_response']

        resolved_episodes.append(curr_episode)
    all_episodes = {'all_episodes': resolved_episodes}
    with open(out_path, 'w', encoding='utf8') as out_file:
        json.dump(all_episodes, out_file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, required=True,
                        help="Path to file containing text and answer candidate pairs.")
    parser.add_argument("--outfile", type=str, required=True, help="Path to an output file")
    args = parser.parse_args()
    coref_resolve_dialogue(args.infile, args.outfile)