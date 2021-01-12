import argparse
import json
import spacy

nlp = spacy.load('en_core_web_sm')

contractions_conversion_dict = {"it ' s": "it is", "he ' s": "he is", "she ' s": "she is", "we ' re": "we are",
                                "they ' re": "they are", "you ' re": "you are", "i ' ve": "i have",
                                "we ' ve": "we have", "they ' ve": "they have", "you ' ve": "you have",
                                "i ' d": "i would",  "it ' d": "it would", "he ' d": "he would",
                                "she ' d": "she would",  "it ' ll": "it will", "he ' ll": "he will",
                                "she ' ll": "she will", "i ' m": "i am", "haven ' t": "have not",
                                "didn ' t": "did not", "don ' t": "do not", "wasn ' t": "was not"}


def convert_contractions(text):
    for key in contractions_conversion_dict.keys():
        text = text.replace(key, contractions_conversion_dict[key])
    return text


def add_missing_punct(text):
    if len(text) == 0:
        return text
    doc = nlp(text)
    if doc[-1].pos_ != 'PUNCT' or doc[-1].text == "\"":
        return text + ' .'
    return text


def clean_data(in_path, out_path):
    with open(in_path) as json_file:
        episodes = json.load(json_file, encoding='utf-8')['all_episodes']
    for episode in episodes:
        for turn in episode['turns']:
            turn['text'] = add_missing_punct(turn['text'])
            turn['model_response'] = convert_contractions(add_missing_punct(turn['model_response']))
            turn['gold_response'] = add_missing_punct(turn['gold_response'])

    all_episodes = {'all_episodes': episodes}
    with open(out_path, 'w', encoding='utf8') as outfile:
        json.dump(all_episodes, outfile, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, required=True, help="Path to a json file containing source documents.")
    parser.add_argument("--outfile", type=str, required=True, help="Path to an output file")
    args = parser.parse_args()
    add_punct(args.infile, args.outfile)