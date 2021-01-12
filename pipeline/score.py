import re
import string
from collections import Counter

from bert_score import score


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\b(a|an|the|in|our)\b', ' ', text)
    return re.sub(' +', ' ', text).strip()


# def clean_text(s):  # From: https://github.com/W4ngatang/qags/blob/master/qa_utils.py
#     """Lower text and remove punctuation, articles and extra whitespace."""
#     def remove_articles(text):
#         return re.sub(r'\b(a|an|the|in|our)\b', ' ', text)
#
#     def white_space_fix(text):
#         return ' '.join(text.split())
#
#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return ''.join(ch for ch in text if ch not in exclude)
#
#     def lower(text):
#         return text.lower()
#
#     return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(a_gold, a_pred):
    if a_pred == '':
        return 0
    gold_toks = clean_text(a_gold).split()
    pred_toks = clean_text(a_pred).split()
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1_bert_score(a_gold, a_pred):
    P, R, F1 = score(a_pred, a_gold, lang="en", verbose=True)
    return F1.mean().item()
