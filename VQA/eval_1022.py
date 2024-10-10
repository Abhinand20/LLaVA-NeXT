import numpy as np
import os
import json, time, copy
import math

from tqdm import tqdm
import random
import pickle
from datetime import datetime
from pytz import timezone
from word2number import w2n
import string, re
from collections import Counter, defaultdict
from pprint import pprint
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner","textcat","parser"])
np.set_printoptions(precision=4)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--Qcate_breakdown", type=str, default='["all"]')
parser.add_argument("--file", type=str, default="img_queries_VLP_vinvl_combinedTraining_val_beam5_img_step13_cleaned.tsv")
parser.add_argument('--no_norm', action='store_true')
parser.add_argument('--dir', type=str, default="")
parser.add_argument('--output_idx', type=int, default=0)
args = parser.parse_args()

import sys
# sys.path.append("/home/yingshac/CYS/WebQnA/VLP/BARTScore")
# from bart_score import BARTScorer

# bart_scorer_ParaBank = BARTScorer(device='cuda', checkpoint='facebook/bart-large-cnn')
# bart_scorer_ParaBank.load(path='bart_score.pth') # Please change the path to bart.pth


def detectNum(l):
    result = []
    for w in l:
        try: result.append(str(int(w)))
        except: pass
    return result
def toNum(word):
    if word == 'point': return word
    try: return w2n.word_to_num(word)
    except:
        return word

def normalize_text(s):
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text): # additional: converting numbers to digit form
        return " ".join([str(toNum(w)) for w in text.split()])

    def remove_punc(text):
        exclude = set(string.punctuation) - set(['.'])
        text1 = "".join(ch for ch in text if ch not in exclude)
        return re.sub(r"\.(?!\d)", "", text1) # remove '.' if it's not a decimal point

    def lower(text):
        return text.lower()
    
    def lemmatization(text):
        return " ".join([token.lemma_ for token in nlp(text)])

    if len(s.strip()) == 1:
        # accept article and punc if input is a single char
        return white_space_fix(lower(s))
    elif len(s.strip().split()) == 1: 
        # accept article if input is a single word
        return lemmatization(white_space_fix(remove_punc(lower(s))))

    return lemmatization(white_space_fix(remove_articles(remove_punc(lower(s)))))

# VQA Eval (SQuAD style EM, F1)
def compute_vqa_metrics(cands, a, exclude="", domain=None):
    if len(cands) == 0: return (0,0,0)
    bow_a = normalize_text(a).split()
    F1 = []
    EM = 0
    RE = []
    PR = []
    e = normalize_text(exclude).split()
    for c in cands:
        bow_c = [w for w in normalize_text(c).split() if not w in e]
        if domain == {"NUMBER"}: 
            bow_c = detectNum(bow_c)
            bow_a = detectNum(bow_a)
        elif domain is not None: 
            bow_c = list(domain.intersection(bow_c))
            bow_a = list(domain.intersection(bow_a))
        if bow_c == bow_a:
            EM = 1
        common = Counter(bow_a) & Counter(bow_c)
        num_same = sum(common.values())
        if num_same == 0:
            return (0,0,0,0,0)
        precision = 1.0 * num_same / len(bow_c)
        recall = 1.0 * num_same / len(bow_a)
        RE.append(recall)
        PR.append(precision)

        f1 = 2*precision*recall / (precision + recall + 1e-5)
        F1.append(f1)
    
    PR_avg = np.mean(PR)
    RE_avg = np.mean(RE)
    F1_avg = np.mean(F1)
    F1_max = np.max(F1)
    return (F1_avg, F1_max, EM, RE_avg, PR_avg)

color_set= {'orangebrown', 'spot', 'yellow', 'blue', 'rainbow', 'ivory', 'brown', 'gray', 'teal', 'bluewhite', 'orangepurple', 'black', 'white', 'gold', 'redorange', 'pink', 'blonde', 'tan', 'turquoise', 'grey', 'beige', 'golden', 'orange', 'bronze', 'maroon', 'purple', 'bluere', 'red', 'rust', 'violet', 'transparent', 'yes', 'silver', 'chrome', 'green', 'aqua'}
shape_set = {'globular', 'octogon', 'ring', 'hoop', 'octagon', 'concave', 'flat', 'wavy', 'shamrock', 'cross', 'cylinder', 'cylindrical', 'pentagon', 'point', 'pyramidal', 'crescent', 'rectangular', 'hook', 'tube', 'cone', 'bell', 'spiral', 'ball', 'convex', 'square', 'arch', 'h', 'cuboid', 'step', 'rectangle', 'dot', 'oval', 'circle', 'star', 'crosse', 'crest', 'octagonal', 'cube', 'triangle', 'semicircle', 'domeshape', 'obelisk', 'corkscrew', 'curve', 'circular', 'xs', 'slope', 'pyramid', 'round', 'bow', 'straight', 'triangular', 'heart', 'fork', 'teardrop', 'fold', 'curl', 'spherical', 'diamond', 'keyhole', 'conical', 'dome', 'sphere', 'bellshaped', 'rounded', 'hexagon', 'flower', 'globe', 'torus'}
yesno_set = {'yes', 'no'}
