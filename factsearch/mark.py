from dash import html
from itertools import chain
from jaro import jaro_winkler_metric
import numpy as np
import portion as P
import re
import string
from time import time

import fasttext


from utils.sem_distance import sem_distance
from utils.stopwords import StopWordList
from utils.tokenization import MorphoDiTaTokenizer

def merge_spans(spanrecs_list):
    # each `spanrecs` is a dict like `{'type': 'importance_sentence','spans': [{'span': [0,200), 'dist': 0.6117089}, {'span': [201,387), 'dist': 0.7668366}]}`
    # `spanrecs` later in `spanrecs_list` have a higher precedence, i.e., overwrite their predecessors 
    # example output: `[{'span': [0,7), 'type': 'claim_words'}, {'span': [7,71), 'dist': 0.6117089, 'type': 'importance_sentence'},` 
    # spanrecs_list = list(map(sort_spans, spanrecs_list))
    sprev = []
    for ps in spanrecs_list[0]["spans"]:
        ps["type"] = spanrecs_list[0]["type"]
        sprev.append(ps)
        
    for sr in spanrecs_list[1:]:
        type_ = sr["type"]
        snext = sr["spans"]
        for ns in snext:
            n = ns["span"]
            ns = ns.copy()
            ns["type"] = sr["type"]
            sprev2 = []
            for i, ps in enumerate(sprev):
                p = ps["span"]
                for interval in p-n:
                    if P.empty() == interval:
                        continue
                    ps = ps.copy()
                    ps["span"] = interval
                    sprev2.append(ps)
#                 print(n, p, len(p-n))
            sprev2.append(ns)
            sprev = sprev2

    firsts = [s["span"].lower for s in sprev] # now sort by span starting indices (there should be no overlaps)
    idxs = np.argsort(firsts)
    sprev = [sprev[idx] for idx in idxs]
    
    return sprev

def unify_spans(spanrecs):
    # reduces the number of spans by computing union of all spans - any span inside a larger one is removed
    spanrecs = spanrecs.copy()
    intervals = [s["span"] for s in spanrecs["spans"]]
    if len(intervals) == 0:
        return spanrecs
    assert len(spanrecs["spans"][0].keys()) == 1, "more information beyond `span` not supported yet!"
    union = intervals[0]
    for interval in intervals[1:]:
        union = union | interval
    # split union to the intervals
    spanrecs["spans"] = [{"span": interval} for interval in union] 
    return spanrecs

def sort_spans(spanrecs):
    spanrecs = spanrecs.copy()
    recs = spanrecs["spans"]
    firsts = [s["span"].lower for s in recs]
    idxs = np.argsort(firsts)
    recs = [recs[idx] for idx in idxs]
    spanrecs["spans"] = recs
    return spanrecs

def mark_retrieval_importance(model, claim, txt, select_top=None, importance_type="sentence", lang="cs"):
    # threshold is the number of most importanf words/sentences (based on distance) to return
    assert importance_type in ["word", "sentence"]
    mdtokenizer = MorphoDiTaTokenizer(lang=lang)
    if importance_type == "word":
        tokenization = list(mdtokenizer.tokenizeWords(txt, spans=True))
    else:
        tokenization = list(mdtokenizer.tokenizeSentences(txt, spans=True))

    # if there are less words/sentences in `txt` do not return any spans
    if select_top is not None and len(tokenization) <= select_top:
        return {"type": f"importance_{importance_type}", "spans": []}
        
    masked = []
    for _, span in tokenization:
        masked.append(txt[0:span.lower] + txt[span.upper:])
        
    distances = sem_distance(model, claim, masked)
    idxs = np.argsort(distances)
    if select_top is not None:
        idxs = idxs[:select_top]
    tokenization = [tokenization[idx] for idx in idxs]
    distances = distances[idxs]
    
    return {"type": f"importance_{importance_type}", "spans": [{"span": tokens[1], "dist": dist} for tokens, dist in zip(tokenization, distances)]}


class FastTextSimilarity:
    def __init__(self, model_path):
        self.model = fasttext.load_model(model_path)

    def similarity(self, a, b):
        va = self.model[a]
        vb = self.model[b]
        return np.dot(va, vb)/(np.linalg.norm(va)*np.linalg.norm(vb))


class EmphasizeClaimWords:
    def __init__(self, similarity=jaro_winkler_metric, min_chars=3, threshold=0.8, stopword_list="data/stopwords/czech.txt", lang="cs"):
        self.similarity = similarity
        self.min_chars = min_chars
        self.threshold = threshold
        self.tokenizer = MorphoDiTaTokenizer(lang=lang)
        self.stopword_list = StopWordList(stopword_list)
        self.lang = lang


    def emphasize(self, claim, doc):
        # returns list of `doc` words which should be emphasised w.r.t. the `claim`
        claim_words = set([w.lower() for w in self.tokenizer.tokenizeWords(claim) if not self.stopword_list.is_stopword(w) and len(w) >= self.min_chars])
        doc_words = set([w.lower() for w in self.tokenizer.tokenizeWords(doc) if not self.stopword_list.is_stopword(w) and len(w) >= self.min_chars])
    #     print(claim_words)
    #     print(doc_words)
        emp_words = set()
        for cw in claim_words:
            for dw in doc_words:
                dist = self.similarity(cw, dw)
    #             print(cw, dw, dist)
                if dist >= self.threshold:
                    emp_words.add(dw)
                
        return emp_words


    def emphasize_spans(self, claim, doc):
        # find words to emphasize
        emp_words = self.emphasize(claim, doc)
        # find them in `doc` and return the spans
        spans = []
        for w in emp_words:
            spans += [{"span": P.closedopen(a.start(), a.end())} for a in re.finditer(w, doc, re.IGNORECASE)]
        
        return unify_spans({"type": "claim_words", "spans": spans})
    