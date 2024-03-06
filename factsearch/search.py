from collections import OrderedDict
import numpy as np
from scipy.sparse import hstack

from time import time
import unicodedata as ud

from prediction.retrieval import FilteringRetriever

import logging
logger = logging.getLogger(__name__)

def fact_search(db, model, claim_txt, k, max_titles, datemin, datemax):
    st = time()
    did2ids = OrderedDict()
    results = model.retrieve(claim_txt, k, max_titles=max_titles, datemin=datemin, datemax=datemax)
    # print(results)
    for result in results:
        id_ = result["id"]
        score = result["score"]["orig"]
        if db.hasid(id_):
            did = db.id2did(id_)
            if did not in did2ids:
                did2ids[did] = OrderedDict()
            did2ids[did][id_] = result
    return did2ids, {"fact_search_time": time()-st}


def scores_tfidf(ranker, claim, txts):
    c = ranker.text2spvec(claim)
    D = hstack([ranker.text2spvec(txt).T for txt in txts])
    return np.array((c*D).todense()).reshape(-1)


def filter_results(results, db, k, titles, datemin, datemax):
    st = time()
    # result items (documents) are sorted in decreasing order by best scoring block (paragraph)
    did2ids = []
    scores = []
    for did, blocks in results.items():
        blockscores = [blocks[id_]['score']
                       for id_ in blocks if titles or not db.istitle(id_)]

        if len(blockscores) == 0:
            continue
        if datemin <= db.did2date(did).date() <= datemax:
            did2ids.append((did, blocks))
            scores.append(np.max(blockscores))
    idxs = np.argsort(np.array(scores))[::-1]

    return OrderedDict([did2ids[i] for i in idxs[:k]]), {"filtering_time": time()-st}


def score_blocks(model, claim_txt, id2txt):
    txts = list(id2txt.values())
    scores = model["scoref"](claim_txt, txts)
    return OrderedDict([(id_, score) for id_, score in zip(list(id2txt.keys()), scores)])


def extract_id_info(did2ids, name="score"):
    return np.array([info[name] for doc in did2ids.values() for info in doc.values()])


def extract_doc_dates(db, did2ids):
    return np.array([db.did2date(did) for did in did2ids.keys()])
