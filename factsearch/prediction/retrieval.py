from dataclasses import dataclass
from datetime import date
from sklearn.metrics.pairwise import paired_cosine_distances
import numpy as np
import os
from collections import OrderedDict, defaultdict
import itertools
from pathlib import Path
import requests
import sqlite3
import time
from typing import List, Optional
import ujson
import unicodedata as ud

from aic_nlp_utils.fever import fever_detokenize

from utils.dbcache import DBCache
from utils.ner import load_ner_pipeline
from utils.sem_distance import sem_distance
from utils.sentence_transformers import load_model

from utils.stopwords import StopWordList
from utils.tokenization import MorphoDiTaTokenizer

from prediction.nli import SupportRefuteNEIModel

import logging
logger = logging.getLogger(__name__)

def load_embeddings(loc):
    if os.path.isfile(loc):
        return torch.load(loc).numpy()
    else: # dir
        files = [pjoin(loc, f) for f in sorted(os.listdir(loc))]
        logger.info("loading {} embedding files from {}".format(len(files), loc))
        return np.vstack([torch.load(f).numpy() for f in files])


class AbstractRetriever:
    pass


class AbstractFilteringRetriever(AbstractRetriever):
    pass


class FilteringRetriever(AbstractFilteringRetriever):
    """ Takes any DR `model` and tries to deliver exact number of documents in line with filtering options as required in `retrieve().`
    """    
    
    def __init__(self, db: DBCache, model, maxk=1000, max_repeats=3, initial_scale=5):
        """Constructor

        Args:
            db (DBCache): corpus
            model (_type_): DR model
            maxk (int, optional): Maxmimal number of documents the model can be asked for. Defaults to 1000.
            max_repeats (int, optional): The number of tries to get retrieve the target number of documents. Defaults to 3.
            initial_scale (int, optional): The model is is asked for `initial_scale` * k in the first iteration. Defaults to 5.
        """        
        self.db = db
        self.model = model
        self.maxk = maxk
        self.max_repeats = max_repeats
        self.initial_scale = initial_scale

    def retrieve(self, claim, k, max_titles=0, datemin=date.min, datemax=date.max, **kwargs):
        # does similar thing to search.py#filter_results but it is better to move it here
        # TODO: add support for datemin/datemax
        allres = []
        print(f"k={k} for: {claim}")
        for i in range(self.max_repeats):
            k2 = min(self.initial_scale * k * (i + 1), self.maxk)
            print(f"k2={k2}")
            results = self.model.retrieve(claim, k2, **kwargs)
            resblocksset = set()
            ntitles = 0
            for result in results:
                id_ = result["id"]
                if (ntitles < max_titles or not self.db.istitle(id_)) and (datemin <= self.db.id2date(id_).date() <= datemax):
                        blocktxt = self.db.get_block_text(id_)
                        if blocktxt not in resblocksset:
                            allres.append(result)
                            resblocksset.add(blocktxt) # no duplicates (there are often multiple copies of a single paragraph)
                            if self.db.istitle(id_):
                                ntitles += 1
            if len(allres) >= k or k2 == self.maxk:
                break
        return allres[0:k]


class TopNDocsDRQATwoTower(AbstractRetriever):
    def __init__(self, db: DBCache, premodel, model_dir):
        self.db = db
        self.premodel = premodel
        self.model = load_model(model_dir)
        self.model.eval()
        self.name = f"{self.premodel.name}+{os.path.basename(model_dir)}"
        
    def retrieve(self, claim, k, prek=500, preprocess=lambda t: t):
        st = time.time()
        doc_names_pre, doc_scores_pre = self.premodel.retrieve(claim, k=prek)
        logger.info(f"pre-selected {len(doc_names_pre)} documents")
        txts = []
        for did in doc_names_pre:
            txt = self.db.get_doc_text(did)
            if txt is None:
                logger.warning(f"document {did} has no text!")
            else:
                txts.append(txt)
        logger.info(f"imported text for {len(txts)} pre-selected documents")

        txts = [claim] + txts
        txts = map(preprocess, txts)
        x = [fever_detokenize(txt) for txt in txts]
        self.preduration = time.time() - st
        logger.info("TT input ready")

        st2 = time.time()
        y = self.model.encode(x, convert_to_numpy=True)
        logger.info("TT model evaluated")

        y_claim = np.tile(y[0:1], (y.shape[0]-1, 1))
        y_pages = y[1:]
        dists = paired_cosine_distances(y_claim, y_pages)
        inds = np.argsort(dists)[:k]
        doc_names, doc_scores = [doc_names_pre[i] for i in inds], [dists[i] for i in inds]
        self.modelduration = time.time() - st2
        self.duration = time.time() - st

        # logger.info(f"doc_names: {doc_names}")
        # logger.info(f"doc_scores: {doc_scores}")

        return [{"id": id_, "score": {"orig": score}, "search": "drqatt"} for id_, score in zip(doc_names, doc_scores)]


class TopNDocsTwoTowerFaiss(AbstractRetriever):
    def __init__(self, db_name, model_dir, embeddings, db_table="documents", norm='NFC', gpu=True, faissindex='Flat', onlytitles=False):
        import faiss

        self.model = load_model(model_dir)
        self.model.eval()
        embeddings = Path(embeddings)
        self.embeddings = load_embeddings(embeddings)
        n, embedding_dim = self.embeddings.shape
        logger.info("{} embeddings of dimension {}".format(n, embedding_dim))
        with sqlite3.connect(db_name) as conn:
            # index must be bypassed so page ids are read in a same ordering as text when page embedding were computed!
            sql = f"SELECT id FROM {db_table} WHERE bid = 0" if onlytitles else f"SELECT id FROM {db_table}"
            self.corpus_pages = sorted(map(lambda e: e[0], conn.execute(sql)))
        logger.info(f"imported {len(self.corpus_pages)} ids from {db_name}")
        assert n == len(self.corpus_pages)
        self.norm = norm

        logger.info("indexing using Faiss")
        index = faiss.index_factory(embedding_dim, faissindex, faiss.METRIC_INNER_PRODUCT)
        logger.info(" ... normalizing")
        faiss.normalize_L2(self.embeddings)
        if gpu:
            logger.info(" ... configuring GPU")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, index) # use gpu
        else:
            logger.info(" ... configuring CPU")
            self.index = index
        logger.info('training index...')
        self.index.train(self.embeddings)
        logger.info('adding embeddings')
        self.index.add(self.embeddings)

        faiss_file = Path(embeddings.parent, f"{embeddings.name}.faiss")
        # logger.info(f"saving to: {faiss_file}")
        # index = faiss.index_gpu_to_cpu(self.index)
        # faiss.write_index(index, faiss_file)
        logger.info(" ... done")

        self.name = os.path.basename(model_dir)
        
    def retrieve(self, claim, k, preprocess=lambda t: t, embeddings=False):
        import faiss
        st = time.time()
        claim = preprocess(ud.normalize(self.norm, claim))
        claim_embedding = self.model.encode([claim], show_progress_bar=False, convert_to_numpy=True)
        faiss.normalize_L2(claim_embedding)
        distances, idxs = self.index.search(claim_embedding, k)
        doc_names = [self.corpus_pages[i] for i in idxs[0]]
        doc_scores = distances[0].tolist()
        self.duration = time.time() - st
        # logger.info(f"doc_names: {doc_names}")
        # logger.info(f"doc_scores: {doc_scores}")
        if embeddings:
            return np.array(doc_names), np.array(doc_scores), claim_embedding[0], self.embeddings[idxs[0], :]
        return [{"id": id_, "score": {"orig": score}, "search": "drqattfaiss"} for id_, score in zip(doc_names, doc_scores)]
    

class Anserini(AbstractRetriever):
    # older code; Anserini now works over REST-API
    def __init__(self, model_dir="/mnt/data/factcheck/CTK/par5/index/anserini"):
        from pyserini.search import LuceneSearcher

        self.index = model_dir
        self.k1 = 0.6
        self.b = 0.5

        self.searcher = LuceneSearcher(str(self.index))
        
    def retrieve(self, query, k):
        assert isinstance(query, str), "Expected string input as a query!"
        print(f'Initializing BM25, setting k1={self.k1} and b={self.b}', flush=True)

        hits = self.searcher.search(query, k)
        ret_ids, ret_scores = [], []
        for i in range(len(hits)):
            ret_ids.append(hits[i].docid)
            ret_scores.append(hits[i].score)
        return [{"id": id_, "score": {"orig": score}, "search": "anserini"} for id_, score in zip(ret_ids, ret_scores)]


@dataclass
class ColBERTArgs:
    query_maxlen: int = 32
    similarity: str = 'cosine'
    rank: int = -1
    faiss_name = None
    faiss_depth: int = 512
    part_range = None
    depth: int = 100  # number of returned top-k documents

    amp: bool = True
    doc_maxlen: int = 180
    mask_punctuation: bool = True
    bsize: int = 32
    # dim: int = 64
    dim: int  = 32 # par6
    nprobe: int = 32
    partitions: int = 32768
    
    # index_root: str = "/mnt/data/factcheck/CTK/par5/colbert/indexes"
    index_root: str = "/home/drchajan/DATASETS/colbert/indexes" # par6

    # index_name: str = "ctk-fever-v2.1.L2.32x200k"
    # index_name: str = "ctk-fever-64dim.L2.32x200k"
    index_name: str = "ctk_par6-32.L2.32x200k" # par6
    
    # checkpoint: str = "/mnt/data/factcheck/CTK/par5/colbert/ctk-fever-v2.1/train.py/ctk-fever-v2.1.l2/checkpoints/colbert.dnn"
    # checkpoint: str = "/mnt/data/factcheck/CTK/par5/colbert/ctk-fever-64dim/train.py/ctk-fever-64dim.l2/checkpoints/colbert.dnn"
    checkpoint: str = "/home/drchajan/DATASETS/colbert/ctk_par6-32/train.py/ctk_par6-32.l2/checkpoints/colbert.dnn" # par6
    
    # idConvPath: str = "/mnt/data/factcheck/CTK/par5/interim/old-id2new-id.tsv"
    idConvPath: str = "/mnt/data/ctknews/factcheck/par6/interim/old-id2new-id.tsv"
    
    qrels=None

class ColBERT:
    def __init__(self, args: ColBERTArgs = None):
        from colbert.evaluation.loaders import load_colbert
        from colbert.modeling.inference import ModelInference
        from colbert.ranking.rankers import Ranker
        from colbert.indexing.faiss import get_faiss_index_name
        self.args = args if args else ColBERTArgs()
        self.args.colbert, self.args.checkpoint = load_colbert(self.args)
        self.args.index_path = os.path.join(self.args.index_root, self.args.index_name)
        if self.args.faiss_name is not None:
            self.args.faiss_index_path = os.path.join(self.args.index_path, self.args.faiss_name)
        else:
            self.args.faiss_index_path = os.path.join(self.args.index_path, get_faiss_index_name(self.args))

        self.inference = ModelInference(self.args.colbert, amp=self.args.amp)
        self.ranker = Ranker(self.args, self.inference, faiss_depth=self.args.faiss_depth)
        
        with open(self.args.idConvPath) as fr:
            self.colbId2parId = {int(l.split('\t')[1].strip()): l.split('\t')[0] for l in fr if l.strip()}
        
    def idx2par(self, ids):
        """Converts ColBERT inner indices to paragraph indices"""
        ret = [self.colbId2parId[i] if int(i) in self.colbId2parId else None for i in ids]
        return ret
        
    def retrieve(self, query, k=None, convert_ids=True):
        assert isinstance(query, str), "Expected string input as a query!"
        k = k if k else self.args.depth
        Q = self.ranker.encode([query])
        pids, scores = self.ranker.rank(Q)
        
        ret_ids, ret_scores = [], []
        for pid, score in itertools.islice(zip(pids, scores), k):
            ret_ids.append(pid)
            ret_scores.append(score)
        if convert_ids:
            ret_ids = self.idx2par(ret_ids)
        # return ret_ids, ret_scores
       
        return [{"id": id_, "score": {"orig": score}, "search": "colbert"} for id_, score in zip(ret_ids, ret_scores)]

class RESTRetrieval():
    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url
        print(f"initializing RESTRetrieval '{name}' at: {url}")

    def retrieve(self, query: str, k: int):
        response = requests.post(self.url, 
                   headers={"accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
                   data={"query": query, "k": k})
        if response.status_code == 200:
            data = ujson.loads(response.content)["retrieved"]
        else:
            print(f"RESTRetrieval failed with code: {response.status_code}")
            data = {'ids': [], 'scores': [], 'duration_s': 0.0}
        return [{"id": id_, "score": {"orig": score}, "search": self.name} for id_, score in zip(data["ids"], data["scores"])]

class MetaRetriever(AbstractFilteringRetriever):
    def __init__(self, 
                 db: DBCache, 
                 semmodel: AbstractFilteringRetriever, 
                 kwmodel:AbstractFilteringRetriever, 
                 nlimodels, 
                 scoring_model=None, 
                 max_titles=0, 
                 sort_nli=True):
        """_summary_

        Args:
            db (DBCache): _description_
            semmodel (AbstractFilteringRetriever): _description_
            kwmodel (AbstractFilteringRetriever): _description_
            nlimodels (_type_): _description_
            scoring_model (_type_, optional): _description_. Defaults to None.
            max_titles (int, optional): Limit on number of retrieved title-only documents. Defaults to 0.
            sort_nli (bool, optional): Resorts the results so they come out in order SUPPORTS, REFUTES, NEI. Defaults to True.
        """        
        self.db = db
        self.semmodel = semmodel
        self.kwmodel = kwmodel
        assert all(isinstance(nm, SupportRefuteNEIModel) for nm in nlimodels), f"wrong types of nli models: {[nm for nm in nlimodels]}"
        self.nlimodels = nlimodels
        self.max_titles = max_titles
        self.scoring_model = scoring_model # should be semantic model
        self.sort_nli = sort_nli

    def retrieve(self, query: str, k: int, **kwargs):
        """_summary_

        Args:
            query (str): query string
            k (int): number of document to retrieve

        Returns:
            _type_: _description_
        """        
        semres = self.semmodel.retrieve(query, k, **kwargs)
        kwres = self.kwmodel.retrieve(query, k, **kwargs)
        
        # now merge them
        allres = semres.copy()
        allidsset = set([r["id"] for r in allres])
        for kwitem in kwres:
            if kwitem["id"] not in allidsset:
                allres.append(kwitem)
                allidsset.add(kwitem["id"])
        allids = [r["id"] for r in allres]
        print(f"retrieved SEM: {len(semres)}, KW: {len(kwres)}, MERGED: {len(allids)}")

        # compute common score for both semantic and keyword search
        if self.scoring_model is not None:
            common_scores = sem_distance(self.scoring_model, query, allids, preprocess=fever_detokenize)
            for res, common in zip(allres, common_scores):
                res["score"]["orig"] = common

        # prepare (context, query) pairs for NLI models
        print("MetaRetriever: WARNING: do not use fever_detokenize be more general")
        nli_sentences = [[fever_detokenize(self.db.get_block_text(block)), fever_detokenize(query)] for block in allids]
        nlires = [nlimodel.predict(nli_sentences, order="rsn", apply_softmax=True) for nlimodel in self.nlimodels]

        # compute mean probabilities of the NLI ensemble
        nlires = np.mean(nlires, axis=0) # NLI ensemble predictions
        confs = np.max(nlires[:, 0:2], axis=1) # highest confidences for either REFUTES or SUPPORTS class for each document
        idxs = np.argsort(confs)[::-1] # sort indices by descending confidence
        idxs = idxs[:k]

        # get top-k documents only (+ NLI model confidences)
        allres = [allres[i] for i in idxs]
        nlires = nlires[idxs, :]

        if self.sort_nli:
            print("CHECK and FIX: watch for stability")
            classes = np.argmax(nlires, axis=1)
            # refutes, supports, nei -> supports, refutes, nei
            c2c = {0:1, 1:0, 2:2}
            classes = [c2c[c] for c in classes]
            idxs = np.argsort(classes, kind="stable")
            allres = [allres[i] for i in idxs]
            nlires = nlires[idxs, :]


        for res, nli in zip(allres, nlires):
            res["nli"] = {"refutes": nli[0], "supports": nli[1], "nei": nli[2]}

        return allres


class DirectedRetriever(AbstractFilteringRetriever):
    def __init__(self, 
                 db: DBCache,
                 lang: str,
                 retrieval_models: List[AbstractFilteringRetriever], 
                 similarity: str, 
                 stopword_list: str,
                 ner_model_name: Optional[str] = None,
                 fast_text_model: Optional[str] = None,
                 similarity_threshold: float = 0.0,
                 similarity_min_chars: int = 3,
                 ner_weight: float = 2.0
                 ):
        """_summary_

        Args:
            db (DBCache): _description_
        """        
        self.db = db
        self.retrieval_models = retrieval_models
        assert similarity in ["fasttext"], f"Unsuported similarity: {similarity}"
        if similarity == "fasttext":
            import fasttext
            print(f"DirectedRetriever: loading FastText similarity model: {fast_text_model}")
            fasttext_model = fasttext.load_model(fast_text_model)

            def fasttext_similarity(a, b):
                va = fasttext_model[a]
                vb = fasttext_model[b]
                return np.dot(va, vb)/(np.linalg.norm(va)*np.linalg.norm(vb))
            
            self.similarity_func = fasttext_similarity

        self.stopword_list = StopWordList(stopword_list)
        self.ner_pipeline = load_ner_pipeline(lang=lang, ner_model_name=ner_model_name)
        self.word_tokenizer = MorphoDiTaTokenizer(lang=lang)
        self.similarity_threshold = similarity_threshold
        self.similarity_min_chars = similarity_min_chars
        self.ner_weight = ner_weight


    def directed_similarity(self, claim: str, doc: str, verbose: bool=False):
        claim_words = set([w.lower() for w in self.word_tokenizer.tokenizeWords(claim) if not self.stopword_list.is_stopword(w) and len(w) >= self.similarity_min_chars])
        claim_entities = list(set([e[0].lower() for e in self.ner_pipeline(claim)]))

        claim_word_weights = {}
        for cw in claim_words:
            for ce in claim_entities:
                if cw in ce: # even substring is ok, NERs are detected from full claim and can span multiple words
                    claim_word_weights[cw] = self.ner_weight
                    break

        for ce in claim_entities:
            claim_words.add(ce)
            for cw in claim_words:
                if cw in ce:
                    claim_word_weights[ce] = self.ner_weight
                else:
                    claim_word_weights[cw] = 1.0

        # claim_word_weights = {e[0].lower():  ner_weight for e in claim_entities}

        doc_words = set([w.lower() for w in self.word_tokenizer.tokenizeWords(doc) if not self.stopword_list.is_stopword(w) and len(w) >= self.similarity_min_chars])
        
        # needed if claim entities are added to claim_words
        doc_entities = set([e[0].lower() for e in self.ner_pipeline(doc)])
        doc_words.update(doc_entities)

        # find closest claim word for each doc word
        dw2cw_sims = defaultdict(dict)
        for dw in doc_words:
            min_cw, min_sim = None, 0.0
            for cw in claim_words:
                sim = claim_word_weights.get(cw, 1.0) * self.similarity_func(cw, dw)
                if sim > min_sim:
                    min_sim = sim
                    min_cw = cw
            if min_sim >= self.similarity_threshold:
                # print(dw, min_cw, min_sim)
                dw2cw_sims[min_cw][dw] = min_sim

        for k, v in dw2cw_sims.items():
            dw2cw_sims[k] = sorted([(word, sim) for word, sim in  dw2cw_sims[k].items()], key=lambda e: -e[1])

        cw_sims = {cw: np.mean([e[1] for e in dw2cw_sims[cw]] if cw in dw2cw_sims else 0.0) for cw in claim_words}

        sim = np.mean(list(cw_sims.get(cw, 0.0) for cw in claim_words))

        if verbose:
            print("claim_word_weights", claim_word_weights)
            print("doc_words", doc_words)
            for cw in claim_words:
                print(cw, dw2cw_sims[cw])
                print(cw, cw_sims[cw])
        return sim


    def retrieve(self, query: str, k: int, **kwargs):
        ret = []
        for model in self.retrieval_models: # NOTE: this returns k x number of retrieval models results!!!
            retrieved = model.retrieve(query, k=k)
            ret += retrieved
        txts = [self.db.get_block_text(e["id"]) for e in ret]
        scores = [self.directed_similarity(query, txt, verbose=False) for txt in txts]

        res = [{"id": r["id"], "score": {"orig": score}, "search": "directed"} for r, score in zip(ret, scores)]

        res.sort(key=lambda e: -e["score"]["orig"])
        return res