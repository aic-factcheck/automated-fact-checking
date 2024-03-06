import numpy as np
import torch
import json
import ujson

from argparse import ArgumentParser
from collections import defaultdict, OrderedDict, Counter
from dataclasses import dataclass
import datetime as dt
from itertools import chain
import os
import pathlib
from pathlib import Path
import pandas as pd
import unicodedata as ud
from time import time
from typing import Dict, Type, Callable, List

from dash import Dash
import dash_bootstrap_components as dbc

from jaro import jaro_winkler_metric

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from mark import EmphasizeClaimWords, FastTextSimilarity
from utils.sentence_transformers import load_model
from utils.dbcache import DBCache
from prediction.retrieval import FilteringRetriever, Anserini, ColBERTArgs, ColBERT, RESTRetrieval, MetaRetriever, DirectedRetriever
from prediction.nli import SupportRefuteNEIModel # Make OBSOLETE

from factsearch import dash_build_app


def dummy_zero_score(claim, txts):
    return np.zeros(len(txts))

def dummy_nan_score(claim, txts):
    a = np.empty(len(txts))
    a[:] = np.nan
    return a

def populate_cfg(cfg_path, db=None, opts:List[str]=None):

    with open(cfg_path, 'r') as f:
        cfg = ujson.load(f)

        # replace options from the imported config based on `opts`
        if opts:
            for k, v in opts.items():
                opt_path = k.split(".")
                path = cfg
                for node_name in opt_path[:-1]:
                    path = path[node_name]
                path[opt_path[-1]] = v
    
        print("------ Load DB")
        if db is None:
            cfg["db"] = DBCache(Path(cfg["db_path"]), default_date=cfg.get("default_date", None), uni_normalize=cfg.get("uni_normalize", None))
        else:
            cfg["db"] = db


        print("------ Scoring models")
        for mname, v in cfg["scoring_models"].items():
            print(mname)
            print(v)
            v["model"] = load_model(v["model_name"])


        print("------ NLI models")
        for mname, v in cfg["nli_models"].items():
            print(mname)
            print(v)
            type_ = v["type"]
            assert type_ in set(["crossencoder", "default"])
            order = v.get("order")
            claim_first = v.get("claim_first", False)
            v["model"] = SupportRefuteNEIModel(v["model_name"], order=order, type_=type_, claim_first=claim_first)

        print("------ Evidence Retrieval models")
        for mname, v in cfg["retrieval_models"].items():
            print(mname)
            print(v)
            if v["type"] == "anserini":
                anserini = Anserini(model_dir=v["index"])
                v["model"] = FilteringRetriever(cfg["db"], anserini, maxk=1000, max_repeats=3, initial_scale=5)
            elif v["type"] == "rest-api":
                rest_retrieval = RESTRetrieval(v["name"], v["url"])
                v["model"] = FilteringRetriever(cfg["db"], rest_retrieval, maxk=1000, max_repeats=3, initial_scale=50)
            elif v["type"] == "colbertv1":
                colbertArgs = ColBERTArgs(
                            dim=v["dim"],
                            index_root=v["index_root"],
                            index_name=v["index_name"],
                            checkpoint=v["checkpoint"],
                            idConvPath=v["idConvPath"],
                )
                colbert = ColBERT(colbertArgs)
                v["model"] = FilteringRetriever(cfg["db"], colbert, maxk=1000, max_repeats=3, initial_scale=50)
            elif v["type"] == "meta":
                kw_model = cfg["retrieval_models"][v["keyword"]]["model"]
                semantic_model = cfg["retrieval_models"][v["semantic"]]["model"]
                scoring_model = cfg["scoring_models"][v["scoring"]]["model"]
                nli_model = cfg["nli_models"][v["nli"]]["model"]
                sort_nli = v.get("sort_nli", True)

                meta = MetaRetriever(db=cfg["db"],
                                     semmodel=semantic_model,
                                     kwmodel=kw_model, 
                                     nlimodels=[nli_model], 
                                     scoring_model=scoring_model, 
                                     max_titles=2,
                                     sort_nli=sort_nli)
                v["model"] = meta
            elif v["type"] == "directed":
                retrieval_models = [cfg["retrieval_models"][rm]["model"] for rm in v["retrieval_models"]]

                direct = DirectedRetriever(db=cfg["db"],
                                     lang=cfg["lang"],
                                     similarity=v["similarity"],
                                     retrieval_models=retrieval_models,
                                     stopword_list=v["stopword_list"],
                                     ner_model_name=v.get("ner_model_name", None),
                                     fast_text_model=v.get("fast_text_model", None),
                                     similarity_threshold=v["similarity_threshold"],
                                     similarity_min_chars=v["similarity_min_chars"],
                                     ner_weight=v["ner_weight"],
                                     )
                v["model"] = direct
            else:
                raise NotImplemented()
            v["scoref"] = dummy_nan_score

        print("------ Emphasize")
        assert "emphasize" in cfg, "emphasize key missing!"
        v = cfg["emphasize"]
        assert v["similarity"] in ["jaro", "fasttext"], f"Unknown emphasize similarity: {v['similarity']}"

        if v["similarity"] == "jaro":
            v["model"] = EmphasizeClaimWords(similarity=jaro_winkler_metric,
                                             min_chars = v["similarity_min_chars"],
                                             threshold=v["similarity_threshold"],
                                             stopword_list=v["stopword_list"])
        else:
            similarity = FastTextSimilarity(model_path=v["model_path"])
            v["model"] = EmphasizeClaimWords(similarity=similarity.similarity,
                                             min_chars = v["similarity_min_chars"],
                                             threshold=v["similarity_threshold"],
                                             stopword_list=v["stopword_list"])
        return cfg
    

def main():
    parser = ArgumentParser()
    parser.add_argument('cfg_file', type=str)
    parser.add_argument('--port', type=int)

    args = parser.parse_args()

    cfg = populate_cfg(args.cfg_file)

    print(f'run: ssh -N -L {args.port}:{os.environ.get("HOSTNAME", "hostname")}:{args.port} rci3')
    assets_folder = os.getcwd() + "/data/assets"
    print(f"assets_folder={assets_folder}")

    pathname_params = dict()
    hosting_path = cfg.get("hosting_path")
    if hosting_path is not None:
        pathname_params["url_base_pathname"] = f"/{hosting_path}/"                                                                                                                                                                                                                              

    app = Dash(__name__, external_stylesheets=[dbc.themes.SIMPLEX], assets_folder=assets_folder, **pathname_params)
    dash_build_app(app, cfg)
    app.run_server(host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()