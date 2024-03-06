import argparse
from collections import defaultdict, OrderedDict, Counter
import numpy as np
import json
import datetime as dt
from itertools import chain
import os
from os.path import join as pjoin
import pandas as pd
from pprint import pprint

import string
from time import time
import torch

import humanize

from flask import request   

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State, ALL, MATCH
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from entailment import evaluate_claim_entailment
from mark import mark_retrieval_importance, merge_spans
from search import fact_search, scores_tfidf, filter_results, score_blocks, extract_id_info, extract_doc_dates
from vis import fig_hist, fig_line

from view_search_row import define_search_row
from view_result_options_row import define_result_options_row
from view_slot_options import define_slot_options
from view_texts import prepare_texts, output_did

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def dash_build_app(app, cfg):
    db, models = cfg["db"], cfg["retrieval_models"]
    temporal = cfg.get("temporal", True)
    scoring_models = cfg["scoring_models"]
    nli_models = cfg["nli_models"]
    slotcfgs = cfg["slots"]
    nslots = cfg["initslots"]

    searching = False

    header_row = dbc.Row([
            dbc.Col([html.Table([
                html.Tr([
                    html.Td([html.H1("Fact Search")]), 
                    html.Td([cfg["corpus_name"]], style={"padding-left": "2em", "font-size": "24px"}),
                    html.Td([cfg["corpus_version"]], style={"padding-left": "2em", "font-size": "11px"})
                ]),
                ])], width=6),
            dbc.Col([html.Img(src=app.get_asset_url('Logo_AIC_FEECTU_Holo.png'), style={'height':'100%', 'width':'100%'})], width=2)
        ], justify="between")

    search_row = define_search_row(cfg)
    result_options = define_result_options_row(db, nslots, len(slotcfgs), temporal)
    # search_row = [dbc.Col([define_search_row(cfg),
    #                    define_result_options_row(db, nslots, len(slotcfgs), temporal)]),
    # ]

    # slot_options_row = dbc.Container(define_slot_options(cfg, slotcfgs, nslots), id="slot-options")
    slot_options_row = dbc.Col(dbc.Row(define_slot_options(cfg, slotcfgs, nslots), id="slot-options"))

    sections = []
    sections.append(html.Div(id="blank-output"))
    sections.append(header_row)
    if cfg.get("merge_search_and_slot_options", False):
        sections.append(dbc.Form(dbc.Row(search_row + [slot_options_row] + result_options, align="end")))
    else:
        sections.append(dbc.Form(dbc.Row(search_row + result_options)))
        sections.append(dbc.Row(slot_options_row, id="slot-options"))
    sections.append(html.Br())
    sections.append(dbc.Row([dbc.Spinner(html.Div(id="loading-results"))]))
    sections.append(dbc.Row([], id="slot-results"))

    app.layout = dbc.Container(sections)

    # still fails for new Claim
    app.clientside_callback(
        """
        function(n_clicks) {
            if(Math.max(...n_clicks) <= 0) return;
            const trig = dash_clientside.callback_context.triggered[0];
            const val = trig.value;
            const did = JSON.parse(trig.prop_id.split(".")[0])["did"]; // terrible!
            const button = document.querySelector("button.show-" + did);
            console.log(val);
            console.log(dash_clientside.callback_context);
            for (const pctx of document.querySelectorAll("p.did-" + did).values()) {
                if(val % 2 == 1) {
                    pctx.style['display'] = 'block';
                    button.textContent = "Collapse";
                } else {
                    pctx.style['display'] = 'none';
                    button.textContent = "Expand";
                }
            }
            return;
        }
        """,
        Output("blank-output", "children"),
        Input({"type": "claim-ctx", "did": ALL}, "n_clicks"),
    )

    @app.callback(
        Output({"type": "collapse", "id": ALL}, "is_open"),
        Input({"type": "collapse-button", "id": ALL}, "n_clicks"),
        State({"type": "collapse", "id": ALL}, "is_open"),
    )
    def toggle_collapse(n, is_open):
        ctx = dash.callback_context
    #     if not ctx.triggered:
        if ctx.triggered[0]['value'] is None:
            return [False] * len(n)
        else:
            button_id = json.loads(
                ctx.triggered[0]["prop_id"].split(".")[0])["id"]
        ret = [not state["value"] if state["id"]["id"] ==
               button_id else state["value"] for state in ctx.states_list[0]]
        return ret

    @app.callback(
        # Output({"type": "results", "id": ALL}, 'children'),
        Output('slot-options', 'children'),
        Output('slot-results', 'children'),
        Output('loading-results', 'children'),
        Input('claim-search', 'n_clicks'),
        Input('nslots', 'value'),
        State('claim-txt', 'value'),
        State('search-switches', 'value'),
        State({"type": "model", "id": ALL}, 'value'),
        State({"type": "model-k", "id": ALL}, 'value'),
        State({"type": "importance-model", "id": ALL}, 'value'),
        State({"type": "importance-type", "id": ALL}, 'value'),
        State({"type": "nli-model", "id": ALL}, 'value'),
        State('date-range', 'start_date'),
        State('date-range', 'end_date'),
        State('order-results', 'value'),
        State('slot-options', 'children'),  # if no change
        State('slot-results', 'children'),
    )
    def update_output_div(n_clicks,
                          nslots,
                          claim_txt,
                          search_switches,
                          model_names,
                          ks,
                          importance_model_names,
                          importance_types,
                          nli_model_names,
                          start_date, end_date,
                          order_results,
                          current_options,
                          current_results):

        nonlocal searching
        print(f"{request.remote_addr} running search: {searching}")
        if searching: # TODO: should we care for the thread safety?
            raise PreventUpdate

        print(f"{request.remote_addr} update_output_div")

        # update slot configuration from UI
        for i, (model_name, model_k, importance_model_name, importance_type, nli_model_name) in enumerate(zip(model_names, ks, importance_model_names, importance_types, nli_model_names)):
            slotcfgs[i] = {"retrieval_model_name": model_name, "k": model_k,
                           "importance": importance_model_name,
                           "importance_type": importance_type,
                           "nli_model_name": nli_model_name}
            
        # print(f"slotcfgs: {slotcfgs}")

        # if changing the number of slots
        trigger = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
        if trigger == "nslots":
            nslots_current = len(current_options)
            # print(f"current: {nslots_current}, new: {nslots}")
            txts = current_results[0:nslots] if nslots <= nslots_current else current_results + [
                dbc.Col([""])] * (nslots - nslots_current)
            return define_slot_options(cfg, slotcfgs, nslots), txts, ""

        # do not change anything for no claim text filled
        if len(claim_txt) == 0:
            return current_options, current_results, ""

        search_titles = "search_titles" in search_switches
        detailed_score = "detailed_score" in search_switches


        def prepare_column(pre, post, modelid, importance_model_name, importance_type, nli_model_name, timeinfo):
            # pre/post filtraton which is now part of search
            importance_model = None if importance_model_name is None else scoring_models[
                importance_model_name]["model"]
            hist_date = fig_hist([pre, post], lambda d: extract_doc_dates(
                db, d), ["pre", "post"], "Date", "fig-hist-date")
            # line_score = fig_line([pre, post], extract_id_info, [
            #                       "pre", "post"], "Score", "fig-list-score", g=lambda y: sorted(y)[: : -1])

            stats = html.Div([
                dbc.Button("Statistics", id={
                           "type": "collapse-button", "id": modelid}, color="secondary"),
                # dbc.Collapse([hist_date, line_score], id={
                            #  "type": "collapse", "id": modelid})
            ])

            st = time()
            id2txt_all, marked_id2txt = prepare_texts(post.keys(), 
                                                      claim_txt,
                                                      db,
                                                      emphasize_model=cfg["emphasize"]["model"],
                                                      importance_model=importance_model,
                                                      importance_type=importance_type)
            mtime = time()-st

            # id2txt_all holds all blocks of articles for which at least one block was retrieved
            # id2txt_retrieved keeps only the retrieved blocks
            retrieved_ids = set() # collect just the retrieved ids
            for block_records in post.values():
                retrieved_ids.update(block_records)    
            id2txt_retrieved = OrderedDict([(k, v) for k, v in id2txt_all.items() if k in retrieved_ids])

            # TODO add switch between `id2txt_all` and `id2txt_retrieved` 
            id2txt = id2txt_retrieved

            st = time()
            id2scores = {model_name: score_blocks(
                models[model_name], claim_txt, id2txt) for model_name in models} if detailed_score else {}
            scoretime = time()-st

            st = time()
            id2nli = evaluate_claim_entailment(
                nli_models[nli_model_name]["model"], claim_txt, id2txt)
            nlitime = time()-st

            if order_results == "date_asc":
                dids = sorted(post.keys(), key = lambda did: db.did2date(did))
            elif order_results == "date_desc":
                dids = sorted(
                    post.keys(), key = lambda did: db.did2date(did))[::-1]
            elif order_results == "score":
                dids = post.keys()

            articles = []
            for did in dids:
                info = post[did]
                articles += output_did(did,
                                       cfg,
                                       db,
                                       info,
                                       marked_id2txt,
                                       id2scores,
                                       id2nli,
                                       modelid=modelid,
                                       search_titles=search_titles)

            mtime=humanize.naturaldelta(dt.timedelta(
                seconds=mtime), minimum_unit = "milliseconds")
            scoretime=humanize.naturaldelta(dt.timedelta(
                seconds=scoretime), minimum_unit = "milliseconds")
            stime=humanize.naturaldelta(dt.timedelta(
                seconds=timeinfo['fact_search_time'] + timeinfo['filtering_time']), minimum_unit = "milliseconds")
            
            status_line = html.Div(f"{len(post)} found, search: {stime}, importances: {mtime}, score: {scoretime}")

            sections = [status_line, html.Br()]
            if cfg["show_stats"]:
                sections.append(stats)
            sections.append(html.Hr())
            sections += articles
            txt = html.Div(sections)
            return txt

        txts = []
        searching = True # TODO concurrency?
        print(f"{request.remote_addr} running search now")
        for modelid, slotcfg in enumerate(slotcfgs[0:nslots]):
            logger.error(f'slotcfg = {slotcfg}')
            model = models[slotcfg["retrieval_model_name"]]["model"]
            k = slotcfg["k"]
            importance_model = slotcfg["importance"]
            importance_type = slotcfg["importance_type"]
            nli_model = slotcfg["nli_model_name"]
            max_titles = 2 if search_titles else 0
            datemin = dt.date.fromisoformat(start_date)
            datemax = dt.date.fromisoformat(end_date)
            res, timeinfo = fact_search(db, model, claim_txt, k, max_titles, datemin, datemax)
            logger.error(f"SEARCH ({modelid}, k={k}, [{datemin}, {datemax}): '{claim_txt}'")
            # pprint(res)

            # post, posttime = filter_results(pre, db,
            #                                 k, search_titles,
            #                                 dt.date.fromisoformat(start_date),
            #                                 dt.date.fromisoformat(end_date))
            # timeinfo = {**pretime, **posttime}
            timeinfo["filtering_time"] = 0.0
            txt = prepare_column(
                res, res, modelid, importance_model, importance_type, nli_model, timeinfo=timeinfo)
            # print("------TEXT-------")
            # print(txt)
            txts.append(dbc.Col([txt]))
        searching = False

        return define_slot_options(cfg, slotcfgs, nslots), txts, ""

# parser = argparse.ArgumentParser()
# parser.add_argument('--ner_model', required=True, type=str, help='location of NameTag2 NER model')
# parser.add_argument('--db_name', required=True, type=str, help='SQLite page database /path/to/fever.db')
# # parser.add_argument('--kw_model', required=True, type=str, help='keyword (NER) model location, e.g., DRQA index file')
# parser.add_argument('--sem_model', required=True, type=str, help='semantic model type, e.g., "bert-base-multilingual-cased" or model dir')
# parser.add_argument('--sem_embeddings', required=True, type=str, help='PyTorch tensor embedding file/dirctory, e.g., /path/to/embedded_pages.pt, if not given')
# parser.add_argument('--sem_faiss_index', required=True, type=str, help='FAISS index specification for the semantic model')
# parser.add_argument('--excludekw', required=False, type=str, default="", help='keywords to exclude separated by semicolon "sport;burza", case insensitive')

# try:
#     args = parser.parse_args()
# except SystemExit as e:
#     logger.error(e)
#     os._exit(e.code)

# main()
