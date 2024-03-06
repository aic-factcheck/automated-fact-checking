from collections import defaultdict, OrderedDict, Counter

import numpy as np

from dash import html
import dash_bootstrap_components as dbc

from aic_nlp_utils.fever import fever_detokenize
from mark import mark_retrieval_importance, merge_spans

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def nli_info(cl_probs):
    # refutes, supports, nei - work for both 2-way and 3-way classifiers
    cl_id = np.argmax(cl_probs)
    cl_name = ["Refutes", "Supports", "NEI"][cl_id]
    cl_col = ["primary", "success", "danger"][cl_id] # for SIMPLEX theme
    cl_prob = 100 * cl_probs[cl_id]
    return cl_name, cl_col, cl_prob


def apply_marks(txt, spans):
    # logger.error(f"apply_marks: {txt}\n")
    if len(spans) == 0:
        return [txt]

    type2color = {"claim_words": "#FADBD8", "importance_sentence": "#fcf8e3", "importance_word": "#fcf8e3"}
    markedtxt = []
    lold = 0
    for span in spans:
        f, l = span["span"].lower, span["span"].upper
        col = type2color[span["type"]]
        if len(txt[lold:f]) > 0:
            markedtxt.append(txt[lold:f])
        markedtxt.append(html.Mark(txt[f:l], style={"padding": "0 0.1em", "background-color": col}))
        lold = l
    if len(txt[lold:]) > 0:
        markedtxt.append(txt[lold:])
    
    return markedtxt

def prepare_texts(dids,
                  claim_txt,
                  db,
                  emphasize_model,
                  importance_model=None,
                  importance_type='sentence'):
    assert importance_type in ["sentence", "word", "none"]
    id2txt = OrderedDict()
    # collect all block texts
    for did in dids:
        id2txt.update(db.get_block_texts(did))
    txts = [fever_detokenize(txt) for txt in id2txt.values()]

    marks = []
    if importance_model is not None and importance_type != "none":
        select_top = 2 if importance_type == "sentence" else 4 
        importance_marks = [mark_retrieval_importance(importance_model, claim_txt, txt, select_top=select_top, importance_type=importance_type) for txt in txts]
        marks.append(importance_marks)

    claim_word_marks = [emphasize_model.emphasize_spans(claim_txt, txt) for txt in txts]
    marks.append(claim_word_marks)

    marks = [merge_spans(ms) for ms in zip(*marks)]

    markedtxts = [apply_marks(txt, ms) for txt, ms in zip(txts, marks)]

    # back to dict
    id2txt = OrderedDict([(id_, txt) for id_, txt in zip(
        list(id2txt.keys()), txts)])  # unmarked version
    markedid2txt = OrderedDict(
        [(id_, txt) for id_, txt in zip(list(id2txt.keys()), markedtxts)])
    return id2txt, markedid2txt


def split_marked_title(markedtxt):
    # search to first '\n'
    # logger.error(f"MT={markedtxt}")
    title = ""
    for i in range(len(markedtxt)):
        e = markedtxt[i]
        if isinstance(e, html.Mark):
            e = e.children
        if '\n' in e:
            idx = e.index('\n')
            pre, post = e[:idx], e[idx:]
            title += pre
            return title, [post] + markedtxt[i+1:]
        title += e
    return title, []


def extract_page_title(markedtxt):
    markedtxt = [m for m in markedtxt if isinstance(m, html.Mark) or len(m) > 0]
    title, rest = split_marked_title(markedtxt)
    return title, rest


def output_did(did,
               cfg,
               db,
               info,
               marked_id2txt,
               id2scores,
               id2nli,
               modelid,
               search_titles):
    ids = db.did2ids(did)
    # only for WIKI!
    page_title_txt, rest = extract_page_title(marked_id2txt[ids[0]])

    # logger.error(f"TITLE: {page_title_txt}")
    # logger.error(f"REST: {rest}")
    header = [dbc.Col(
        html.A(
                html.H4(page_title_txt),
                href=f"{cfg['original_site_prefix']}{did}", target="_blank", style={"color": "black"}
                )
            )]

    if len(ids) > 1:
        header +=  [dbc.Col(
            dbc.Button('Expand', 
                        id={"type": "claim-ctx", "did": f"{did}-{modelid}"}, 
                        color="link", size="sm", className=f"show-{did}-{modelid}", n_clicks=0))]

    res = [dbc.Row(header, justify="between")]
    for id_ in ids:
        # logger.error(f"ID: {id_}")
        markedtxt = marked_id2txt[id_]

        if id_ in id2nli:
            cl_name, cl_col, cl_prob  = nli_info(id2nli[id_])
            nlitxt = [
                dbc.Badge(f"{cl_name} {cl_prob:.2f}%", color=cl_col, id=f"nli_badge_{id_}"),
                dbc.Tooltip("Evidence classification according to the claim. Evidence is either: Supported, Refuted or NEI (Not Enough Info is available).", target=f"nli_badge_{id_}", placement="auto")
                ]
        else:
            nlitxt = []

        def method_txt():
            if id_ in info:
                if 'search' in info[id_]:
                    search = info[id_]['search']
                    return [
                        dbc.Badge(search.upper(), pill=True, color="warning", id=f"method_badge_{id_}"),
                        dbc.Tooltip("Method used to retrieve the evidence.", target=f"method_badge_{id_}", placement="auto")
]
            return []

        def score_txt():
            # this score is given by FAISS and can differ from permodel scores for the semantic models: see retrieval.retrieve
            if len(id2scores) > 0: # show nothing if per-model scores are computed
                return []
            if id_ in info:
                if db.istitle(id_) and not search_titles:
                    return []
                score = info[id_]["score"]["orig"]
                return [dbc.Badge(f"score: {score:.3f}", pill=True, color="primary")]
            else:
                return []

        def permodel_score_txts():
            # score for all blocks of all articles are evaluated separately see search.sem_distance and search.scores_tfidf
            if len(id2scores) == 0:
                return []
            scores = [id2scores[model_name].get(
                id_, np.nan) for model_name in cfg["retrieval_models"]]
            return [
                dbc.Badge(f"{model_name}: {score:.3f}", pill=True, color="info") for model_name, score in zip(cfg["retrieval_models"], scores) if not np.isnan(score)]


        infotxt = nlitxt + method_txt()
        if cfg["show_score"]:
            infotxt += score_txt() + permodel_score_txts()
        infotxt += [html.Br()]

        # do not use for other than Wikipedia with prepended titles
        def fix_new_lines_headings(src_lst, page_title_txt):
            dst_lst = []
            for t in src_lst:
                # logger.error(f"T: {t}")
                if isinstance(t, str):
                    parts = [p for p in t.split("\n")]

                elif isinstance(t, html.Mark):
                    parts = [html.Mark(p, style=t.style) if len(p) > 0 else "" for p in t.children.split("\n")]
                else:
                    logger.error(f"NO STRING OR MARKER! {t}")
                for p in parts[:-1]:
                    dst_lst.append(html.Br() if p == "" else p)
                if len(parts) > 0:
                    dst_lst.append(html.Br() if parts[-1] == "" else parts[-1])

            # remove titles
            prefix = ""
            pre_lst = []
            flt_lst = []
            for p in dst_lst:
                pre_lst.append(p)
                if isinstance(p, html.Br):
                    if prefix != page_title_txt:
                        flt_lst += pre_lst
                    # else:
                        # logger.error(f"skipping: {pre_lst}")
                    prefix = ""
                    pre_lst = []
                else:
                    prefix += p.children if isinstance(p, html.Mark) else p
                    # logger.error(f"prefix: {prefix}")
            flt_lst += pre_lst

            # HACK :( remove leading line breaks
            idx  = 0
            while isinstance(flt_lst[idx], html.Br):
                idx += 1

            return flt_lst[idx:]
        

        markedtxt = fix_new_lines_headings(markedtxt, page_title_txt)
        # for p in markedtxt:
        #     logger.error(f"AFT: {type(p)} '{p}'")

        def render_title():
            date_ = db.did2date(did).strftime("%d.%m. %Y, %H:%M:%S")
            if cfg["temporal"]:
                return infotxt + [html.H4(markedtxt), html.Small(date_)]
            else:
                return infotxt + [html.H4(markedtxt)]

        def render_selected_par():
            date_ = db.did2date(did).strftime("%d.%m. %Y, %H:%M:%S")
            if cfg.get("show_paragraph_date", False):
                return [html.P(html.B(infotxt + markedtxt), style={'display': 'block'}), html.Small(date_)]
            else:
                return [html.P(html.B(infotxt + markedtxt), style={'display': 'block'})]

        def render_other_par():
            return [html.P(infotxt + markedtxt,  className=f"did-{did}-{modelid}", style={'display': 'none'})]

        if db.istitle(id_):
            res += render_title()
        elif id_ in info:
            res += render_selected_par()
        else:
            res += render_other_par()
        
    # res += [
    #     dbc.Row([
    #         dbc.Button('Expand', id={"type": "claim-ctx", "did": f"{did}-{modelid}"},
    #                    color="link", size="sm", className=f"show-{did}-{modelid}", n_clicks=0),
    #         dbc.Button(f"Source: {did}", href=f"{cfg['original_site_prefix']}{did}",
    #                    target="_blank", color="link", size="sm"),
    #     ]),] 
    res += [
        html.Hr(),
    ]

    return res
