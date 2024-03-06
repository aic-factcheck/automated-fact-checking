from collections import Counter
import os
from os.path import join as pjoin
import argparse
import json
import re
import sys

import torch
import numpy as np
import scipy
# from transformers import XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer

import logging
logger = logging.getLogger(__name__)

def load_BERT(modeltype):
    modeltype = str(modeltype) # can be pathlib.Path
    word_embedding_model = Transformer(modeltype)
    pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

def load_model(modeltype):
    modeltype = str(modeltype) # can be pathlib.Path
    logger.info("reading transformer: {}".format(modeltype))
    if torch.cuda.is_available():
        logger.info("using GPU")
        device = torch.cuda.current_device()
    else:
        logger.info("using CPU")
        device = torch.device('cpu')
    #TODO improve this
    if modeltype in ["bert-base-multilingual-cased", "distilbert-base-cased"]:
        model = load_BERT(modeltype)
    else:
        model = SentenceTransformer(modeltype, device=device)
    logger.info("sentence transformer prepared")
    return model