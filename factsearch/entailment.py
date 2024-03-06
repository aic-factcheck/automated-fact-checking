from collections import OrderedDict
import numpy as np

import logging
logger = logging.getLogger(__name__)

def evaluate_claim_entailment(nli_model, claim, context):
    X = [(context_, claim) for context_ in context.values()]
    if len(X) == 1: # Crossencoder model fails otherwise!
        print(claim)
        print(context)
        return OrderedDict()
    probs = OrderedDict([(k, v) for k, v in zip(context.keys(), nli_model.predict(X, apply_softmax=True))])
    return probs
