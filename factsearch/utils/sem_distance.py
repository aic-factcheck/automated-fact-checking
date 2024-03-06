import numpy as np
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.preprocessing import normalize
import unicodedata as ud

def sem_distance(model, claim, docs, preprocess=lambda t: t, norm='NFC'):
    claim = preprocess(ud.normalize(norm, claim))
    docs = [preprocess(ud.normalize(norm, doc)) for doc in docs]
    y = model.encode([claim] + docs, show_progress_bar=False, convert_to_numpy=True)
    y = normalize(y)
    
    y_claim = np.tile(y[0:1], (y.shape[0]-1, 1))
    y_pages = y[1:]
    
    dists = 1 - paired_cosine_distances(y_claim, y_pages)
    return dists