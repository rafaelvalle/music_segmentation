import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pyclust import KMedoids

from statistics import findElbow

def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


data = np.array([1,1,0,0,1,0,1,0,1,1,0,0,1,0,0,1,
                 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
feats = {}
lengths = (2, 4, 8)
max_l = max(lengths)
for l in lengths:
    feats[l] = np.array([data[i:i+l] for i in xrange(0, len(data), l)])

patterns = {}
K_MIN, K_MAX = 2, max_l/2

kmeans = {}
scores = {}
for l in lengths:
    clustering = [KMedoids(n_clusters=k).fit(feats[l])
                  for k in xrange(K_MIN, K_MAX+1)]
    score = [silhouette_score(feats[l], clustering[k].labels_)
             for k in xrange(len(clustering))]
    idx_best = findElbow(np.dstack((xrange(K_MIN, K_MAX+1), score))[0])
    scores[l] = score
    kmeans[l] = clustering[idx_best]

