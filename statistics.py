import numpy as np
import scipy as sp
from scipy.signal import butter, lfilter
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF


def movingAverage(a, n=3):
    ret = np.cumsum(a, axis=0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def butterLowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def lpf(data, cutoff, sr, order=5):
    b, a = butterLowpass(cutoff, sr, order=order)
    y = lfilter(b, a, data, axis=0)
    return y


##############################
#  DIMENSIONALITY REDUCTION  #
##############################
def reduceDimensionality(name, X, n_singv, inc_proj=False):
    if name == 'SVD':
        return svd(X, n_singv)
    elif name == 'NMF':
        return NMF(n_singv).fit_transform(X)
    else:
        raise Exception("Dimensionality Reduction {} not found".format(name))


def svd(data, n_singv=0, threshold=0.9, inc_proj=False):
    if n_singv > 0:
        lsv, sv, rsv = svds(data, n_singv, which='LM')
    else:
        lsv, sv, rsv = svds(data, data.shape[1] - 1, which='LM')
        # find number of singular values that explain 90% of variance
        n_singv = 1
        while np.sum(sv[-n_singv:]) / np.sum(sv) < threshold:
            n_singv += 1

    # compute reduced data and data projected onto principal components space
    data_redu = np.dot(data, rsv.T)
    if inc_proj:
        data_proj = np.dot(lsv[:, -n_singv:],
                           np.dot(np.diag(sv[-n_singv:]), rsv[-n_singv:, ]))
        return data_redu, data_proj
    return data_redu


#############
#  METRICS  #
#############
def findElbow(coordinates):
    """
    Returns
    -------
    idx : int
        Returns the idx of the coordinate farthest from a line drawn
        between first and last coordinates
    """
    line_vec = coordinates[-1] - coordinates[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    vec_from_first = coordinates - coordinates[0]
    scalar_product = np.sum(vec_from_first * np.tile(
        line_vec_norm, (len(coordinates), 1)), axis=1)
    vec_to_line = vec_from_first - np.outer(scalar_product, line_vec_norm)
    return np.argmax(np.sqrt(np.sum(vec_to_line ** 2, axis=1)))


def computeBic(clustering, X):
    """
    From https://github.com/bobhancock/goxmeans/blob/master/doc/BIC_notes.pdf
    BIC built around the 'identical spherical assumption', that is, data is
    modeled by n gaussians, each wtih identical variance and differing
    positions.
    """

    # assign centers and labels
    centers = [clustering.cluster_centers_]
    labels = clustering.labels_
    # number of clusters
    m = clustering.n_clusters
    # size of the clusters
    n = np.bincount(labels, minlength=m)
    # size of data set
    N, d = X.shape
    # compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * np.sum(
        [np.sum(sp.spatial.distance.cdist(
            X[np.where(labels == i)], [centers[0][i]], 'euclidean')**2)
            for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
                  n[i] * np.log(N) -
                  ((n[i] * d) / 2) * np.log(2 * np.pi) -
                  (n[i] * d / 2) * np.log(2 * np.pi * cl_var) -
                  ((n[i] - 1) * d / 2) for i in xrange(m)]) - const_term

    return BIC
