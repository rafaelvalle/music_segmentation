#!/usr/bin/python
import matplotlib as mpl
mpl.use('Agg')
font = {'size': 7}
mpl.rc('font', **font)
import os
import argparse
import functools
import glob2 as glob
import numpy as np
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.mixture import DPGMM
from sklearn.metrics import silhouette_score
import seaborn
from statistics import svd, computeBic, findElbow, lpf
from feature_extraction import extractFeature
import matplotlib.pylab as plt


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


def dim_red_fn(name, X, n_singv, inc_proj=False):
    if name == 'SVD':
        return svd(X, n_singv)
    elif name == 'NMF':
        return NMF(n_singv).fit_transform(X)
    else:
        raise Exception("Dimensionality Reduction {} not found".format(name))


def plotClustering(fullpath, order=1, sr=4, cutoff=.1, n_singv=3,
                   feature='chroma', dim_red='SVD', round_to=0, normalize=1,
                   scale=1, length=4, clustering='KMEANS'):
    feat = {}
    print ('Analyzing {} with feature {},  order {}, sr {}, cutoff {}, '
           'n_singv {}, scale {} normalize {}, round_to {}'.format(
               fullpath, feature, order, sr, cutoff, n_singv, scale, normalize,
               round_to))
    # extract filename, filepath and beat aligned feature
    filename, file_ext = os.path.splitext(fullpath)

    # extract filter and apply pre-processing
    feat[feature], beat_times = extractFeature(
        filename, file_ext, feature, scale, round_to, normalize, save=True)

    # set dimensionality reduction technique
    feat['LPF'] = lpf(feat[feature], cutoff, sr, order)
    feat[dim_red] = dim_red_fn(dim_red, feat[feature], n_singv)
    feat['{}(LPF)'.format(dim_red)] = dim_red_fn(
        dim_red, feat['LPF'], n_singv)
    feat['LPF({})'.format(dim_red)] = lpf(feat[dim_red], cutoff, sr, order)
    feat['{}-LPF'.format(feature)] = feat[feature] - feat['LPF']
    feat['{}({}-LPF)'.format(dim_red, feature)] = dim_red_fn(
        dim_red, feat['{}-LPF'.format(feature)], n_singv)

    # create vars for plotting
    ts = np.arange(0, len(feat[feature]))
    step_size = max(1, int(len(ts) * .01))
    fig = plt.figure(figsize=(98, 64))
    fig.suptitle('feature {} order {}, cutoff {}, sr {}'.format(
        feature, order, cutoff, sr))

    gs = mpl.gridspec.GridSpec(14, 4, width_ratios=[1, 1, 1, 1])
    i = 0
    print "\tPlot data and pre-processing"
    for name in (feature, 'LPF', '{}-LPF'.format(feature), dim_red,
                 '{}(LPF)'.format(dim_red), 'LPF({})'.format(dim_red),
                 '{}({}-LPF)'.format(dim_red, feature)):
        data = feat[name]

        data_wide = np.array([feat[name][m:m+length, :]
                              for m in xrange(len(feat[name])-length)])
        data_wide = data_wide.reshape(
            data_wide.shape[0], data_wide.shape[1]*data_wide.shape[2])

        # build codebook using kmeans or DP-GMM
        if clustering == 'KMEANS':
            K_MIN, K_MAX = 2, 16
            KM = [KMeans(n_clusters=l, init='k-means++').fit(data_wide)
                  for l in xrange(K_MIN, K_MAX+1)]

            # compute scores to assess fit
            scores_bic = [computeBic(KM[x], data_wide) for x in xrange(len(KM))]
            scores_inertia = [KM[x].inertia_ for x in xrange(len(KM))]
            scores_silhouette = [silhouette_score(data_wide, KM[x].labels_,
                                                  metric='euclidean')
                                 for x in xrange(len(KM))]

            # get best clusters
            idx_best_bic = findElbow(np.dstack(
                (xrange(K_MIN, K_MAX+1), scores_bic))[0])
            idx_best_inertia = findElbow(np.dstack(
                (xrange(K_MIN, K_MAX+1), scores_inertia))[0])
            idx_best_silhouette = findElbow(np.dstack(
                (xrange(K_MIN, K_MAX+1), scores_silhouette))[0])
            idx_best = int(np.median(
                (idx_best_bic, idx_best_inertia, idx_best_silhouette))) + 1

            # get clusters and cluster allocations given best K
            k_best = idx_best + K_MIN
            centroids = KM[idx_best].cluster_centers_
            centroid_idx = KM[idx_best].labels_
        elif clustering == 'DPGMM':
            n_components = 12
            dpgmm = DPGMM(
                n_components=n_components, tol=1e-3, n_iter=32, alpha=1000,
                covariance_type='diag', verbose=True)
            dpgmm.fit(data_wide)

            # compute scores to assess fit
            scores_bic = dpgmm.bic(data_wide)
            scores_silhouette = [silhouette_score(data_wide, centroids,
                                 metric='euclidean')]
            scores_silhouette = [0.0]

            # get clusters and cluster allocations given best K
            k_best = dpgmm.means_.shape[0]
            centroids = dpgmm.means_
            centroid_idx = np.argmax(dpgmm.predict_proba(data_wide), axis=1)

        # plot data
        if data.shape[1] == 3:
            data = data.reshape(1, data.shape[0], data.shape[1])
        else:
            data = data.T

        ax = fig.add_subplot(gs[i, :])
        ax.set_title(name)
        ax.imshow(data,
                  interpolation='nearest',
                  origin='low',
                  aspect='auto',
                  cmap=plt.cm.Oranges)
        ax.set_xticks(ts[::step_size])
        ax.set_xticklabels(beat_times[::step_size], rotation=60)
        ax.grid(False)

        # plot clustering on raw feature
        changes = np.hstack(([True], centroid_idx[:-1] != centroid_idx[1:]))
        ax_twin = ax.twiny()
        ax_twin.set_xlim(ax.get_xlim())
        ax_twin.set_xticks(np.argwhere(changes)[:, 0])
        ax_twin.set_xticklabels(centroid_idx[changes])
        ax_twin.grid(False)

        # plot codebook (centroids)
        ax = fig.add_subplot(gs[i+1, 0])
        ax.set_title(name)

        if centroids.shape[1] == 3:
            centroids = centroids.reshape(
                1, centroids.shape[0], centroids.shape[1])
        elif centroids.shape[1] == n_singv * length:
            centroids = centroids.reshape(
                1, centroids.shape[0]*length, centroids.shape[1]/length)
        else:
            centroids = centroids.reshape(
                centroids.shape[0] * length,
                centroids.shape[1] / length).T
        ax.imshow(centroids,
                  interpolation='nearest',
                  origin='low',
                  aspect='auto',
                  cmap=plt.cm.Oranges)
        ax.set_xticks(xrange(0, centroids.shape[1], 4))
        ax.set_xticklabels(xrange(k_best))
        ax.grid(False)

        # plot elbow curve
        c = 1
        for k, v, idx in (('BIC', scores_bic, idx_best_bic),
                          ('INERTIA', scores_inertia, idx_best_inertia),
                          ('SILHOUETTE', scores_silhouette, idx_best_silhouette)
                          ):
            ax = fig.add_subplot(gs[i+1, c])
            ax.set_title('{}, {} best K {}'.format(name, k, idx+K_MIN))
            ax.plot(xrange(K_MIN, K_MAX+1), v,  'b*-')
            ax.set_xlim((K_MIN, K_MAX+1))
            ax.set_xlabel('Number of clusters')
            ax.set_ylabel('Score')
            ax.grid(True)
            ax.axvline(idx+K_MIN, color='r')
            c += 1
        i += 2

        """
        if 'SVD' in name:
            # scikit-image clustering
            segments_slic = slic(
                data, n_segments=10, compactness=10, sigma=1)
            segments_quickshift = quickshift(
                data, kernel_size=3, max_dist=6, ratio=0.5)

            ax = fig.add_subplot(gs[k, 0])
            ax.set_title('{} with quickshift'.format(name))
            ax.imshow(mark_boundaries(data, segments_quickshift, mode='outer'),
                      interpolation='nearest',
                      origin='low',
                      aspect='auto',
                      cmap=plt.cm.Oranges)
            ax.set_xticks(ts[::step_size])
            ax.set_xticklabels(beat_times[::step_size], rotation=60)
            ax.grid(False)

            ax = fig.add_subplot(gs[k, 1])

            ax.set_title('{} with slic'.format(name))
            ax.imshow(mark_boundaries(data, segments_slic, mode='outer'),
                      interpolation='nearest',
                      origin='low',
                      aspect='auto',
                      cmap=plt.cm.Oranges)
            ax.set_xticks(ts[::step_size])
            ax.set_xticklabels(beat_times[::step_size], rotation=60)
            ax.grid(False)
            k += 1
        """

    plt.tight_layout()
    plt.savefig("{}_clustering_{}_{}_r_{}_n_{}_s_{}_l_{}_{}.png".format(
        filename, feature, cutoff, round_to, normalize, scale, length, dim_red))

    # save with large size
    plt.savefig("{}_clustering_{}_{}_r_{}_n_{}_s_{}_l_{}_{}.png".format(
        filename, feature, cutoff, round_to, normalize, scale, length, dim_red))
    # save with smaller size
    fig.set_figwidth(36)
    fig.set_figheight(18)
    plt.tight_layout()
    plt.savefig("{}_clustering_{}_{}_r_{}_n_{}_s_{}_l_{}_{}_small.png".format(
        filename, feature, cutoff, round_to, normalize, scale, length, dim_red))

    plt.close(fig)


def plotData(glob_str, feature, dim_red, cutoff, order, sr, round_to, normalize,
             scale, length, clustering):
    print glob_str
    tracks = [x for x in glob.glob(os.path.join(glob_str))]
    map(functools.partial(plotClustering, cutoff=cutoff, order=order, sr=sr,
        feature=feature, dim_red=dim_red, round_to=round_to,
        normalize=normalize, scale=scale, length=length, clustering=clustering),
        tracks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("glob_str", help="Glob string for input files",
                        type=str)
    parser.add_argument("feature", help="Feature (mfcc, chroma, cqt)",
                        type=str)
    parser.add_argument("-d", "--dim_red", help='SVD or NMF',
                        type=str, default='SVD', nargs='?')
    parser.add_argument("-c", "--cutoff", help='Low-Pass Filter Cuttoff',
                        type=float, default=0.1, nargs='?')
    parser.add_argument("-o", "--order", help='Low-Pass Filter Order',
                        type=int, default=1, nargs='?')
    parser.add_argument("-sr", "--sr", help='Low-Pass Filter Sampling Rate',
                        type=int, default=4, nargs='?')
    parser.add_argument("-r", "--round_to", help='Round to decimal',
                        type=float, default=0.25, nargs='?')
    parser.add_argument("-n", "--normalize", help='Normalize data?',
                        type=int, default=1, nargs='?')
    parser.add_argument("-s", "--scale", help='Scale data?',
                        type=int, default=1, nargs='?')
    parser.add_argument("-l", "--length", help='Length of code for clustering',
                        type=int, default=4, nargs='?')
    parser.add_argument("-k", "--clustering", help='KMEANS or DPGMM',
                        type=str, default='KMEANS', nargs='?')

    args = parser.parse_args()
    plotData(
        args.glob_str, args.feature, args.dim_red, args.cutoff,
        args.order, args.sr, args.round_to, args.normalize,
        args.scale, args.length, args.clustering)
    exit(0)
