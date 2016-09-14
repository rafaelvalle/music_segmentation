#!/usr/bin/env python
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.use('Agg')
font = {'size': 7}
mpl.rc('font', **font)
import os
import argparse
import functools
import glob2 as glob
import matplotlib.pylab as plt
# import seaborn

import numpy as np
from sklearn.cluster import DBSCAN

from feature_extraction import extractFeature
from statistics import reduceDimensionality, lpf


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


def plotDimReduced(fullpath, feature='chroma', order=1, sr=4, cutoff=.1,
                   dim_red='SVD', n_singv=3, round_to=0, normalize=1,
                   scale=1, length=4):
    feat = {}
    print ('Analyzing {} with feature {},  order {}, sr {}, cutoff {}, '
           'n_singv {}, scale {} normalize {}, round_to {}'.format(
               fullpath, feature, order, sr, cutoff, n_singv, scale, normalize,
               round_to))
    # extract filename, filepath and beat aligned feature
    filename, file_ext = os.path.splitext(fullpath)

    # extract filter and apply pre-processing
    feat[feature], beat_times = extractFeature(
        filename, file_ext, feature, scale, round_to, normalize)

    # set dimensionality reduction technique
    feat[dim_red] = reduceDimensionality(
        dim_red, feat[feature], n_singv)
    feat['{}(LPF)'.format(dim_red)] = reduceDimensionality(
        dim_red, lpf(feat[feature], cutoff, sr, order), n_singv)
    feat['LPF({})'.format(dim_red)] = lpf(
        feat[dim_red], cutoff, sr, order)

    # perform DB clustering
    clustering = {}
    for k, v in feat.items():
        if 'LPF' in k:
            eps = 0.05
            min_samples = 8
        else:
            eps = 0.05
            min_samples = 8
        clustering[k] = DBSCAN(eps=eps, min_samples=min_samples).fit(v)

    # create vars for plotting
    fig = plt.figure(figsize=(50, 20))
    fig.suptitle('feature {} order {}, cutoff {}, sr {}'.format(
        feature, order, cutoff, sr))

    gs = mpl.gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    i = 0
    print "\tPlot data and pre-processing"
    for name, data in feat.items():
        if name != feature:
            ax = fig.add_subplot(gs[0, i], projection='3d')
            ax.set_title(name)
            unique_labels = set(clustering[name].labels_)
            print name, unique_labels
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

            for lbl, color in zip(unique_labels, colors):
                if lbl == -1:
                    # Black used for noise.
                    color = 'k'

                mask = (clustering[name].labels_ == lbl)
                ax.scatter(data[mask][:, 0], data[mask][:, 1], data[mask][:, 2],
                           c=color)

            for n in xrange(len(data)):
                ax.text(data[n, 0], data[n, 1], data[n, 2], n, None)
            ax.grid(True)
            i += 1

    plt.tight_layout()
    plt.savefig("{}_scatter3d_{}_{}_r_{}_n_{}_s_{}_l_{}_{}.png".format(
        filename, feature, cutoff, round_to, normalize, scale, length, dim_red))
    plt.close(fig)


def plotData(glob_str, cutoff, order, sr, feature, dim_red, round_to, normalize,
             scale):
    tracks = [x for x in glob.glob(os.path.join(glob_str))]
    map(functools.partial(plotDimReduced, feature=feature, order=order, sr=sr,
                          cutoff=cutoff, dim_red=dim_red, round_to=round_to,
                          normalize=normalize, scale=scale), tracks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("glob_str", help="Glob string for input files",
                        type=str)
    parser.add_argument("feature", help="Feature (mfcc, chroma, cqt)",
                        type=str)
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
    parser.add_argument("-d", "--dim_red", help='SVD or NMF',
                        type=str, default='SVD', nargs='?')

    args = parser.parse_args()
    plotData(
        args.glob_str, args.cutoff, args.order, args.sr, args.feature,
        args.dim_red, args.round_to, args.normalize, args.scale)
    exit(0)
