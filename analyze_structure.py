#!/usr/bin/env python
import matplotlib as mpl
mpl.use('Agg')
font = {'size': 7}
mpl.rc('font', **font)
import os
import argparse
import functools
import glob2 as glob
import numpy as np
from numpy import matlib
import scipy as sp
# from scipy.fftpack import fft
from scipy.signal import butter, lfilter
from scipy.signal import medfilt
# from scipy.cluster.vq import kmeans
# from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# from skimage.segmentation import slic, quickshift
# from skimage.segmentation import mark_boundaries
import seaborn
import pretty_midi
import librosa
from librosa.core import stft
from helpers import cropEdges
from statistics import svd
import matplotlib.pylab as plt

SR = 8192
N_FFT = 4096
HOP_LENGTH = 512
BINS_PER_OCT = 12
OCTAVES = 7


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


def find_elbow(coordinates):
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
    scalar_product = np.sum(vec_from_first * np.matlib.repmat(
        line_vec_norm, len(coordinates), 1), axis=1)
    vec_to_line = vec_from_first - np.outer(scalar_product, line_vec_norm)
    return np.argmax(np.sqrt(np.sum(vec_to_line ** 2, axis=1)))


def compute_bic(clustering, X):
    """
    From https://github.com/bobhancock/goxmeans/blob/master/doc/BIC_notes.pdf
    BIC built around the 'identical spherical assumption', that is, data is
    modeled by n gaussians, each wtih identical variance and differing positions.
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


def moving_average(a, n=3):
    ret = np.cumsum(a, axis=0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def lpf(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data, axis=0)
    return y


def completeBeatTimes(beat_times):
    diff = beat_times[1] - beat_times[0]
    extra_beats = [0]

    while (extra_beats[-1] + diff) < beat_times[0]:
        extra_beats.append(extra_beats[-1] + diff)
    beat_times = np.hstack((extra_beats, beat_times))
    return beat_times


def extractChroma(filename, file_ext, beat_sync=True):
    if file_ext == ".mid":
        # load file and extract chromagram as bool and transpose to C
        try:
            data = pretty_midi.PrettyMIDI(filename+file_ext)
        except:
            return
        beat_times = data.get_beats()
        chroma = data.get_chroma(beat_times).astype(bool).T
        # remove silence from beginning and end
        chroma = cropEdges(chroma)
    elif file_ext in ('.wav', '.mp3'):
        # Load the example clip
        y, _ = librosa.load(filename+file_ext, sr=SR, mono=True)
        # Separate harmonics and percussives into two waveforms
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        # Compute chroma features from the harmonic signal
        chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=SR,
                                             n_chroma=12, n_fft=N_FFT,
                                             hop_length=HOP_LENGTH)
        # Beat track on the percussive signal
        tempo, beat_times = librosa.beat.beat_track(y=y_percussive,
                                                    sr=SR,
                                                    hop_length=HOP_LENGTH)
        # apply heuristics to have all beats
        beat_times = completeBeatTimes(beat_times)

        # Aggregate chroma features between beat events
        # We'll use the median value of each feature between beat frames
        chroma = librosa.feature.sync(chroma,
                                      beat_times,
                                      aggregate=np.median)

        # convert beat frame to seconds
        beat_times = librosa.frames_to_time(beat_times, sr=SR)

        # transpose such that rows are notes
        chroma = chroma.T

    else:
        raise Exception('{} is not a supported filetype'.format(file_ext))
    return chroma, beat_times


def extractMFCC(filename, file_ext):
    if file_ext in ('.wav', '.mp3'):
        y, _ = librosa.load(filename+file_ext, sr=SR, mono=True)
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        tempo, beat_times = librosa.beat.beat_track(y=y_percussive, sr=SR)
        beat_times = completeBeatTimes(beat_times)
        mfcc = librosa.feature.mfcc(
            y=y, sr=SR, n_mfcc=15, hop_length=HOP_LENGTH, fmin=27.5)
        mfcc = librosa.feature.sync(mfcc,
                                    beat_times,
                                    aggregate=np.median).T
        # remove amplitude column
        mfcc = mfcc[:, 1:]
        beat_times = librosa.frames_to_time(beat_times, sr=SR)
    else:
        raise Exception('{} is not a supported filetype'.format(file_ext))

    return mfcc, beat_times


def extractCQT(filename, file_ext):
    if file_ext in ('.wav', '.mp3'):
        # Load the example clip
        y, _ = librosa.load(filename+file_ext, sr=SR, mono=True)
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        tempo, beat_times = librosa.beat.beat_track(y=y_percussive, sr=SR)
        beat_times = completeBeatTimes(beat_times)
        cqt = np.abs(librosa.core.cqt(
            y=y, sr=SR, hop_length=HOP_LENGTH, resolution=1.0, fmin=27.5,
            n_bins=BINS_PER_OCT*OCTAVES,
            bins_per_octave=BINS_PER_OCT, real=False))
        cqt = librosa.feature.sync(cqt, beat_times, aggregate=np.median).T
        beat_times = librosa.frames_to_time(beat_times, sr=SR)
    else:
        raise Exception('{} is not a supported filetype'.format(file_ext))

    return cqt, beat_times


def extractAll(filename, file_ext):
    if file_ext in ('.wav', '.mp3'):
        # Load the example clip
        y, _ = librosa.load(filename+file_ext, sr=SR, mono=True)
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        tempo, beat_times = librosa.beat.beat_track(y=y_percussive, sr=SR)
        beat_times = completeBeatTimes(beat_times)
        # mfcc
        feat = librosa.feature.mfcc(
            y=y, sr=SR, n_mfcc=15, hop_length=HOP_LENGTH, fmin=27.5)
        feat = librosa.feature.sync(feat,
                                    beat_times,
                                    aggregate=np.median).T
        # cqt
        cqt = np.abs(librosa.core.cqt(
            y=y, sr=SR, hop_length=HOP_LENGTH, resolution=1.0, fmin=27.5,
            n_bins=BINS_PER_OCT*OCTAVES, bins_per_octave=BINS_PER_OCT,
            real=True))
        cqt = librosa.feature.sync(cqt, beat_times, aggregate=np.median).T
        feat = np.column_stack((feat, cqt))
        del cqt

        # Compute chroma features from the harmonic signal
        chromagram = librosa.feature.chroma_stft(y=y_harmonic, sr=SR,
                                                 n_chroma=12, n_fft=N_FFT,
                                                 hop_length=HOP_LENGTH)
        # Aggregate chroma features between beat events
        # We'll use the median value of each feature between beat frames
        chromagram = librosa.feature.sync(chromagram,
                                          beat_times,
                                          aggregate=np.median).T

        feat = np.column_stack((feat, chromagram))
        del chromagram
        beat_times = librosa.frames_to_time(beat_times, sr=SR)
    else:
        raise Exception('{} is not a supported filetype'.format(file_ext))

    return feat, beat_times


def extractFeature(filename, file_ext, feature, scale, round_to, normalize):
    print '\tExtracting Feature {}'.format(feature)
    if feature == 'chroma':
        feat, beat_times = extractChroma(filename, file_ext)
    elif feature == 'mfcc':
        feat, beat_times = extractMFCC(filename, file_ext)
    elif feature == 'cqt':
        feat, beat_times = extractCQT(filename, file_ext)
    elif feature == 'all':
        feat, beat_times = extractAll(filename, file_ext)
    else:
        raise Exception('Feature {} is not supported'.format(feature))

    # pre-processing
    if scale:
        print "\tScaling Data to max == 1"
        feat /= np.max(np.abs(feat), axis=1, keepdims=True)

    if round_to:
        # print "\tRounding to {} decimal places".format(round_to)
        # feat = np.around(feat, round_to)
        print "\tRounding to {} step".format(round_to)
        feat = np.round(feat / round_to) * round_to

    if normalize:
        print "\tNormalizing data"
        mean, std = feat.mean(axis=0), feat.std(axis=0)
        feat = (feat - mean) / std
    feat = np.nan_to_num(feat)

    beat_times = beat_times.astype(int)

    return feat, beat_times


def plotClustering(fullpath, order=1, sr=4, cutoff=.1, n_singv=3,
                   feature='chroma', dim_red='SVD', round_to=0, normalize=1,
                   scale=1, length=4, metric='inertia'):
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
    feat['LPF'] = lpf(feat[feature], cutoff, sr, order)
    feat[dim_red] = dim_red_fn(dim_red, feat[feature], n_singv)
    feat['{}(LPF)'.format(dim_red)] = dim_red_fn(
        dim_red, feat['LPF'], n_singv)
    feat['LPF({})'.format(dim_red)] = lpf(feat[dim_red], cutoff, sr, order)

    # create vars for plotting
    ts = np.arange(0, len(feat[feature]))
    step_size = max(1, int(len(ts) * .01))
    fig = plt.figure(figsize=(36, 18))
    fig.suptitle('feature {} order {}, cutoff {}, sr {}'.format(
        feature, order, cutoff, sr))

    gs = mpl.gridspec.GridSpec(10, 4, width_ratios=[1, 1, 1, 1])
    i = 0
    print "\tPlot data and pre-processing"
    for name in (feature, 'LPF', dim_red , '{}(LPF)'.format(dim_red),
                 'LPF({})'.format(dim_red)):
        data = feat[name]

        data_wide = np.array([feat[name][m:m+length, :]
                              for m in xrange(len(feat[name])-length)])
        data_wide = data_wide.reshape(
            data_wide.shape[0], data_wide.shape[1]*data_wide.shape[2])

        K_MIN, K_MAX = 2, 12
        KM = [KMeans(n_clusters=l, init='k-means++').fit(data_wide)
              for l in xrange(K_MIN, K_MAX+1)]

        scores_bic = [compute_bic(KM[x], data_wide) for x in xrange(len(KM))]
        scores_inertia = [KM[x].inertia_ for x in xrange(len(KM))]
        scores_silhouette = [silhouette_score(data_wide, KM[x].labels_,
                                              metric='euclidean')
                             for x in xrange(len(KM))]

        # get best clusters
        idx_best_bic = find_elbow(np.dstack(
            (xrange(K_MIN, K_MAX+1), scores_bic))[0])
        idx_best_inertia = find_elbow(np.dstack(
            (xrange(K_MIN, K_MAX+1), scores_inertia))[0])
        idx_best_silhouette = find_elbow(np.dstack(
            (xrange(K_MIN, K_MAX+1), scores_silhouette))[0])
        idx_best = int(np.median(
            (idx_best_bic, idx_best_inertia, idx_best_silhouette))) + 1

        k_best = idx_best + K_MIN
        centroids = KM[idx_best].cluster_centers_

        # compute cluster allocations given best K
        centroid_idx = KM[idx_best].labels_

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
        ax_twin = ax.twiny()
        ax_twin.set_xlim(ax.get_xlim())
        ax_twin.set_xticks(range(len(centroid_idx))[::step_size])
        ax_twin.set_xticklabels(centroid_idx[::step_size])
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
    plt.close(fig)


def plotStructure(fullpath, order=1, sr=4, cutoff=.1, n_singv=3, window=8,
                  step_size=2, feature='chroma', dim_red='SVD', as_diff=0,
                  round_to=0, normalize=1, scale=1, medfil_len=0):
    print 'Analyzing {}'.format(fullpath)
    # extract filename, filepath and beat aligned feature
    filename, file_ext = os.path.splitext(fullpath)
    feats = {}
    feats[feature], beat_times = extractFeature(
        filename, file_ext, feature, scale, round_to, normalize)

    # apply low-pass filter and running mean on featgram
    feats['LPF'] = lpf(feats[feature], cutoff, sr, order)

    # perform dimensionality reduction (NMF or SVD)

    if dim_red == 'NMF':
        print '\tNon-Negative Matrix Factorization for {}'.format(feature)
        feats['NMF'] = NMF(n_singv).fit_transform(feats[feature].astype(float))
        feats['NMF(LPF)'] = NMF(n_singv).fit_transform(feats['LPF'])
        feats['LPF(NMF)'] = lpf(feats['NMF'], cutoff, sr, order)
    elif dim_red == 'NMF':
        print '\tSingular Vector Decomposition'
        feats['SVD'] = svd(feats[feature], n_singv, inc_proj=False)
        feats['SVD(LPF)'] = svd(feats['LPF'], n_singv, inc_proj=False)
        feats['LPF(SVD)'] = lpf(feats['SVD'], cutoff, sr, order)
    else:
        raise Exception(
            "{} is not a supported dimensionality reduction".format(dim_red))

    if round_to:
        feats['LPF'] = np.round(
            lpf(feat[feature], cutoff, sr, order) / round_to) * round_to
        feats[dim_red] = np.round(
            dim_red_fn(dim_ref, feats[feature], n_singv) / round_to) * round_to
        feats['{}(LPF)'.format(dim_red)] = np.round(dim_red_fn(dim_red,
            feats['LPF'], n_singv) / round_to) * round_to
        feats['LPF({})'.format(dim_red)] = np.round(
            lpf(feats[dim_red], cutoff, sr, order) / round_to) * round_to
    else:
        feats['LPF'] = lpf(feats[feature], cutoff, sr, order)
        feats[dim_red] = dim_red_fn(dim_red, feats[feature], n_singv)
        feats['{}(LPF)'.format(dim_red)] = dim_red_fn(
            dim_red, feats['LPF'], n_singv)
        feats['LPF({})'.format(dim_red)] = lpf(feats[dim_red], cutoff, sr, order)

    # FFT on all features
    n_fft = 8
    hop_length = 1
    for k, v in feats.items():
        data = np.array([stft(f, n_fft, hop_length)[1:, :] for f in v.T])
        data = data.T
        data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
        feats['FFT({})'.format(k)] = np.abs(data) ** 2

    def compute_distance(i, X, window, step_size):
        return np.sqrt(np.sum((
            X[i:i+window] - X[i+step_size:i+step_size+window]) ** 2))

    distances = {}
    for k, v in feats.items():
        distances[k] = np.array(map(functools.partial(compute_distance,
                                                      X=v,
                                                      window=window,
                                                      step_size=step_size),
                                xrange(0, len(v)+1-window*2)))
    if as_diff:
        print("\tComputing features as difference")
        for k, v in feats.items():
            feats[k] = np.append([0], np.diff(v))

    if medfil_len:
        print("\tApplying median filter {} to distances".format(medfil_len))
        for k, v in distances.items():
            distances[k] = medfilt(v, medfil_len)
    i = 0
    j = 0
    gs = mpl.gridspec.GridSpec(len(feats), 2, width_ratios=[1, 1])
    fig = plt.figure(figsize=(36, 18))
    for k in feats.keys():
        ts = np.arange(0, len(feats[k]))
        step_size = max(4, int(len(ts) * .02))
        data = feats[k]

        if data.shape[1] == 3:
            data = data.reshape(1, data.shape[0], data.shape[1])
        else:
            data = data.T
        if 'FFT' in k:
            step = hop_length * 2
        else:
            step = step_size

        ax = fig.add_subplot(gs[i, j])
        ax.set_title(k)
        ax.imshow(data,
                  interpolation='nearest',
                  origin='low',
                  aspect='auto',
                  cmap=plt.cm.Oranges)
        ax.set_xticks(ts[::step])
        ax.set_xticklabels(beat_times[::step], rotation=60)
        ax.grid(False)

        ax = fig.add_subplot(gs[i+1, j], sharex=ax)
        ax.set_title('{} Distances'.format(k))
        ax.plot(distances[k])
        ax.set_xticks(ts[::step])
        ax.set_xticklabels(beat_times[::step], rotation=60)
        ax.grid(False)
        if j == 1:
            i += 2
        j = (j+1) % 2

    plt.tight_layout()
    plt.savefig("{}_{}_{}_asdiff_{}_wab_{}_r_{}_n_{}_s_{}_{}.png".format(
        filename, feature, cutoff, as_diff, window, round_to, normalize, scale,
    dim_red))
    plt.close(fig)


def plotData(glob_str, window, cutoff, order, sr, step_size, feature, dim_red,
             as_diff, round_to, normalize, scale, medfil_len, length,
             anal_type=0):
    tracks = [x for x in glob.glob(os.path.join(glob_str))]
    if anal_type:
        map(functools.partial(plotClustering,
            cutoff=cutoff, order=order, sr=sr, feature=feature, dim_red=dim_red,
            round_to=round_to, normalize=normalize, scale=scale, length=length),
            tracks)
    else:
        map(functools.partial(plotStructure,
            window=window, cutoff=cutoff, order=order, sr=sr,
            step_size=step_size, feature=feature, dim_red=dim_red,
            as_diff=as_diff, round_to=round_to, normalize=normalize,
            scale=scale, medfil_len=medfil_len), tracks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("glob_str", help="Glob string for input files",
                        type=str)
    parser.add_argument("feature", help="Feature (mfcc, chroma, cqt)",
                        type=str)
    parser.add_argument("as_diff", help="Plot distances or diff of dist?",
                        type=int)
    parser.add_argument("-w", "--window", help='Analysis Window Size',
                        type=int, default=6, nargs='?')
    parser.add_argument("-c", "--cutoff", help='Low-Pass Filter Cuttoff',
                        type=float, default=0.1, nargs='?')
    parser.add_argument("-o", "--order", help='Low-Pass Filter Order',
                        type=int, default=1, nargs='?')
    parser.add_argument("-sr", "--sr", help='Low-Pass Filter Sampling Rate',
                        type=int, default=4, nargs='?')
    parser.add_argument("-ss", "--ss", help='Step-Size',
                        type=int, default=1, nargs='?')
    parser.add_argument("-r", "--round_to", help='Round to decimal',
                        type=float, default=0.25, nargs='?')
    parser.add_argument("-n", "--normalize", help='Normalize data?',
                        type=int, default=1, nargs='?')
    parser.add_argument("-s", "--scale", help='Scale data?',
                        type=int, default=1, nargs='?')
    parser.add_argument("-m", "--medfil", help='Median Filter on Difference',
                        type=int, default=0, nargs='?')
    parser.add_argument("-t", "--anal_type", help='0 Difference, 1 KMeans',
                        type=int, default=0, nargs='?')
    parser.add_argument("-l", "--length", help='Length of code for clustering',
                        type=int, default=4, nargs='?')
    parser.add_argument("-d", "--dim_red", help='SVD or NMF',
                        type=str, default='SVD', nargs='?')


    args = parser.parse_args()
    plotData(
        args.glob_str, args.window, args.cutoff, args.order, args.sr, args.ss,
        args.feature, args.dim_red, bool(args.as_diff), args.round_to, args.normalize,
        args.scale, args.medfil, args.length, args.anal_type)
    exit(0)
