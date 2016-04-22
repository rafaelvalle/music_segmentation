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
from scipy.fftpack import rfft
from scipy.signal import butter, lfilter
from sklearn.decomposition import NMF
import pretty_midi
import librosa
from helpers import cropEdges, reduceDimensionality
import matplotlib.pylab as plt
import seaborn

SR = 11025.0
N_FFT = 4096
HOP_LENGTH = 512
BINS_PER_OCT = 12
OCTAVES = 7


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


def moving_average(a, n=3):
    ret = np.cumsum(a, axis=0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data, axis=0)
    return y


def extractChroma(filename, file_ext):
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
        # Beat track on the percussive signal
        tempo, beat_times = librosa.beat.beat_track(y=y_percussive,
                                                    sr=SR,
                                                    hop_length=HOP_LENGTH)
        # Compute chroma features from the harmonic signal
        chromagram = librosa.feature.chroma_stft(y=y_harmonic, sr=SR,
                                                 n_chroma=12, n_fft=N_FFT,
                                                 hop_length=HOP_LENGTH)
        # Aggregate chroma features between beat events
        # We'll use the median value of each feature between beat frames
        chroma = librosa.feature.sync(chromagram,
                                      beat_times,
                                      aggregate=np.median).T
        # convert beat frame to seconds
        beat_times = librosa.frames_to_time(beat_times, sr=SR)
    else:
        raise Exception('{} is not a supported filetype'.format(file_ext))
    return chroma, beat_times


def extractMFCC(filename, file_ext):
    if file_ext in ('.wav', '.mp3'):
        y, _ = librosa.load(filename+file_ext, sr=SR, mono=True)
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        tempo, beat_times = librosa.beat.beat_track(y=y_percussive, sr=SR)
        mfcc = librosa.feature.mfcc(
            y=y, sr=SR, n_mfcc=15, hop_length=HOP_LENGTH, fmin=27.5)
        mfcc = librosa.feature.sync(mfcc,
                                    beat_times,
                                    aggregate=np.median).T
        beat_times = librosa.frames_to_time(beat_times, sr=SR)
    else:
        raise Exception('{} is not a supported filetype'.format(file_ext))

    return mfcc, beat_times


def extractCQT(filename, file_ext):
    if file_ext in ('.wav', '.mp3'):
        # Load the example clip
        y, _ = librosa.load(filename+file_ext, sr=SR, mono=True)
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        tempo, beat_times = librosa.beat.beat_track(y=y_percussive,
                                                    sr=SR)
        cqt = librosa.core.cqt(
            y=y, sr=SR, hop_length=HOP_LENGTH, resolution=1.0, fmin=27.5,
            n_bins=BINS_PER_OCT*OCTAVES, bins_per_octave=BINS_PER_OCT)
        cqt = librosa.feature.sync(cqt, beat_times, aggregate=np.median).T
        beat_times = librosa.frames_to_time(beat_times, sr=SR)
    else:
        raise Exception('{} is not a supported filetype'.format(file_ext))

    return cqt, beat_times


def plotStructure(fullpath, ma_window=8, order=1, sr=4, cutoff=.1, n_singv=3,
                  window_a=8, window_b=9, feature='chroma', as_diff=0):
    print 'Analyzing {}'.format(fullpath)
    # extract filename, filepath and beat aligned chroma
    filename, file_ext = os.path.splitext(fullpath)
    print '\tExtracting Feature {}'.format(feature)
    if feature == 'chroma':
        feat, beat_times = extractChroma(filename, file_ext)
        # normalize
        feat /= feat.std(axis=0)
    elif feature == 'mfcc':
        feat, beat_times = extractMFCC(filename, file_ext)
        # normalize and remove amplitude column
        feat = feat[:, 1:]
        mean, std = feat.mean(axis=0), feat.std(axis=0)
        feat = (feat - mean) / std
    elif feature == 'cqt':
        feat, beat_times = extractCQT(filename, file_ext)
        # normalize and remove amplitude column
        mean, std = feat.mean(axis=0), feat.std(axis=0)
        feat = (feat - mean) / std
    else:
        raise Exception('Feature {} is not supported'.format(feature))
    beat_times = beat_times.astype(int)

    # apply low-pass filter and running mean on featgram
    feat_lpf = butter_lowpass_filter(feat, cutoff, sr, order)

    # perform dimensionality reduction (NMF)
    if feature == 'chroma':
        print '\tNon-Negative Matrix Factorization'
        feat_red = NMF(n_singv).fit_transform(feat.astype(float))
        feat_lpf_red = NMF(n_singv).fit_transform(feat_lpf)
    else:
        print '\tSingular Vector Decompotision'
        feat_red = reduceDimensionality(
            feat.astype(float), n_singv, inc_proj=False)
        feat_lpf_red = reduceDimensionality(feat_lpf, n_singv, inc_proj=False)

    feat_red_lpf = butter_lowpass_filter(feat_red, cutoff, sr, order)
    feat_ma = moving_average(feat_red, ma_window)

    # define distance functions
    def dist(a, b):
        return np.sqrt(np.sum(np.power((a-b), 2)))

    def distanceFFTa(i):
        return dist(np.power(rfft(feat[i:i+window_a].T), 2),
                    np.power(rfft(feat[i+window_a:i+window_a*2].T), 2))

    def distanceFFTb(i):
        return dist(np.power(rfft(feat[i:i+window_b].T), 2),
                    np.power(rfft(feat[i+window_b:i+window_b*2].T), 2))

    def distanceFFTaLPF(i):
        return dist(np.power(rfft(feat_lpf[i:i+window_a].T), 2),
                    np.power(rfft(feat_lpf[i+window_a:i+window_a*2].T), 2))

    def distanceFFTbLPF(i):
        return dist(np.power(rfft(feat_lpf[i:i+window_b].T), 2),
                    np.power(rfft(feat_lpf[i+window_b:i+window_b*2].T), 2))

    def distanceFFTaRed(i):
        return dist(np.power(rfft(feat_red[i:i+window_a].T), 2),
                    np.power(rfft(feat_red[i+window_a:i+window_a*2].T), 2))

    def distanceFFTbRed(i):
        return dist(np.power(rfft(feat_red[i:i+window_b].T), 2),
                    np.power(rfft(feat_red[i+window_b:i+window_b*2].T), 2))

    def distanceFFTaLPFRed(i):
        return dist(
            np.power(rfft(feat_red_lpf[i:i+window_a].T), 2),
            np.power(rfft(feat_red_lpf[i+window_a:i+window_a*2].T), 2))

    def distanceFFTbLPFRed(i):
        return dist(
            np.power(rfft(feat_red_lpf[i:i+window_b].T), 2),
            np.power(rfft(feat_red_lpf[i+window_b:i+window_b*2].T), 2))

    def distanceRedLPFa(i):
        return dist(
            np.power(feat_lpf_red[i:i+window_a].T, 2),
            np.power(feat_lpf_red[i+window_a:i+window_a*2].T, 2))

    def distanceRedLPFb(i):
        return dist(
            np.power(feat_lpf_red[i:i+window_b].T, 2),
            np.power(feat_lpf_red[i+window_b:i+window_b*2].T, 2))

    def distanceLPFa(i):
        return dist(feat_red_lpf[i:i+window_a],
                    feat_red_lpf[i+window_a:i+window_a*2])

    def distanceLPFb(i):
        return dist(feat_red_lpf[i:i+window_b],
                    feat_red_lpf[i+window_b:i+window_b*2])

    def distanceMAa(i):
        return dist(feat_ma[i:i+window_a],
                    feat_ma[i+window_a:i+window_a*2])

    def distanceMAb(i):
        return dist(feat_ma[i:i+window_b],
                    feat_ma[i+window_b:i+window_b*2])

    # iterate through data at 1 ts and compute distance between adjacent data
    dist_ffta = np.array(
        map(distanceFFTa, xrange(0, len(feat)+1-window_a*2, 1)))
    dist_fftb = np.array(
        map(distanceFFTb, xrange(0, len(feat)+1-window_b*2, 1)))
    dist_ffta_lpf = np.array(
        map(distanceFFTaLPF, xrange(0, len(feat_red_lpf)+1-window_a*2, 1)))
    dist_fftb_lpf = np.array(
        map(distanceFFTbLPF, xrange(0, len(feat_red_lpf)+1-window_b*2, 1)))
    dist_ffta_red = np.array(
        map(distanceFFTaRed, xrange(0, len(feat_red)+1-window_a*2, 1)))
    dist_fftb_red = np.array(
        map(distanceFFTbRed, xrange(0, len(feat_red)+1-window_b*2, 1)))
    dist_ffta_lpf_red = np.array(
        map(distanceFFTaLPFRed, xrange(0, len(feat_red_lpf)+1-window_a*2, 1)))
    dist_fftb_lpf_red = np.array(
        map(distanceFFTbLPFRed, xrange(0, len(feat_red_lpf)+1-window_b*2, 1)))
    dist_lpfa = np.array(
        map(distanceLPFa, xrange(0, len(feat_red_lpf)+1-window_a*2)))
    dist_lpfb = np.array(
        map(distanceLPFb, xrange(0, len(feat_red_lpf)+1-window_b*2)))
    dist_lpf_red_a = np.array(
        map(distanceRedLPFa, xrange(0, len(feat_red_lpf)+1-window_a*2)))
    dist_lpf_red_b = np.array(
        map(distanceRedLPFb, xrange(0, len(feat_red_lpf)+1-window_b*2)))
    dist_ma2 = np.array(
        map(distanceMAa, xrange(0, len(feat_ma)+1-window_a*2)))
    dist_mab = np.array(
        map(distanceMAb, xrange(0, len(feat_ma)+1-window_b*2)))

    if as_diff:
        dist_ffta = np.append([0], np.diff(dist_ffta))
        dist_fftb = np.append([0], np.diff(dist_fftb))
        dist_ffta_lpf = np.append([0], np.diff(dist_ffta_lpf))
        dist_fftb_lpf = np.append([0], np.diff(dist_fftb_lpf))
        dist_ffta_red = np.append([0], np.diff(dist_ffta_red))
        dist_fftb_red = np.append([0], np.diff(dist_fftb_red))
        dist_ffta_lpf_red = np.append([0], np.diff(dist_ffta_lpf_red))
        dist_fftb_lpf_red = np.append([0], np.diff(dist_fftb_lpf_red))
        dist_lpfa = np.append([0], np.diff(dist_lpfa))
        dist_lpfb = np.append([0], np.diff(dist_lpfb))
        dist_lpf_red_a = np.append([0], np.diff(dist_lpf_red_a))
        dist_lpf_red_b = np.append([0], np.diff(dist_lpf_red_b))
        dist_ma2 = np.append([0], np.diff(dist_ma2))
        dist_mab = np.append([0], np.diff(dist_mab))

    ts = np.arange(0, len(feat_lpf))
    ts_ma = np.arange(0, len(feat_ma))
    step_size = len(ts) / 20

    print '\tSaving plots'
    # plot everything
    fig = plt.figure(figsize=(24, 10))
    ax = fig.add_subplot(6, 2, 1)
    ax.set_title('Original Data')
    ax.imshow(
        feat.T,
        interpolation='nearest',
        origin='low',
        aspect='auto',
        cmap=plt.cm.Oranges)
    ax.set_xticks(ts[::step_size])
    ax.set_xticklabels(beat_times[::step_size], rotation=60)
    ax.grid(False)

    ax = fig.add_subplot(6, 2, 2)
    ax.set_title('Low-Pass Filtered Data')
    ax.imshow(
        feat_lpf.T,
        interpolation='nearest',
        origin='low',
        aspect='auto',
        cmap=plt.cm.Oranges)
    ax.set_xticks(ts[::step_size])
    ax.set_xticklabels(beat_times[::step_size], rotation=60)
    ax.grid(False)

    ax = fig.add_subplot(6, 2, 3)
    ax.set_title('NMF({})'.format(feature))
    ax.imshow(
        feat_red.reshape(1, feat_red.shape[0], feat_red.shape[1]),
        interpolation='nearest',
        origin='low',
        aspect='auto',
        cmap=plt.cm.Oranges)
    ax.set_xticks(ts[::step_size])
    ax.set_xticklabels(beat_times[::step_size], rotation=60)
    ax.grid(False)

    ax = fig.add_subplot(6, 2, 4)
    ax.set_title('Low-pass filtered NMF({}), order {}, cutoff {}, sr {}'.format(
        feature, order, cutoff, sr))
    ax.imshow(
        feat_red_lpf.reshape(
            1, feat_red_lpf.shape[0], feat_red_lpf.shape[1]),
        interpolation='nearest',
        origin='low',
        aspect='auto',
        cmap=plt.cm.Oranges)
    ax.set_xticks(ts[::step_size])
    ax.set_xticklabels(beat_times[::step_size], rotation=60)
    ax.grid(False)

    ax = fig.add_subplot(6, 2, 5)
    ax.set_title(
        'Difference on window {} with FFT(LPF({}))'.format(window_a, feature))
    ax.plot(dist_ffta_lpf)
    ax.set_xticks(ts[::step_size])
    ax.set_xticklabels(beat_times[::step_size], rotation=60)

    ax = fig.add_subplot(6, 2, 6)
    ax.set_title(
        'Difference on window {} with FFT(LPF({}))'.format(window_b, feature))
    ax.plot(dist_fftb_lpf)
    ax.set_xticks(ts[::step_size])
    ax.set_xticklabels(beat_times[::step_size], rotation=60)

    ax = fig.add_subplot(6, 2, 7)
    ax.set_title(('Difference on window {} with '
                  'FFT(LPF(NMF({})))').format(window_a, feature))
    ax.plot(dist_ffta_lpf_red)
    ax.set_xticks(ts[::step_size])
    ax.set_xticklabels(beat_times[::step_size], rotation=60)

    ax = fig.add_subplot(6, 2, 8)
    ax.set_title(('Difference on window {} with '
                  'FFT(LPF(NMF({})))').format(window_b, feature))
    ax.plot(dist_fftb_lpf_red)
    ax.set_xticks(ts[::step_size])
    ax.set_xticklabels(beat_times[::step_size], rotation=60)

    ax = fig.add_subplot(6, 2, 9)
    ax.set_title(
        'Difference on window {} with LPF(NMF({}))'.format(window_a, feature))
    ax.plot(dist_lpfa)
    ax.set_xticks(ts[::step_size])
    ax.set_xticklabels(beat_times[::step_size], rotation=60)

    ax = fig.add_subplot(6, 2, 10)
    ax.set_title(
        'Difference on window {} with LPF(NMF({}))'.format(window_b, feature))
    ax.plot(dist_lpfb)
    ax.set_xticks(ts[::step_size])
    ax.set_xticklabels(beat_times[::step_size], rotation=60)

    ax = fig.add_subplot(6, 2, 11)
    ax.set_title(
        'Difference on window {} with NMF(LPF({}))'.format(window_a, feature))
    ax.plot(dist_lpf_red_a)
    ax.set_xticks(ts[::step_size])
    ax.set_xticklabels(beat_times[::step_size], rotation=60)

    ax = fig.add_subplot(6, 2, 12)
    ax.set_title(
        'Difference on window {} with NMF(LPF({}))'.format(window_b, feature))
    ax.plot(dist_lpf_red_b)
    ax.set_xticks(ts[::step_size])
    ax.set_xticklabels(beat_times[::step_size], rotation=60)

    plt.tight_layout()
    plt.savefig("{}_{}_asdiff_{}.png".format(
        filename, feature, as_diff))


def plotData(glob_str, ma_window, anal_window_a, anal_window_b,
             cutoff, order, sr, feature, as_diff):
    tracks = [x for x in glob.glob(os.path.join(glob_str))]
    map(functools.partial(plotStructure,
        ma_window=ma_window, window_a=anal_window_a, window_b=anal_window_b,
        cutoff=cutoff, order=order, sr=sr, feature=feature, as_diff=as_diff),
        tracks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("glob_str", help="Glob string for input files",
                        type=str)
    parser.add_argument("feature", help="Feature (mfcc, chroma, cqt)",
                        type=str)
    parser.add_argument("as_diff", help="Plot distances or diff of dist?",
                        type=int)
    parser.add_argument("-m", "--ma_window", help='Moving Average Window Size',
                        type=int, default=8, nargs='?')
    parser.add_argument("-a", "--anal_window_a", help='Analysis A Window Size',
                        type=int, default=8, nargs='?')
    parser.add_argument("-b", "--anal_window_b", help='Analysis B Window Size',
                        type=int, default=9, nargs='?')
    parser.add_argument("-c", "--cutoff", help='Low-Pass Filter Cuttoff',
                        type=float, default=0.1, nargs='?')
    parser.add_argument("-o", "--order", help='Low-Pass Filter Order',
                        type=int, default=1, nargs='?')
    parser.add_argument("-s", "--sr", help='Low-Pass Filter Sampling Rate',
                        type=int, default=4, nargs='?')

    args = parser.parse_args()

    plotData(
        args.glob_str, args.ma_window, args.anal_window_a, args.anal_window_b,
        args.cutoff, args.order, args.sr, args.feature, bool(args.as_diff))
    exit(0)
