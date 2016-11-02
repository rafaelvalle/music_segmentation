import os
import pdb
import numpy as np
import deepdish as dd
import librosa
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
import pretty_midi
from helpers import completeBeatTimes
from params import SR, N_FFT, HOP_LENGTH, BINS_PER_OCT, OCTAVES


def beatSyncFeature(feature, audio, sr, hop_length):
    # Aggregate feature between beat events
    fps = SR/HOP_LENGTH
    beat_proc = DBNBeatTrackingProcessor(fps=100)
    beat_act = RNNBeatProcessor()(audio)
    beat_times = beat_proc(beat_act)
    # We'll use the median value of each feature between beat frames
    feature = librosa.feature.sync(feature,
                                   (beat_times*fps).astype(int),
                                   aggregate=np.median)
    return feature, beat_times

"""
def beatSyncFeature(feature, y_percussive, sr, hop_length, step_size=4):
    # Beat track on the percussive signal
    tempo, beat_times = librosa.beat.beat_track(y=y_percussive,
                                                sr=SR,
                                                hop_length=HOP_LENGTH)

    beat_times = beat_times[::step_size]
    # apply heuristics to have all beats
    beat_times = completeBeatTimes(beat_times)

    # Aggregate feature between beat events
    # We'll use the median value of each feature between beat frames
    feature = librosa.feature.sync(feature,
                                   beat_times,
                                   aggregate=np.median)
    # convert beat frame to seconds
    beat_times = librosa.frames_to_time(beat_times, sr=SR)
    return feature, beat_times
"""

def extractChroma(filename, file_ext, beat_sync, transpose=True):
    if file_ext == ".mid":
        # load file and extract chromagram as bool and transpose to C
        try:
            data = pretty_midi.PrettyMIDI(filename+file_ext)
        except:
            return
        beat_times = data.get_beats()
        chroma = data.get_chroma(beat_times).astype(bool).T
    elif file_ext in ('.wav', '.mp3'):
        # Load the example clip
        y, _ = librosa.load(filename+file_ext, sr=SR, mono=True)

        # Separate harmonics and percussives into two waveforms
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # Compute chroma features from the harmonic signal
        chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=SR,
                                             n_chroma=12, n_fft=N_FFT,
                                             hop_length=HOP_LENGTH)
        beat_times = None
        if beat_sync:
            chroma, beat_times = beatSyncFeature(chroma, y_percussive,
                                                 SR, HOP_LENGTH)
        # transpose such that rows are features
        if transpose:
            chroma = chroma.T
    else:
        raise Exception('{} is not a supported filetype'.format(file_ext))
    return chroma, beat_times


def extractMFCC(filename, file_ext, beat_sync, transpose=True):
    if file_ext in ('.wav', '.mp3'):
        y, _ = librosa.load(filename+file_ext, sr=SR, mono=True)

        mfcc = librosa.feature.mfcc(
            y=y, sr=SR, n_mfcc=15, hop_length=HOP_LENGTH, fmin=27.5)

        beat_times = None
        if beat_sync:
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            mfcc, beat_times = beatSyncFeature(
                mfcc, y_percussive, SR, HOP_LENGTH
                )

        # transpose such that rows are features
        if transpose:
            mfcc = mfcc.T
    else:
        raise Exception('{} is not a supported filetype'.format(file_ext))

    return mfcc, beat_times


def extractCQT(filename, file_ext, beat_sync, transpose=True):
    if file_ext in ('.wav', '.mp3'):
        # Load the example clip
        y, _ = librosa.load(filename+file_ext, sr=SR, mono=True)

        cqt = np.abs(librosa.core.cqt(
            y=y, sr=SR, hop_length=HOP_LENGTH, resolution=1.0, fmin=27.5,
            n_bins=BINS_PER_OCT*OCTAVES,
            bins_per_octave=BINS_PER_OCT, real=False))

        beat_times = None
        if beat_sync:
            """
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            cqt, beat_times = beatSyncFeature(
                cqt, y_percussive, R, HOP_LENGTH)
            """
            cqt, beat_times = beatSyncFeature(
                cqt, filename+file_ext, SR, HOP_LENGTH)
        # transpose such that rows are features
        if transpose:
            cqt = cqt.T
    else:
        raise Exception('{} is not a supported filetype'.format(file_ext))

    return cqt, beat_times


def extractPS(filename, file_ext, beat_sync, transpose=True):
    if file_ext in ('.wav', '.mp3'):
        # Load the example clip
        y, _ = librosa.load(filename+file_ext, sr=SR, mono=True)

        y_harmonic, y_percussive = librosa.effects.hpss(y)
        ps = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))**2

        beat_times = None
        if beat_sync:
            ps, beat_times = beatSyncFeature(ps, y_percussive,
                                             SR, HOP_LENGTH)
        # transpose such that rows are features
        if transpose:
            ps = ps.T
    else:
        raise Exception('{} is not a supported filetype'.format(file_ext))

    return ps, beat_times


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


def extractFeature(filename, file_ext, feature, scale, round_to, normalize,
                   beat_sync=True, transpose=True, save=False):
    # check if feature has been saved already
    save_path = "{}_{}_beat_{}_scale_{}_round_{}_norm_{}.h5".format(
    filename, feature, beat_sync, scale, round_to, normalize)
    if os.path.isfile(save_path):
        feat = dd.io.load(save_path)
        beat_times = feat['beat_times']
        feat = feat[feature]
    else:
        print '\tExtracting Feature {}'.format(feature)
        if feature == 'chroma':
            feat, beat_times = extractChroma(
                filename, file_ext, beat_sync, transpose)
        elif feature == 'mfcc':
            feat, beat_times = extractMFCC(filename, file_ext, beat_sync, transpose)
        elif feature == 'cqt':
            feat, beat_times = extractCQT(filename, file_ext, beat_sync, transpose)
        elif feature == 'ps':
            feat, beat_times = extractPS(filename, file_ext, beat_sync, transpose)
        elif feature == 'all':
            feat, beat_times = extractAll(filename, file_ext, beat_sync, transpose)
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

        if save:
            dd.io.save(
                "{}_{}_beat_{}_scale_{}_round_{}_norm_{}.h5".format(
                    filename, feature, beat_sync, scale, round_to, normalize),
                {feature: feat, 'beat_times': beat_times}
                )

    return feat, beat_times
