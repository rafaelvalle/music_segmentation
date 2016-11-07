import os
import numpy as np
import deepdish as dd
import librosa
import gensim as gs
import matplotlib.pylab as plt
from scipy.io import savemat
from sklearn.mixture import DPGMM
from feature_extraction import extractFeature
from params import SR, HOP_LENGTH
plt.ion()

filepath = 'data/M29/Beck-Motorcade.wav'
feature = 'cqt'
beat_sync = False
scale = False
round_to = False
normalize = False

# extract features
file_path, file_ext = os.path.splitext(filepath)
save_path = "{}_{}_beat_{}_scale_{}_round_{}_norm_{}.h5".format(
    filepath, feature, beat_sync, scale, round_to, normalize)
if not os.path.isfile(save_path):
    # extract features
    feats, beat_times = extractFeature(
        file_path, file_ext, feature, scale=1, round_to=0, normalize=0,
        beat_sync=beat_sync, transpose=False, save=True)
else:
    feats = dd.io.load(save_path)
    beat_times = feats['beat_times']
    feats = feats[feature]

# Convert to db
feats_log = librosa.logamplitude(feats, ref_power=feats.max())
# L2 normalize the columns, force features to lie on a sphere!
feats_log_normed = librosa.util.normalize(feats_log, norm=2., axis=0)

savemat(save_path[:-3]+'.mat', dict(feats_log=feats_log))
fig, axes = plt.subplots(3, 1, figsize=(18, 6))
axes[0].set_title(feature)
axes[1].set_title('dB Feature')
axes[2].set_title('Normed(dB Feature)')
axes[0].imshow(feats,
               aspect='auto', origin='low', interpolation='nearest',
               cmap=plt.cm.plasma)
axes[1].imshow(feats_log,
               aspect='auto', origin='low', interpolation='nearest',
               cmap=plt.cm.plasma)
axes[2].imshow(feats_log_normed,
               aspect='auto', origin='low', interpolation='nearest',
               cmap=plt.cm.plasma)
fig.tight_layout()


# Clustering with DP-GMM
n_components = 32
dpgmm = DPGMM(n_components=n_components, tol=1e-3, n_iter=32, alpha=1000,
              covariance_type='diag', verbose=True)
dpgmm.fit(feats_log.T)
preds_proba = dpgmm.predict_proba(feats_log.T)
preds = np.argmax(preds_proba, axis=1)
np.unique(preds)
# resynthesis by sampling from clusters
resynthesis = dpgmm.means_[preds.astype(int), :]

fig, axes = plt.subplots(4, 1, figsize=(18, 8))
axes[0].set_title(feature)
axes[1].set_title('Prediction Probability')
axes[2].set_title('Resynthesis')
axes[3].set_title('Max(Prediction Probability)')

axes[0].imshow(feats_log,
               aspect='auto', origin='low', interpolation='nearest',
               cmap=plt.cm.plasma)
axes[1].imshow(preds_proba.T,
               aspect='auto', origin='low', interpolation='nearest',
               cmap=plt.cm.plasma)
axes[2].imshow(resynthesis.T,
               aspect='auto', origin='low', interpolation='nearest',
               cmap=plt.cm.plasma)
axes[3].plot(preds, ls='none', marker='.')
axes[3].set_xlim(0, preds.shape[0])
axes[3].set_ylim(0, n_components)
fig.tight_layout()