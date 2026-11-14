import numpy as np
import pandas as pd
import mne
from mne.datasets import eegbci
from scipy.signal import welch
import matplotlib.pyplot as plt

mne.set_log_level("WARNING")

SUBJECTS = list(range(1, 11))
RUNS = [1, 2, 3, 4, 5, 6, 7, 8]
WINDOW_SEC = 4.0
LOW_HZ, HIGH_HZ = 1.0, 40.0

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40),
}


def load_run(subject, run):
    path = eegbci.load_data(subject, runs=[run])[0]
    raw = mne.io.read_raw_edf(path, preload=True)
    eegbci.standardize(raw)
    raw.filter(LOW_HZ, HIGH_HZ, fir_design="firwin")
    return raw


def extract_features(windows, sfreq, ch_names):
    freqs, psd = welch(windows, fs=sfreq, nperseg=int(sfreq * 2), axis=-1)
    df = freqs[1] - freqs[0]
    total_mask = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    total = psd[..., total_mask].sum(-1) * df

    feats, names = [], []
    for band, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        power = psd[..., mask].sum(-1) * df
        feats.append(np.log10(power + 1e-12))
        names += [f"{ch}_{band}_logpow" for ch in ch_names]
        feats.append(power / (total + 1e-12))
        names += [f"{ch}_{band}_relpow" for ch in ch_names]

    feats.append(windows.mean(-1))
    names += [f"{ch}_mean" for ch in ch_names]
    feats.append(windows.std(-1))
    names += [f"{ch}_std" for ch in ch_names]

    return np.concatenate(feats, axis=1), names
