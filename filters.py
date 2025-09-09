import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, hilbert, savgol_filter

NYQ = 250 / 2  # default, puede ser sobrescrito en main

def bandpass_sos(x, low, high, order=6, fs=250):
    sos = butter(order, [low / (fs/2), high / (fs/2)], btype='band', output='sos')
    return sosfiltfilt(sos, x)

def highpass_sos(x, cutoff=0.5, order=4, fs=250):
    sos = butter(order, cutoff / (fs/2), btype='highpass', output='sos')
    return sosfiltfilt(sos, x)

def notch_filter(x, notch_freq=50.0, q=30.0, fs=250):
    b, a = iirnotch(notch_freq, q, fs)
    return filtfilt(b, a, x)

def smooth_signal(x, window=21, poly=3):
    if len(x) > window:
        return savgol_filter(x, window_length=window, polyorder=poly)
    return x

def preprocess_signal(x, fs=250):
    x = highpass_sos(x, cutoff=0.5, fs=fs)
    x = notch_filter(x, notch_freq=50, fs=fs)
    x = smooth_signal(x)
    return x

def envelope(x):
    return np.abs(hilbert(x))
