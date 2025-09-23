import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, hilbert, savgol_filter, sosfreqz

def bandpass_sos(x, low, high, order=4, fs=250, padlen=None):
    nyq = fs / 2
    lown, highn = low / nyq, high / nyq
    # Ensancha un poco la transición si la banda está alta
    if high > 0.6 * nyq:
        order = 3  # más estable en altas
    sos = butter(order, [lown, highn], btype='band', output='sos')
    # filtfilt con pad controlado para ventanas cortas
    y = sosfiltfilt(sos, x, padlen=padlen if padlen is not None else 3 * (max(len(sos), 1)))
    return y

def check_bandpass_gain(low, high, fs=250, order=4):
    sos = butter(order, [low/(fs/2), high/(fs/2)], btype='band', output='sos')
    w, h = sosfreqz(sos, worN=2048, fs=fs)
    # Ganancia media en banda
    mask = (w >= low) & (w <= high)
    g_db = 20*np.log10(np.maximum(np.abs(h[mask]), 1e-12))
    return float(np.median(g_db))

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
    #x = smooth_signal(x)
    return x

def envelope(x):
    return np.abs(hilbert(x))
