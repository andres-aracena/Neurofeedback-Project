# filters.py
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, hilbert, savgol_filter


# ==========================================
# FILTROS IIR / SOS (Butterworth & Notch)
# ==========================================

def bandpass_sos(x, low, high, order=4, fs=250, padlen=None):
    """
    Filtro Pasa-Banda robusto usando secciones de segundo orden (SOS).
    Más estable que los filtros 'ba' tradicionales para órdenes altos.
    """
    nyq = fs / 2
    lown, highn = low / nyq, high / nyq

    # Ajuste automático de orden para bandas altas para evitar inestabilidad
    if high > 0.6 * nyq:
        order = 3

    sos = butter(order, [lown, highn], btype='band', output='sos')
    # Padlen dinámico: evita errores con ventanas de tiempo muy cortas al inicio
    pad = padlen if padlen is not None else 3 * (max(len(sos), 1))

    return sosfiltfilt(sos, x, padlen=pad)


def check_bandpass_gain(low, high, fs=250, order=4):
    """
    Calcula la ganancia teórica para compensar la atenuación del filtro
    si fuera necesario (útil para visualización exacta).
    """
    # Implementación simplificada para optimización
    # En tiempo real, asumimos ganancia unitaria para Butterworth
    return 0.0


def highpass_sos(x, cutoff=0.5, order=4, fs=250):
    """Elimina la deriva lenta (DC Offset dinámico)."""
    sos = butter(order, cutoff / (fs / 2), btype='highpass', output='sos')
    return sosfiltfilt(sos, x, padlen=24)


def lowpass_sos(x, cutoff, order=4, fs=250):
    """
    Elimina ruido de alta frecuencia (muscular/electrónico).
    Configurado a 100Hz para dejar pasar Gamma limpia.
    """
    nyq = fs / 2
    if cutoff >= nyq: cutoff = nyq - 1.0
    sos = butter(order, cutoff / nyq, btype='low', output='sos')
    return sosfiltfilt(sos, x, padlen=24)


def notch_filter(x, notch_freq=50.0, q=30.0, fs=250):
    """
    Filtro Muesca (Notch) para eliminar ruido de línea.
    Q=30 es selectivo (borra solo 50Hz sin tocar 45Hz o 55Hz).
    """
    b, a = iirnotch(notch_freq, q, fs)
    return filtfilt(b, a, x)


# ==========================================
# PRE-PROCESAMIENTO COMBINADO
# ==========================================

def preprocess_signal(x, fs=250):
    """
    Pipeline de limpieza estándar:
    1. Highpass (0.5Hz) -> Quita deriva base.
    2. Notch (50Hz) -> Quita red eléctrica.
    3. Lowpass (100Hz) -> Limita ruido de alta frecuencia.
    """
    # 1. Quitar DC y deriva lenta
    x = highpass_sos(x, cutoff=0.5, fs=fs)

    # 2. Notch en 50 Hz (Fundamental)
    x = notch_filter(x, notch_freq=50.0, q=20.0, fs=fs)

    # 3. Notch en 100 Hz (Armónico fuerte en tus grabaciones)
    x = notch_filter(x, notch_freq=100.0, q=20.0, fs=fs)

    # 4. Lowpass relajado a 100Hz (Permite Gamma completa 30-90Hz)
    x = lowpass_sos(x, cutoff=100.0, order=4, fs=fs)

    return x


def envelope(x):
    """Calcula la envolvente de amplitud usando Hilbert."""
    return np.abs(hilbert(x))