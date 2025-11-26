# processing.py
import time
import numpy as np
import pywt
import pyqtgraph as pg
from collections import deque
from filters import bandpass_sos, preprocess_signal, envelope


# =========================
# CLASE: Suavizado Temporal
# =========================
class RatioSmoother:
    """
    Mantiene un historial reciente del ratio para eliminar el 'jitter' (temblor).
    Usa la Mediana para ignorar picos repentinos de ruido.
    """

    def __init__(self, history_size=20):
        # history_size=20 @ 80ms update = 1.6 segundos de suavizado
        self.history = deque(maxlen=history_size)

    def update(self, new_val):
        # Protección contra valores matemáticos inválidos
        if np.isnan(new_val) or np.isinf(new_val):
            return self.get_current()

        self.history.append(new_val)
        return self.get_current()

    def get_current(self):
        if not self.history:
            return 0.5
        # La mediana es mejor que el promedio para filtrar ruido impulsivo
        return np.median(self.history)


# Instancia global del suavizador
ratio_smoother = RatioSmoother(history_size=25)


# =========================
# WAVELET TRANSFORM
# =========================
def compute_wavelet(raw_win, fs, freqs, wavelet='cmor1.5-1.0'):
    """Calcula la potencia espectral usando CWT Morlet."""
    scales = pywt.central_frequency(wavelet) * fs / freqs
    coeffs, _ = pywt.cwt(raw_win, scales, wavelet, sampling_period=1 / fs)
    power = np.abs(coeffs) ** 2
    # Normalización por escala para corregir el espectro 1/f
    return power / (scales[:, None] + 1e-18)


# =========================
# CÁLCULO DE RATIO
# =========================
def compute_tg_ratio(theta_power, gamma_power, eps=1e-12):
    """Fórmula: Theta / (Theta + Gamma)"""
    total_power = theta_power + gamma_power + eps
    ratio = theta_power / total_power
    return np.clip(ratio, 0.0, 1.0)


# =========================
# VISUALIZACIÓN
# =========================
def update_wavelet_plot(ui, spec_db, freqs, win_sec):
    """Actualiza el mapa de calor (espectrograma) en la UI."""
    ui['img_cwt'].setImage(
        spec_db,
        autoLevels=False,
        lut=ui['lut'],
        levels=(np.percentile(spec_db, 5), np.percentile(spec_db, 95)),
        interpolation=True
    )
    # Ajustar coordenadas físicas del gráfico
    ui['img_cwt'].setRect(pg.QtCore.QRectF(
        ui['t_cwt'][0],
        freqs[0],
        win_sec,
        freqs[-1] - freqs[0]
    ))


# =========================
# LOOP PRINCIPAL DE ACTUALIZACIÓN
# =========================
def update_loop(buffers, fs, theta_band, gamma_band, eps, ui, t0, ch_sel, win_sec, offset, mode='wavelet'):
    """
    Procesa los buffers de EEG, calcula potencias, actualiza gráficos y retorna el Ratio Suavizado.
    """

    t_axis = np.linspace(-win_sec, 0, win_sec * fs)
    freqs = ui['freqs']

    # --- 1. PREPARACIÓN DE DATOS (DC REMOVAL) ---
    raw_data_matrix = []
    for buff in buffers:
        sig = np.asarray(buff)[-win_sec * fs:]
        if len(sig) > 0:
            # RESTA LA MEDIA INMEDIATA: Centra la señal en 0 uV
            sig = sig - np.mean(sig)
        raw_data_matrix.append(sig)

    # --- 2. VISUALIZACIÓN SEÑAL CRUDA ---
    sig_disp = raw_data_matrix[ch_sel]
    ui['curve_raw'].setData(t_axis, sig_disp)
    # Escala fija tolerante para ver saturación
    if len(sig_disp) > 0:
        ui['p_raw'].setYRange(-200, 200)

    # Arrays para acumular resultados por canal
    theta_pows, gamma_pows, raw_ratios = [], [], []

    # Máscaras booleanas para Wavelet (Pre-calculadas)
    mask_theta = (freqs >= theta_band[0]) & (freqs <= theta_band[1])
    mask_gamma = (freqs >= gamma_band[0]) & (freqs <= gamma_band[1])

    # --- 3. PROCESAMIENTO CANAL POR CANAL ---
    for i, raw_win in enumerate(raw_data_matrix):
        # Filtros de limpieza (Notch 50Hz ya aplicado aquí si filters.py está correcto)
        clean_win = preprocess_signal(raw_win, fs=fs)

        theta_power = 0
        gamma_power = 0

        # Variables para visualizar curvas
        theta_viz = np.zeros_like(clean_win)
        gamma_viz = np.zeros_like(clean_win)

        # === OPCIÓN WAVELET (Recomendada) ===
        if mode == 'wavelet':
            power_norm = compute_wavelet(clean_win, fs, freqs)

            # Potencia Theta
            if np.any(mask_theta):
                # Promedio en banda Theta
                theta_vec = np.mean(power_norm[mask_theta, :], axis=0)
                theta_power = np.mean(theta_vec)
                theta_viz = np.sqrt(theta_vec)  # Para visualización (amplitud)

            # Potencia Gamma (Banda Completa)
            if np.any(mask_gamma):
                # Promedio en banda Gamma completa
                gamma_vec = np.mean(power_norm[mask_gamma, :], axis=0)
                gamma_power = np.mean(gamma_vec)
                gamma_viz = np.sqrt(gamma_vec)  # Para visualización (amplitud)

            # Visualización Espectrograma (Solo canal seleccionado)
            if i == ch_sel:
                spec_db = 10 * np.log10(np.clip(power_norm.T, 1e-18, None)).astype(np.float32)
                update_wavelet_plot(ui, spec_db, freqs, win_sec)

        # === OPCIÓN BUTTERWORTH (Más rápida) ===
        else:
            # Theta
            t_sig = bandpass_sos(clean_win, *theta_band, fs=fs)
            t_env = envelope(t_sig)
            theta_power = np.mean(t_env ** 2)
            theta_viz = t_env

            # Gamma (Banda Completa)
            g_sig = bandpass_sos(clean_win, *gamma_band, fs=fs)
            g_env = envelope(g_sig)
            gamma_power = np.mean(g_env ** 2)
            gamma_viz = g_env

        theta_pows.append(theta_power)
        gamma_pows.append(gamma_power)

        # Ratio CRUDO de este canal (instantáneo)
        raw_ratios.append(compute_tg_ratio(theta_power, gamma_power, eps))

        # Visualización de curvas filtradas (Solo canal seleccionado)
        if i == ch_sel:
            ui['curve_theta'].setData(t_axis, theta_viz)
            ui['curve_gamma'].setData(t_axis, gamma_viz)

    # --- 4. AGREGACIÓN Y SUAVIZADO ---

    # A. Mediana Espacial:
    spatial_ratio = np.median(raw_ratios) if raw_ratios else 0.5

    # B. Suavizado Temporal:
    smoothed_ratio = ratio_smoother.update(spatial_ratio)

    # --- 5. ACTUALIZACIÓN UI FINAL ---
    ui['bar_theta'].setOpts(height=np.array(theta_pows))
    ui['bar_gamma'].setOpts(height=np.array(gamma_pows))

    t_now = time.time() - t0
    ui['ratio_t'].append(t_now)

    # Graficamos el ratio suavizado
    ui['ratio_y'].append(smoothed_ratio)

    ui['curve_ratio'].setData(
        np.fromiter(ui['ratio_t'], float),
        np.fromiter(ui['ratio_y'], float)
    )
    ui['p_ratio'].setXRange(max(0, ui['ratio_t'][-1] - 30), ui['ratio_t'][-1])

    return smoothed_ratio