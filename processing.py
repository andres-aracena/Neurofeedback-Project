# processing.py
import time
import numpy as np
import pywt
import pyqtgraph as pg
from filters import bandpass_sos, preprocess_signal


# =========================
# Wavelet + Potencias
# =========================
def compute_wavelet_and_powers(raw_win, fs, freqs, theta_band, gamma_band):
    """
    Calcula CWT, espectrograma y potencias theta/gamma en una sola pasada.
    Devuelve:
      - power: matriz (n_freqs, n_times)
      - theta_power: potencia media en banda theta
      - gamma_power: potencia media en banda gamma
    """
    # Escalas del wavelet complejo Morlet
    scales = pywt.central_frequency('cmor1.5-1.0') * fs / freqs
    coeffs, _ = pywt.cwt(raw_win, scales, 'cmor1.5-1.0', sampling_period=1/fs)

    # Potencia espectral
    power = np.abs(coeffs) ** 2

    # Selección de bandas
    theta_mask = (freqs >= theta_band[0]) & (freqs <= theta_band[1])
    gamma_mask = (freqs >= gamma_band[0]) & (freqs <= gamma_band[1])

    theta_power = np.mean(power[theta_mask, :])
    gamma_power = np.mean(power[gamma_mask, :])

    return power, theta_power, gamma_power


# =========================
# Ratio Theta/Gamma
# =========================
def compute_tg_ratio(theta_power, gamma_power, eps=1e-12):
    """
    Calcula la relación Theta/Gamma normalizada:
        ratio = Pθ / (Pθ + Pγ + eps)
    """
    return theta_power / (theta_power + gamma_power + eps)


# =========================
# Update Loop
# =========================
def update_plots(buffers, fs, theta_band, gamma_band, eps, ui, t0, ch_sel, win_sec, offset):
    """
    Actualiza todas las gráficas:
      - Señales crudas (8 canales)
      - Señal filtrada (canal seleccionado)
      - Wavelet (canal seleccionado)
      - Envolventes (barras)
      - Ratio Theta/Gamma
    """

    # Eje temporal
    t_axis = np.linspace(-win_sec, 0, win_sec * fs)

    # --- Señales crudas ---
    for i, curve in enumerate(ui['curves_raw']):
        y = np.asarray(buffers[i])[-win_sec * fs:] + i * offset
        curve.setData(t_axis, y)

    # --- Variables ---
    theta_env_means, gamma_env_means, ratios = [], [], []

    # --- Procesamiento por canal ---
    for i in range(len(buffers)):
        raw_win = np.asarray(buffers[i])[-win_sec * fs:]
        raw_win = preprocess_signal(raw_win, fs=fs)

        # Wavelet + potencias en una sola pasada
        power, theta_power, gamma_power = compute_wavelet_and_powers(
            raw_win, fs, ui['freqs'], theta_band, gamma_band
        )

        # Ratio normalizado
        ratio_val = compute_tg_ratio(theta_power, gamma_power, eps=eps)
        ratios.append(ratio_val)

        # Guardar valores medios (para barras)
        theta_env_means.append(theta_power)
        gamma_env_means.append(gamma_power)

        # --- Señal filtrada + Wavelet (solo canal seleccionado) ---
        if i == ch_sel:
            ui["p_filt"].setTitle(f"Señal filtrada (Canal {ch_sel + 1})")
            ui["p_cwt"].setTitle(f"Espectrograma Wavelet (Canal {ch_sel + 1})")

            # Filtros Butterworth solo para visualización
            theta = bandpass_sos(raw_win, *theta_band, fs=fs)
            gamma = bandpass_sos(raw_win, *gamma_band, fs=fs)
            ui['curve_theta'].setData(t_axis, theta)
            ui['curve_gamma'].setData(t_axis, gamma)

            # Actualizar espectrograma
            power_T = power.T.astype(np.float32)  # (n_times, n_freqs)
            ui['img_cwt'].setImage(
                power_T,
                autoLevels=False,
                lut=ui['lut'],
                levels=(np.percentile(power_T, 5), np.percentile(power_T, 95))
            )
            ui['img_cwt'].setRect(pg.QtCore.QRectF(
                ui['t_cwt'][0], ui['freqs'][0], win_sec, ui['freqs'][-1] - ui['freqs'][0]
            ))
            ui['cbar'].setLevels((np.percentile(power_T, 5), np.percentile(power_T, 95)))

    # --- Barras ---
    ui['bar_theta'].setOpts(height=np.array(theta_env_means))
    ui['bar_gamma'].setOpts(height=np.array(gamma_env_means))

    # --- Ratio global ---
    t_now = time.time() - t0
    ui['ratio_t'].append(t_now)
    ui['ratio_y'].append(np.median(ratios))
    ui['curve_ratio'].setData(
        np.fromiter(ui['ratio_t'], float),
        np.fromiter(ui['ratio_y'], float)
    )
    ui['p_ratio'].setXRange(max(0, ui['ratio_t'][-1] - 30), ui['ratio_t'][-1])