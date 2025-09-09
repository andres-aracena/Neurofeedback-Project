import time
import numpy as np
import pywt
import pyqtgraph as pg
from filters import bandpass_sos, preprocess_signal, envelope

# =========================
# Wavelet
# =========================
def compute_wavelet(raw_win, fs, win_sec, freqs, img_cwt, cbar, lut, t_cwt):
    """Calcular y actualizar el espectrograma Wavelet."""
    scales = pywt.central_frequency('cmor1.5-1.0') * fs / freqs
    coeffs, _ = pywt.cwt(raw_win, scales, 'cmor1.5-1.0', sampling_period=1/fs)
    power = np.abs(coeffs).astype(np.float32).T

    img_cwt.setImage(
        power,
        autoLevels=False,
        lut=lut,
        levels=(np.percentile(power, 5), np.percentile(power, 95))
    )
    img_cwt.setRect(pg.QtCore.QRectF(t_cwt[0], freqs[0], win_sec, freqs[-1] - freqs[0]))
    cbar.setLevels((np.percentile(power, 5), np.percentile(power, 95)))


# =========================
# Update Loop
# =========================
def update_plots(buffers, fs, theta_band, gamma_band, eps, ui, t0, ch_sel, win_sec, offset=200):
    """
    Actualiza todas las gráficas:
      - Señales crudas (8 canales)
      - Señal filtrada (canal seleccionado)
      - Wavelet (canal seleccionado)
      - Envolventes (barras)
      - Ratio Theta/Gamma
    """

    # Eje temporal para ventana completa
    t_axis = np.linspace(-win_sec, 0, win_sec * fs)

    # --- Señales crudas ---
    for i, curve in enumerate(ui['curves_raw']):
        y = np.asarray(buffers[i])[-win_sec * fs:] + i * offset
        curve.setData(t_axis, y)

    # --- Variables para cálculos de bandas ---
    theta_env_means, gamma_env_means, ratios = [], [], []

    for i in range(len(buffers)):
        raw_win = np.asarray(buffers[i])[-win_sec * fs:]
        raw_win = preprocess_signal(raw_win, fs=fs)

        # Filtrado de bandas
        theta = bandpass_sos(raw_win, *theta_band, fs=fs)
        gamma = bandpass_sos(raw_win, *gamma_band, fs=fs)

        # Envolventes
        env_theta = envelope(theta)
        env_gamma = envelope(gamma)

        # Ratio instantáneo
        inst_ratio = env_theta / (env_gamma + eps)
        ratios.append(np.median(inst_ratio))

        # Valores medios (para barras)
        theta_env_means.append(np.median(env_theta))
        gamma_env_means.append(np.median(env_gamma))

        # --- Señal filtrada + Wavelet (solo canal seleccionado) ---
        if i == ch_sel:
            ui['curve_theta'].setData(t_axis, theta)
            ui['curve_gamma'].setData(t_axis, gamma)
            compute_wavelet(
                raw_win, fs, win_sec,
                ui['freqs'], ui['img_cwt'], ui['cbar'], ui['lut'], ui['t_cwt']
            )

    # --- Barras de envolventes ---
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
