# processing.py
import time
import numpy as np
import pywt
import pyqtgraph as pg
from filters import bandpass_sos, preprocess_signal, check_bandpass_gain, envelope

# =========================
# Wavelet transform
# =========================
def compute_wavelet(raw_win, fs, freqs, wavelet='cmor1.5-1.0'):
    """
    Continuous Wavelet Transform (CWT) usando Morlet compleja.
    Devuelve matriz de potencia normalizada (n_freqs, n_times) en µV².
    """
    scales = pywt.central_frequency(wavelet) * fs / freqs
    coeffs, _ = pywt.cwt(raw_win, scales, wavelet, sampling_period=1/fs)

    # Potencia instantánea
    power = np.abs(coeffs) ** 2  # µV²

    # Normalización por escala
    return power / (scales[:, None] + 1e-18)


# =========================
# Theta/Gamma Ratio
# =========================
def compute_tg_ratio(theta_power, gamma_power, eps=1e-12):
    """
    Calcula el ratio normalizado Theta/Gamma en [0,1]:
        ratio = Pθ / (Pθ + Pγ + eps)
    """
    return theta_power / (theta_power + gamma_power + eps)


# =========================
# Auxiliares de visualización
# =========================
def update_wavelet_plot(ui, spec_db, freqs, win_sec):
    """
    Actualiza el espectrograma Wavelet en la interfaz.
    """
    ui['img_cwt'].setImage(
        spec_db,
        autoLevels=False,
        lut=ui['lut'],
        levels=(np.percentile(spec_db, 5), np.percentile(spec_db, 95)),
        interpolation=True
    )
    ui['img_cwt'].setRect(pg.QtCore.QRectF(
        ui['t_cwt'][0],
        freqs[0],
        win_sec,
        freqs[-1] - freqs[0]
    ))
    ui['cbar'].setLevels((np.percentile(spec_db, 5), np.percentile(spec_db, 95)))


# =========================
# Update Loop principal
# =========================
def update_loop(buffers, fs, theta_band, gamma_band,
                 eps, ui, t0, ch_sel, win_sec, offset,
                 mode='wavelet'):
    """
    Actualiza todas las gráficas en tiempo real:
      1) Ratio Theta/Gamma global
      2) Señales crudas (N_CH)
      3) Señal filtrada o envolvente wavelet (canal seleccionado)
      4) Espectrograma wavelet (canal seleccionado)
      5) Potencia media de bandas (barras)
    """

    # Eje temporal y máscaras
    t_axis = np.linspace(-win_sec, 0, win_sec * fs)
    freqs = ui['freqs']
    theta_mask = (freqs >= theta_band[0]) & (freqs <= theta_band[1])
    gamma_mask = (freqs >= gamma_band[0]) & (freqs <= gamma_band[1])

    # --- 2) Señales crudas ---
    for i, curve in enumerate(ui['curves_raw']):
        sig = np.asarray(buffers[i])[-win_sec * fs:]
        curve.setData(t_axis, sig + i * offset)

    # Resultados agregados
    theta_pows, gamma_pows, ratios = [], [], []

    # --- Procesar cada canal ---
    for i in range(len(buffers)):
        raw_win = np.asarray(buffers[i])[-win_sec * fs:]
        raw_win = preprocess_signal(raw_win, fs=fs)

        # ---  CWT + potencias ---
        power_norm = compute_wavelet(raw_win, fs, freqs)

        if mode == 'butterworth':
            # Ganancia de banda calculada aquí (más seguro que pasarla como arg)
            gdb_theta = check_bandpass_gain(*theta_band, fs=fs)
            gdb_gamma = check_bandpass_gain(*gamma_band, fs=fs)

            # Filtrado Butterworth
            theta_filt = bandpass_sos(raw_win, *theta_band, fs=fs)
            gamma_filt = bandpass_sos(raw_win, *gamma_band, fs=fs)

            # Corrección de ganancia
            theta_filt /= (10 ** (gdb_theta / 20) + 1e-12)
            gamma_filt /= (10 ** (gdb_gamma / 20) + 1e-12)

            # Envolventes Hilbert
            theta_env = envelope(theta_filt)
            gamma_env = envelope(gamma_filt)

            # Potencias
            theta_power = np.mean(theta_env ** 2)
            gamma_power = np.mean(gamma_env ** 2)

            ui["p_filt"].setLabel('left', 'Amplitud (µV)')
            ui['p_env'].setTitle("Envolventes por canal (Theta / Gamma)")
            ui['p_env'].setLabel('left', 'Amplitud (µV)')

        else:  # === mode == 'wavelet' ===
            theta_env = np.sqrt(np.mean(power_norm[theta_mask, :], axis=0))
            gamma_env = np.sqrt(np.mean(power_norm[gamma_mask, :], axis=0))

            theta_power = np.mean(theta_env ** 2)
            gamma_power = np.mean(gamma_env ** 2)

            ui["p_filt"].setLabel('left', 'Amplitud Media (µV)')
            ui['p_env'].setTitle("Potencia por canal (Theta / Gamma)")
            ui['p_env'].setLabel('left', 'Potencia Instantanea (µV²)')

        # Guardar para barras y ratio
        theta_pows.append(theta_power)
        gamma_pows.append(gamma_power)
        ratios.append(compute_tg_ratio(theta_power, gamma_power, eps))

        # --- 3 y 4) Canal seleccionado ---
        if i == ch_sel:
            ui["p_filt"].setTitle(f"Señal filtrada {mode}(Canal {ch_sel+1})")
            if mode == 'butterworth':
                ui['curve_theta'].setData(t_axis, theta_filt)
                ui['curve_gamma'].setData(t_axis, gamma_filt)
            else:
                ui['curve_theta'].setData(t_axis, theta_env)
                ui['curve_gamma'].setData(t_axis, gamma_env)

            # Espectrograma en dB
            spec_db = 10 * np.log10(np.clip(power_norm.T, 1e-18, None)).astype(np.float32)
            ui['p_cwt'].setTitle(f"Espectrograma Wavelet (Canal {ch_sel+1})")
            update_wavelet_plot(ui, spec_db, freqs, win_sec)

    # --- 5) Barras ---
    ui['bar_theta'].setOpts(height=np.array(theta_pows))
    ui['bar_gamma'].setOpts(height=np.array(gamma_pows))

    # --- 1) Ratio global ---
    t_now = time.time() - t0
    ui['ratio_t'].append(t_now)
    ui['ratio_y'].append(np.median(ratios))
    ui['curve_ratio'].setData(
        np.fromiter(ui['ratio_t'], float),
        np.fromiter(ui['ratio_y'], float)
    )
    ui['p_ratio'].setXRange(max(0, ui['ratio_t'][-1] - 30), ui['ratio_t'][-1])

    return np.median(ratios)
