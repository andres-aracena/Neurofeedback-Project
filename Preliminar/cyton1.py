import sys
import time
import numpy as np
from collections import deque
from scipy.signal import butter, sosfiltfilt, hilbert
from PyQt5 import QtWidgets
import pyqtgraph as pg

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# =========================
# Parámetros generales
# =========================
FS = 250                              # Frecuencia de muestreo
N_CH = 8                              # Canales a mostrar
WIN_SEC = 5                           # Ventana visible (scroll) en segundos
ANA_WIN_SEC = 3                       # Ventana para análisis (envolventes)
UPDATE_MS = 60                        # Intervalo de actualización de GUI (ms)

# Bandas
THETA_BAND = (4.0, 8.0)
GAMMA_BAND = (30.0, 100.0)
BW_THETA = THETA_BAND[1] - THETA_BAND[0]
BW_GAMMA = GAMMA_BAND[1] - GAMMA_BAND[0]
# Compensación por ancho de banda (amplitud ~ sqrt(banda)): reduce gamma
GAMMA_BW_COMP = np.sqrt(BW_GAMMA / BW_THETA)  # ≈ 3.535

EPS = 1e-12

# =========================
# Filtros
# =========================
NYQ = FS / 2.0

def bandpass_sos(x, low, high, order=6):
    sos = butter(order, [low / NYQ, high / NYQ], btype='band', output='sos')
    return sosfiltfilt(sos, x)

def envelope(x):
    return np.abs(hilbert(x))

# =========================
# BrainFlow (Cyton Board)
# =========================
BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
params.serial_port = "COM3"   # <--- CAMBIA esto al puerto real de tu dongle

board = BoardShim(BoardIds.CYTON_BOARD.value, params)
board.prepare_session()
board.start_stream()
print("Board Cyton conectado.")

# Confirmar canales EEG
eeg_channels = board.get_eeg_channels(BoardIds.CYTON_BOARD.value)
if len(eeg_channels) >= N_CH:
    eeg_channels = eeg_channels[:N_CH]
else:
    times = int(np.ceil(N_CH / len(eeg_channels)))
    eeg_channels = (eeg_channels * times)[:N_CH]


# =========================
# Buffers de datos (scroll)
# =========================
WIN_SAMPLES = WIN_SEC * FS
ANA_SAMPLES = ANA_WIN_SEC * FS

# Deques por canal (más rápido que listas para scrolling)
buffers = [deque(maxlen=WIN_SAMPLES) for _ in range(N_CH)]

# Inicializar con ceros para no ver saltos al inicio
for d in buffers:
    d.extend(np.zeros(WIN_SAMPLES, dtype=np.float64))

# =========================
# UI con PyQtGraph
# =========================
app = QtWidgets.QApplication(sys.argv)
pg.setConfigOptions(antialias=True)

win = pg.GraphicsLayoutWidget(title="Neurofeedback - Theta/Gamma (PyQtGraph)")
win.resize(1200, 800)

# Plot 1: Señales crudas 8 canales (apiladas)
p_raw = win.addPlot(row=0, col=0, colspan=2)
p_raw.setTitle("EEG Crudo (8 canales, scroll)")
p_raw.setLabel('bottom', 'Tiempo', units='s')
p_raw.setLabel('left', 'Amplitud', units='uV')
p_raw.showGrid(x=True, y=True)

curves_raw = []
offset = 200.0  # separación vertical entre canales
t_axis = np.linspace(-WIN_SEC, 0, WIN_SAMPLES)
for ch in range(N_CH):
    curve = p_raw.plot(t_axis, np.zeros(WIN_SAMPLES), pen=pg.mkPen(color=(50 + ch*30, 200, 120, 200), width=1))
    curves_raw.append(curve)

# Plot 2: Envolventes por canal (BarChart)
p_env = win.addPlot(row=1, col=0)
p_env.setTitle("Envolvente media por canal (uV)")
p_env.setLabel('bottom', 'Canal')
p_env.setLabel('left', 'Amplitud', units='uV')
p_env.showGrid(x=True, y=True)

# Barras para theta y gamma
x_idx = np.arange(N_CH)
bar_w = 0.4
bar_theta = pg.BarGraphItem(x=x_idx - bar_w/2, height=np.zeros(N_CH), width=bar_w, brush=(0, 180, 0, 180))
bar_gamma = pg.BarGraphItem(x=x_idx + bar_w/2, height=np.zeros(N_CH), width=bar_w, brush=(220, 50, 50, 180))
p_env.addItem(bar_theta)
p_env.addItem(bar_gamma)

# Plot 3: Señal filtrada (canal 1: theta & gamma)
p_filt = win.addPlot(row=1, col=1)
p_filt.setTitle("Canal 1 filtrado: Theta (verde) / Gamma (rojo)")
p_filt.setLabel('bottom', 'Tiempo', units='s')
p_filt.setLabel('left', 'uV')
p_filt.showGrid(x=True, y=True)

curve_theta = p_filt.plot(t_axis[-ANA_SAMPLES:], np.zeros(ANA_SAMPLES), pen=pg.mkPen('g', width=2))
curve_gamma = p_filt.plot(t_axis[-ANA_SAMPLES:], np.zeros(ANA_SAMPLES), pen=pg.mkPen('r', width=2))

# Plot 4: Ratio theta/gamma (mediana de envolvente)
p_ratio = win.addPlot(row=2, col=0, colspan=2)
p_ratio.setTitle("Ratio Theta/Gamma (mediana de envolvente)")
p_ratio.setLabel('bottom', 'Tiempo', units='s')
p_ratio.setLabel('left', 'Ratio')
p_ratio.showGrid(x=True, y=True)
ratio_history_len = 30 * int(1000 / UPDATE_MS)  # ~30 s de historial
ratio_t = deque(maxlen=ratio_history_len)
ratio_y = deque(maxlen=ratio_history_len)
curve_ratio = p_ratio.plot([], [], pen=pg.mkPen('b', width=2))

t0 = time.time()

# =========================
# Update loop
# =========================
def update():
    # Traer datos recientes (al menos ANA_SAMPLES para análisis)
    global bar_theta, bar_gamma
    data = board.get_current_board_data(max(WIN_SAMPLES, ANA_SAMPLES))
    if data.shape[1] < ANA_SAMPLES:
        return  # aún no hay suficientes muestras

    # Actualizar buffers de scroll por canal
    for i, ch in enumerate(eeg_channels):
        buffers[i].extend(data[ch][-WIN_SAMPLES:])

    # ——————— Dibujar señales crudas apiladas ———————
    # 1) Cálculo offset dinámico
    all_data = np.vstack([np.asarray(buffers[i]) for i in range(N_CH)])
    min_raw, max_raw = all_data.min(), all_data.max()
    data_range = max_raw - min_raw + 1e-9
    offset = 200

    # 2) Plot de cada canal
    for i, curve in enumerate(curves_raw):
        y = np.asarray(buffers[i]) + i * offset
        curve.setData(t_axis, y)

    # 3) Autoajuste del eje Y
    y_min = min_raw + 50
    y_max = min_raw + (N_CH - 1) * offset + data_range * 0.7
    p_raw.setYRange(y_min, y_max )

    # Análisis en ventana de 2 s (últimas ANA_SAMPLES)
    theta_env_means = np.zeros(N_CH)
    gamma_env_means = np.zeros(N_CH)
    channel_ratios = np.zeros(N_CH)

    for i in range(N_CH):
        raw_win = np.asarray(buffers[i], dtype=np.float64)[-ANA_SAMPLES:]

        # Filtros
        theta = bandpass_sos(raw_win, *THETA_BAND, order=6)
        gamma = bandpass_sos(raw_win, *GAMMA_BAND, order=6)

        # Envolventes
        env_theta = envelope(theta)
        env_gamma = envelope(gamma) / (GAMMA_BW_COMP + EPS)  # compensación por banda

        # Normalización por std para evitar dominancias por escala
        env_theta_n = env_theta / (np.std(env_theta) + EPS)
        env_gamma_n = env_gamma / (np.std(env_gamma) + EPS)

        # Ratio por muestra (envolvente), y agregamos con MEDIANA (no mean)
        inst_ratio = env_theta_n / (env_gamma_n + EPS)
        channel_ratios[i] = np.median(inst_ratio)

        # Valores medios de envolventes (sin normalizar) para barras (uV aprox.)
        theta_env_means[i] = np.median(env_theta)
        gamma_env_means[i] = np.median(env_gamma)

        # Para el canal 1 (index 0), ploteo filtrado
        if i == 0:
            curve_theta.setData(t_axis[-ANA_SAMPLES:], theta)
            curve_gamma.setData(t_axis[-ANA_SAMPLES:], gamma)

    # Actualizar barras
    p_env.removeItem(bar_theta)
    p_env.removeItem(bar_gamma)
    # Limitar para que las barras no exploten visualmente
    theta_heights = np.clip(theta_env_means, 0, np.percentile(theta_env_means, 95) + 1e-9)
    gamma_heights = np.clip(gamma_env_means, 0, np.percentile(gamma_env_means, 95) + 1e-9)
    # Re-crear BarGraphItem (pyqtgraph es más rápido así que actualizar cada barra)
    bg_theta = pg.BarGraphItem(x=x_idx - bar_w/2, height=theta_heights, width=bar_w, brush=(0, 180, 0, 180))
    bg_gamma = pg.BarGraphItem(x=x_idx + bar_w/2, height=gamma_heights, width=bar_w, brush=(220, 50, 50, 180))
    p_env.addItem(bg_theta)
    p_env.addItem(bg_gamma)

    bar_theta, bar_gamma = bg_theta, bg_gamma

    # Ratio global como mediana entre canales
    ratio_global = float(np.median(channel_ratios))
    t_now = time.time() - t0
    ratio_t.append(t_now)
    ratio_y.append(ratio_global)
    curve_ratio.setData(np.fromiter(ratio_t, float), np.fromiter(ratio_y, float))
    # Rango automático y suave
    p_ratio.setXRange(max(0, t_now - 30), t_now)
    p_ratio.setYRange(0, max(1.0, np.percentile(ratio_y, 95)))

timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(UPDATE_MS)

# =========================
# Run
# =========================
if __name__ == '__main__':
    try:
        win.show()
        sys.exit(app.exec_())
    finally:
        board.stop_stream()
        board.release_session()
