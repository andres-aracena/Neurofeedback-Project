import sys, time
import numpy as np
from collections import deque
from scipy.signal import butter, sosfiltfilt, hilbert, iirnotch, filtfilt
import pywt
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# =========================
# Parameters
# =========================
FS = 250
N_CH = 8
WIN_SEC = 5
UPDATE_MS = 60

THETA_BAND = (4.0, 8.0)
GAMMA_BAND = (30.0, 100.0)
EPS = 1e-12
NYQ = FS / 2

# === Filtros ===
def bandpass_sos(x, low, high, order=6):
    sos = butter(order, [low / NYQ, high / NYQ], btype='band', output='sos')
    return sosfiltfilt(sos, x)

def highpass_sos(x, cutoff=0.5, order=4):
    sos = butter(order, cutoff / NYQ, btype='highpass', output='sos')
    return sosfiltfilt(sos, x)

def notch_filter(x, notch_freq=50.0, q=30.0):
    # Q=30 es estándar → balance entre ancho de rechazo y distorsión
    b, a = iirnotch(notch_freq, q, FS)
    return filtfilt(b, a, x)

def preprocess_signal(x):
    """Filtro previo a la extracción de bandas"""
    x = highpass_sos(x, cutoff=0.5)
    x = notch_filter(x, notch_freq=50)
    return x

def envelope(x):
    return np.abs(hilbert(x))

# =========================
# BrainFlow Board
# =========================
BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)  # cambiar a CYTON_BOARD.value en real
board.prepare_session()
board.start_stream()
print("Board connected.")

eeg_channels = board.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
eeg_channels = eeg_channels[:N_CH]

# =========================
# Buffers
# =========================
WIN_SAMPLES = WIN_SEC * FS
buffers = [deque(np.zeros(WIN_SAMPLES), maxlen=WIN_SAMPLES) for _ in range(N_CH)]

# =========================
# UI
# =========================
app = QtWidgets.QApplication(sys.argv)
pg.setConfigOptions(antialias=True, background='#111218', foreground='w')

main = QtWidgets.QWidget()
main.setWindowTitle("Neurofeedback - Filtering + Wavelet")
main.setStyleSheet("background-color: #111218; color: #e6e6e6;")
layout = QtWidgets.QVBoxLayout(main)

# --- Channel selector with arrow buttons ---
ctrl_layout = QtWidgets.QHBoxLayout()
ctrl_layout.addStretch()

btn_prev = QtWidgets.QToolButton()
btn_prev.setArrowType(QtCore.Qt.LeftArrow)
btn_prev.setStyleSheet("QToolButton { color: white; background: transparent; font-size:24px; }")

lbl_channel = QtWidgets.QLabel("Channel 1")
lbl_channel.setAlignment(QtCore.Qt.AlignCenter)
lbl_channel.setStyleSheet("font-size:18px; color:white; padding:5px;")

btn_next = QtWidgets.QToolButton()
btn_next.setArrowType(QtCore.Qt.RightArrow)
btn_next.setStyleSheet("QToolButton { color: white; background: transparent; font-size:24px; }")

ctrl_layout.addWidget(btn_prev)
ctrl_layout.addWidget(lbl_channel)
ctrl_layout.addWidget(btn_next)
ctrl_layout.addStretch()
layout.addLayout(ctrl_layout)

# --- Graph area ---
graph_area = pg.GraphicsLayoutWidget()
layout.addWidget(graph_area)
main.resize(1500, 900)

# --- Ratio ---
p_ratio = graph_area.addPlot(row=0, col=0, colspan=2)
p_ratio.setTitle("Theta/Gamma Ratio (Median across channels)")
p_ratio.setLabel('bottom', 'Time', units='s')
p_ratio.setLabel('left', 'Ratio')
p_ratio.showGrid(x=True, y=True)
ratio_t, ratio_y = deque(maxlen=30*1000//UPDATE_MS), deque(maxlen=30*1000//UPDATE_MS)
curve_ratio = p_ratio.plot([], [], pen=pg.mkPen('c', width=2))

# --- Raw signals ---
p_raw = graph_area.addPlot(row=1, col=0)
p_raw.setTitle("EEG Raw Signals (8 channels)")
p_raw.setLabel('bottom', 'Time', units='s')
p_raw.setLabel('left', 'Amplitude', units='µV')
p_raw.showGrid(x=True, y=True)
t_axis = np.linspace(-WIN_SEC, 0, WIN_SAMPLES)
curves_raw = [p_raw.plot(t_axis, np.zeros(WIN_SAMPLES), pen=pg.mkPen((i*30, 200, 120), width=1)) for i in range(N_CH)]
OFFSET = 200

# --- Filtered signal ---
p_filt = graph_area.addPlot(row=2, col=0)
p_filt.setTitle("Filtered Signal (Selected Channel)")
p_filt.setLabel('bottom', 'Time', units='s')
p_filt.setLabel('left', 'Amplitude', units='µV')
p_filt.showGrid(x=True, y=True)
curve_theta = p_filt.plot(t_axis, np.zeros(WIN_SAMPLES), pen=pg.mkPen('#99FF00', width=2))
curve_gamma = p_filt.plot(t_axis, np.zeros(WIN_SAMPLES), pen=pg.mkPen('r', width=2))

# --- Band envelopes ---
p_env = graph_area.addPlot(row=1, col=1)
p_env.setTitle("Band Envelopes (Theta / Gamma)")
p_env.setLabel('bottom', 'Channel')
p_env.setLabel('left', 'Envelope Amplitude', units='µV')
p_env.showGrid(x=True, y=True)
x_idx = np.arange(N_CH)
bar_w = 0.4
bar_theta = pg.BarGraphItem(x=x_idx - bar_w/2, height=np.zeros(N_CH), width=bar_w, brush=(0, 200, 0))
bar_gamma = pg.BarGraphItem(x=x_idx + bar_w/2, height=np.zeros(N_CH), width=bar_w, brush=(200, 50, 50))
p_env.addItem(bar_theta)
p_env.addItem(bar_gamma)

# --- Wavelet with colorbar ---
p_cwt = graph_area.addPlot(row=2, col=1)
p_cwt.setTitle("Wavelet Spectrogram (Selected Channel)")
p_cwt.setLabel('bottom','Time', units='s')
p_cwt.setLabel('left','Frequency', units='Hz')
img_cwt = pg.ImageItem()
p_cwt.addItem(img_cwt)
freqs = np.linspace(2, 80, 60)
scales = pywt.central_frequency('cmor1.5-1.0') * FS / freqs
t_cwt = np.linspace(-WIN_SEC, 0, WIN_SAMPLES)
colormap = pg.colormap.get("viridis")
lut = colormap.getLookupTable(0.0, 1.0, 256)

# Add colorbar
cbar = pg.ColorBarItem(values=(0,1), colorMap=colormap)
cbar.setImageItem(img_cwt, insert_in=p_cwt)

# =========================
# Channel control
# =========================
ch_sel = 0
def set_channel(idx):
    global ch_sel
    ch_sel = max(0, min(N_CH-1, idx))
    lbl_channel.setText(f"Channel {ch_sel+1}")
btn_prev.clicked.connect(lambda: set_channel(ch_sel-1))
btn_next.clicked.connect(lambda: set_channel(ch_sel+1))

# =========================
# Update loop
# =========================
t0 = time.time()
def update():
    global bar_theta, bar_gamma, ch_sel
    data = board.get_current_board_data(WIN_SAMPLES)
    if data.shape[1] < WIN_SAMPLES:
        return

    # update buffers
    for i, ch in enumerate(eeg_channels):
        buffers[i].extend(data[ch][-WIN_SAMPLES:])

    # raw plot
    for i, curve in enumerate(curves_raw):
        curve.setData(t_axis, np.asarray(buffers[i]) + i*OFFSET)

    # analysis
    theta_env_means, gamma_env_means, ratios = [], [], []

    for i in range(N_CH):
        raw_win = np.asarray(buffers[i])[-WIN_SAMPLES:]
        raw_win = preprocess_signal(raw_win)

        theta = bandpass_sos(raw_win, *THETA_BAND)
        gamma = bandpass_sos(raw_win, *GAMMA_BAND)

        env_theta = envelope(theta)
        env_gamma = envelope(gamma)

        inst_ratio = env_theta/(env_gamma+EPS)
        ratios.append(np.median(inst_ratio))

        theta_env_means.append(np.median(env_theta))
        gamma_env_means.append(np.median(env_gamma))

        # filtered + wavelet only for selected channel
        if i == ch_sel:
            curve_theta.setData(t_axis, theta)
            curve_gamma.setData(t_axis, gamma)

            coeffs, _ = pywt.cwt(raw_win, scales, 'cmor1.5-1.0', sampling_period=1/FS)
            power = np.abs(coeffs).astype(np.float32)

            # ⚡ IMPORTANTE: transponer para que eje X = tiempo, eje Y = frecuencia
            power = power.T

            img_cwt.setImage(power, autoLevels=False, lut=lut,
                             levels=(np.percentile(power,5), np.percentile(power,95)))
            img_cwt.setRect(QtCore.QRectF(t_cwt[0], freqs[0], WIN_SEC, freqs[-1]-freqs[0]))
            cbar.setLevels((np.percentile(power,5), np.percentile(power,95)))

    # update bars
    bar_theta.setOpts(height=np.array(theta_env_means))
    bar_gamma.setOpts(height=np.array(gamma_env_means))

    # update ratio
    ratio_t.append(time.time()-t0)
    ratio_y.append(np.median(ratios))
    curve_ratio.setData(np.fromiter(ratio_t,float), np.fromiter(ratio_y,float))
    p_ratio.setXRange(max(0, ratio_t[-1]-30), ratio_t[-1])

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(UPDATE_MS)

# run
if __name__ == '__main__':
    try:
        main.show()
        sys.exit(app.exec_())
    finally:
        board.stop_stream()
        board.release_session()
