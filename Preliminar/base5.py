import sys, time
import numpy as np
from collections import deque
from scipy.signal import butter, sosfiltfilt, hilbert
import pywt
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# =========================
# Parameters
# =========================
FS = 250
N_CH = 8
WIN_SEC = 5
ANA_WIN_SEC = 3
UPDATE_MS = 60

THETA_BAND = (4.0, 8.0)
GAMMA_BAND = (30.0, 100.0)
BW_THETA = THETA_BAND[1] - THETA_BAND[0]
BW_GAMMA = GAMMA_BAND[1] - GAMMA_BAND[0]
GAMMA_BW_COMP = np.sqrt(BW_GAMMA / BW_THETA)
EPS = 1e-12
NYQ = FS / 2

def bandpass_sos(x, low, high, order=6):
    sos = butter(order, [low / NYQ, high / NYQ], btype='band', output='sos')
    return sosfiltfilt(sos, x)

def envelope(x):
    return np.abs(hilbert(x))

# =========================
# BrainFlow Board
# =========================
BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
board.prepare_session()
board.start_stream()
print("Board connected (Synthetic).")

eeg_channels = board.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
eeg_channels = eeg_channels[:N_CH]

# =========================
# Buffers
# =========================
WIN_SAMPLES = WIN_SEC * FS
ANA_SAMPLES = ANA_WIN_SEC * FS
buffers = [deque(np.zeros(WIN_SAMPLES), maxlen=WIN_SAMPLES) for _ in range(N_CH)]

# =========================
# UI
# =========================
app = QtWidgets.QApplication(sys.argv)
pg.setConfigOptions(antialias=True)

win = pg.GraphicsLayoutWidget(title="Neurofeedback UI")
win.resize(1400, 900)

# --- Dropdown for channel selection ---
ch_selector = QtWidgets.QComboBox()
for i in range(N_CH):
    ch_selector.addItem(f"Channel {i+1}")
ch_selector.setCurrentIndex(0)
proxy = QtWidgets.QGraphicsProxyWidget()
proxy.setWidget(ch_selector)
win.addItem(proxy, row=0, col=0)

# --- Raw signals ---
p_raw = win.addPlot(row=1, col=0, colspan=2)
p_raw.setTitle("EEG Raw Signals (8 channels)")
p_raw.setLabel('bottom', 'Time', units='s')
p_raw.setLabel('left', 'Amplitude', units='µV')
p_raw.showGrid(x=True, y=True)
t_axis = np.linspace(-WIN_SEC, 0, WIN_SAMPLES)
curves_raw = [p_raw.plot(t_axis, np.zeros(WIN_SAMPLES), pen=pg.mkPen((i*30, 200, 120), width=1)) for i in range(N_CH)]
OFFSET = 200

# --- Band envelopes ---
p_env = win.addPlot(row=2, col=0)
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

# --- Filtered signal ---
p_filt = win.addPlot(row=2, col=1)
p_filt.setTitle("Filtered Signal (Selected Channel)")
p_filt.setLabel('bottom', 'Time', units='s')
p_filt.setLabel('left', 'Amplitude', units='µV')
p_filt.showGrid(x=True, y=True)
curve_theta = p_filt.plot(t_axis[-ANA_SAMPLES:], np.zeros(ANA_SAMPLES), pen='g')
curve_gamma = p_filt.plot(t_axis[-ANA_SAMPLES:], np.zeros(ANA_SAMPLES), pen='r')

# --- Ratio ---
p_ratio = win.addPlot(row=3, col=0, colspan=2)
p_ratio.setTitle("Theta/Gamma Ratio (Median across channels)")
p_ratio.setLabel('bottom', 'Time', units='s')
p_ratio.setLabel('left', 'Ratio')
p_ratio.showGrid(x=True, y=True)
ratio_t, ratio_y = deque(maxlen=30*1000//UPDATE_MS), deque(maxlen=30*1000//UPDATE_MS)
curve_ratio = p_ratio.plot([], [], pen='b')

# --- Wavelet ---
p_cwt = win.addPlot(row=4, col=0, colspan=2)
p_cwt.setTitle("Wavelet Spectrogram (Selected Channel)")
p_cwt.setLabel('bottom','Time', units='s')
p_cwt.setLabel('left','Frequency', units='Hz')
img_cwt = pg.ImageItem()
p_cwt.addItem(img_cwt)
freqs = np.linspace(2, 80, 60)
scales = pywt.central_frequency('cmor1.5-1.0') * FS / freqs
t_cwt = np.linspace(-ANA_WIN_SEC, 0, ANA_SAMPLES)

t0 = time.time()

# =========================
# Update loop
# =========================
def update():
    global bar_theta, bar_gamma
    data = board.get_current_board_data(max(WIN_SAMPLES, ANA_SAMPLES))
    if data.shape[1] < ANA_SAMPLES:
        return

    # update buffers
    for i, ch in enumerate(eeg_channels):
        buffers[i].extend(data[ch][-WIN_SAMPLES:])

    # raw plot
    for i, curve in enumerate(curves_raw):
        curve.setData(t_axis, np.asarray(buffers[i]) + i*OFFSET)

    # analysis
    theta_env_means, gamma_env_means, ratios = [], [], []
    ch_sel = ch_selector.currentIndex()

    for i in range(N_CH):
        raw_win = np.asarray(buffers[i])[-ANA_SAMPLES:]
        theta = bandpass_sos(raw_win, *THETA_BAND)
        gamma = bandpass_sos(raw_win, *GAMMA_BAND)

        env_theta = envelope(theta)
        env_gamma = envelope(gamma) / (GAMMA_BW_COMP+EPS)

        inst_ratio = env_theta/(env_gamma+EPS)
        ratios.append(np.median(inst_ratio))

        theta_env_means.append(np.median(env_theta))
        gamma_env_means.append(np.median(env_gamma))

        # filtered plots (only selected channel)
        if i == ch_sel:
            curve_theta.setData(t_axis[-ANA_SAMPLES:], theta)
            curve_gamma.setData(t_axis[-ANA_SAMPLES:], gamma)

            coeffs, _ = pywt.cwt(raw_win, scales, 'cmor1.5-1.0', sampling_period=1/FS)
            power = np.abs(coeffs).astype(np.float32)
            img_cwt.setImage(power, autoLevels=False,
                             levels=(np.percentile(power,5), np.percentile(power,95)))
            img_cwt.setRect(QtCore.QRectF(t_cwt[0], freqs[0], ANA_WIN_SEC, freqs[-1]-freqs[0]))

    # update bars
    bar_theta.setOpts(height=np.array(theta_env_means))
    bar_gamma.setOpts(height=np.array(gamma_env_means))

    # update ratio
    ratio_t.append(time.time()-t0)
    ratio_y.append(np.median(ratios))
    curve_ratio.setData(np.fromiter(ratio_t,float), np.fromiter(ratio_y,float))
    p_ratio.setXRange(max(0, ratio_t[-1]-30), ratio_t[-1])

timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(UPDATE_MS)

# run
if __name__ == '__main__':
    try:
        win.show()
        sys.exit(app.exec_())
    finally:
        board.stop_stream()
        board.release_session()
