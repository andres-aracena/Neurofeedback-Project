import sys, time
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from collections import deque
import numpy as np

from board_manager import init_board, get_eeg_channels
from processing import update_plots
from plotting import create_ui

# =========================
# Parámetros
# =========================
FS = 250
N_CH = 8
WIN_SEC = 5
UPDATE_MS = 100
OFFSET = 250

THETA_BAND = (4.0, 8.0)
GAMMA_BAND = (30.0, 100.0)
EPS = 1e-12

# =========================
# Inicializar Board
# =========================
board = init_board()
eeg_channels = get_eeg_channels(board, N_CH)

# =========================
# Buffers
# =========================
WIN_SAMPLES = WIN_SEC * FS
buffers = [deque(np.zeros(WIN_SAMPLES), maxlen=WIN_SAMPLES) for _ in range(N_CH)]

# =========================
# Interfaz gráfica
# =========================
app = QtWidgets.QApplication(sys.argv)
pg.setConfigOptions(antialias=True, background='#111218', foreground='w')

main, ui = create_ui(N_CH, WIN_SEC, OFFSET)
ch_sel = 0

# =========================
# Control de canal
# =========================
def set_channel(idx):
    nonlocal_ch = idx % N_CH  # Recorrido circular
    globals()['ch_sel'] = nonlocal_ch
    ui['lbl_channel'].setText(f"Canal {nonlocal_ch+1}")

ui['btn_prev'].clicked.connect(lambda: set_channel(ch_sel-1))
ui['btn_next'].clicked.connect(lambda: set_channel(ch_sel+1))

# =========================
# Update loop
# =========================
t0 = time.time()

def update():
    global ch_sel
    data = board.get_current_board_data(WIN_SAMPLES)
    if data.shape[1] < WIN_SAMPLES:
        return

    # actualizar buffers
    for i, ch in enumerate(eeg_channels):
        buffers[i].extend(data[ch][-WIN_SAMPLES:])

    # análisis y ploteo
    update_plots(buffers, FS, THETA_BAND, GAMMA_BAND, EPS, ui, t0, ch_sel, WIN_SEC, OFFSET)

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
