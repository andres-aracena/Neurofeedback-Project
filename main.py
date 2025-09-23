# main.py
import sys, time, os
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from collections import deque
import numpy as np
from datetime import datetime

from board_manager import init_board, get_eeg_channels
from processing import update_loop
from plotting import create_ui, connect_channel_controls, ConfigDialog
from gamification.corsi import CorsiGame

# =========================
# Configuración inicial con ventana
# =========================
app = QtWidgets.QApplication(sys.argv)

# Opciones disponibles
fs_values = [125, 250]
n_ch_values = [4, 8, 16]
win_sec_values = [5, 10, 15]
mode_values = ["butterworth", "wavelet"]

dlg = ConfigDialog(fs_values, n_ch_values, win_sec_values, mode_values)
if dlg.exec_() == QtWidgets.QDialog.Rejected:
    sys.exit(0)
cfg = dlg.get_config()

FS, N_CH, WIN_SEC, MODE = cfg["FS"], cfg["N_CH"], cfg["WIN_SEC"], cfg["MODE"]

UPDATE_MS = 80
OFFSET = 250

THETA_BAND = (4.0, 8.0)
GAMMA_BAND = (30.0, 100.0)
EPS = 1e-12

# =========================
# Inicializar Board
# =========================
board = init_board()
print('init')
eeg_channels = get_eeg_channels(board, N_CH)
print('channels')

# =========================
# Buffers
# =========================
WIN_SAMPLES = WIN_SEC * FS
buffers = [deque(np.zeros(WIN_SAMPLES), maxlen=WIN_SAMPLES) for _ in range(N_CH)]

# Acumulador de datos crudos (para guardar)
record_data = []

# =========================
# Interfaz gráfica
# =========================
pg.setConfigOptions(antialias=True, background='#111218', foreground='w')
main, ui = create_ui(N_CH, WIN_SEC, OFFSET, FS)
ch_sel = {"idx": 0}
connect_channel_controls(ui, N_CH, lambda new_idx: ch_sel.update(idx=new_idx))

# =========================
# Inicializar juego Corsi
# =========================
game = CorsiGame(grid_size=3, sequence_len=5)

# =========================
# Update loop
# =========================
t0 = time.time()
def update():
    global record_data
    data = board.get_current_board_data(WIN_SAMPLES)
    #print(data)
    if data.shape[1] == 0:
        return

    # Actualizar buffers para visualización
    for i, ch in enumerate(eeg_channels):
        buffers[i].extend(data[ch][-WIN_SAMPLES:])

    # Guardar los datos crudos (en fila por canal)
    record_data.append(data[eeg_channels])

    # ---- Neurofeedback ----
    ratio = update_loop(buffers, FS, THETA_BAND, GAMMA_BAND, EPS,
                 ui, t0, ch_sel["idx"], WIN_SEC, OFFSET, MODE)
    # Enviar ratio al juego
    game.set_brain_ratio(ratio)

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(UPDATE_MS)

# =========================
# Guardado periódico
# =========================
SAVE_INTERVAL = 15  # segundos
last_save = time.time()
save_dir = "recordings"
os.makedirs(save_dir, exist_ok=True)
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

def save_data():
    global record_data, last_save
    if not record_data:
        return
    arr = np.hstack(record_data)  # concat por tiempo
    record_data = []  # vaciar para ahorrar RAM

    filename = os.path.join(save_dir, f"session_{session_id}.npz")
    np.savez_compressed(filename,
                        eeg=arr,
                        fs=FS,
                        channels=N_CH,
                        mode=MODE,
                        theta_band=THETA_BAND,
                        gamma_band=GAMMA_BAND)
    print(f"[INFO] Datos guardados en {filename}")

def periodic_save():
    global last_save
    if time.time() - last_save >= SAVE_INTERVAL:
        save_data()
        last_save = time.time()

save_timer = QtCore.QTimer()
save_timer.timeout.connect(periodic_save)
save_timer.start(1000)

# =========================
# Run
# =========================
if __name__ == '__main__':
    try:
        main.show()
        # Ejecutar el juego en paralelo (otro thread Qt)
        QtCore.QTimer.singleShot(2000, game.run)  # lanzar tras 2 seg
        sys.exit(app.exec_())
    finally:
        save_data()
        board.stop_stream()
        board.release_session()
