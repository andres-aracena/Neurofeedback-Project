# postprocess.py
"""
Script de post-procesamiento offline:
- Carga grabaciones guardadas en formato .npz
- Permite seleccionar sesión
- Visualiza los datos usando la misma interfaz gráfica en modo offline
"""

import os, sys, time
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

from processing import update_loop
from plotting import create_ui, connect_channel_controls

# =========================
# Selección de archivo
# =========================
def select_npz_file(folder="recordings"):
    """Devuelve la ruta de un archivo .npz seleccionado por el usuario."""
    files = [f for f in os.listdir(folder) if f.endswith(".npz")]
    if not files:
        raise FileNotFoundError(f"No se encontraron grabaciones en {folder}")

    # selector Qt
    app = QtWidgets.QApplication(sys.argv)
    dlg = QtWidgets.QFileDialog()
    dlg.setDirectory(folder)
    dlg.setNameFilter("NPZ files (*.npz)")
    dlg.setWindowTitle("Seleccionar grabación")
    if dlg.exec_() == QtWidgets.QFileDialog.Accepted:
        return dlg.selectedFiles()[0]
    else:
        sys.exit(0)

# =========================
# Reproducción offline
# =========================
def replay_offline(filepath, update_ms=80, offset=250, mode="wavelet"):
    data = np.load(filepath, allow_pickle=True)
    eeg = data["eeg"]
    fs = int(data["fs"])
    n_ch = int(data["channels"])
    theta_band = tuple(data["theta_band"])
    gamma_band = tuple(data["gamma_band"])
    mode = str(data["mode"])

    n_samples = eeg.shape[1]
    print(f"[INFO] Sesión cargada: {filepath}")
    print(f"  Canales: {n_ch}, Fs={fs} Hz, muestras={n_samples}")

    # Ventana de visualización
    win_sec = 10
    win_samples = win_sec * fs

    # Interfaz gráfica
    pg.setConfigOptions(antialias=True, background='#111218', foreground='w')
    app = QtWidgets.QApplication(sys.argv)
    main, ui = create_ui(n_ch, win_sec, offset, fs)
    ch_sel = {"idx": 0}
    connect_channel_controls(ui, n_ch, lambda idx: ch_sel.update(idx=idx))

    # buffers circulares
    from collections import deque
    buffers = [deque(np.zeros(win_samples), maxlen=win_samples) for _ in range(n_ch)]

    # iterador de datos offline
    cursor = {"pos": 0}
    t0 = time.time()

    def update():
        pos = cursor["pos"]
        if pos + win_samples >= n_samples:
            print("[INFO] Fin de la grabación.")
            timer.stop()
            return

        # llenar buffers
        for i in range(n_ch):
            buffers[i].extend(eeg[i, pos:pos+win_samples])
        cursor["pos"] += fs // (1000 // update_ms)  # avanzar proporcional a update rate

        # actualizar gráficos
        ratio = update_loop(buffers, fs, theta_band, gamma_band, 1e-12,
                     ui, t0, ch_sel["idx"], win_sec, offset, mode)

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(update_ms)

    main.show()
    sys.exit(app.exec_())

# =========================
# Main
# =========================
if __name__ == "__main__":
    filepath = select_npz_file("recordings")
    replay_offline(filepath)
