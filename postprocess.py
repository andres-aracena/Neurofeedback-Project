import sys, time, os
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from collections import deque
import numpy as np
from datetime import datetime

from processing import compute_wavelet, update_wavelet_plot
from filters import bandpass_sos, preprocess_signal, check_bandpass_gain, envelope
from plotting import create_ui, connect_channel_controls
from gamification.corsi import CorsiGame


class NPZPlayer:
    def __init__(self, filename, update_ms=80):
        # Cargar datos
        self.data = np.load(filename)
        self.eeg = self.data['eeg']
        self.fs = int(self.data['fs'])
        self.n_ch = int(self.data['channels'])

        # Parámetros de la sesión original
        self.theta_band = tuple(self.data['theta_band']) if 'theta_band' in self.data else (4.0, 8.0)
        self.gamma_band = tuple(self.data['gamma_band']) if 'gamma_band' in self.data else (30.0, 100.0)
        self.mode = str(self.data['mode']) if 'mode' in self.data else "butterworth"
        self.win_sec = int(self.data['win_sec']) if 'win_sec' in self.data else 10

        # Configuración de reproducción
        self.update_ms = update_ms
        self.chunk_size = max(1, int((update_ms / 1000.0) * self.fs))
        self.win_samples = self.win_sec * self.fs
        self.offset = 250
        self.eps = 1e-12

        # Estado de reproducción
        self.cursor_pos = 0
        self.simulated_time = 0.0
        self.is_playing = True

        # Buffers idénticos a main.py
        self.buffers = [deque([0.0] * self.win_samples, maxlen=self.win_samples)
                        for _ in range(self.n_ch)]

        print(f"Reproduciendo: {filename}")
        print(f"Configuración: {self.n_ch} canales, {self.fs} Hz, {self.win_sec}s ventana")
        print(f"Modo: {self.mode}, Theta: {self.theta_band}, Gamma: {self.gamma_band}")
        print(f"Duración total: {self.eeg.shape[1] / self.fs:.1f}s, Chunk: {self.chunk_size} muestras")


def compute_tg_ratio(theta_power, gamma_power, eps=1e-12):
    """Calcula el ratio normalizado Theta/Gamma"""
    return theta_power / (theta_power + gamma_power + eps)


def update_loop_offline(buffers, fs, theta_band, gamma_band, eps,
                        ui, ch_idx, win_sec, offset, mode, simulated_time):
    """
    Versión offline de update_loop que replica EXACTAMENTE el comportamiento
    del procesamiento en tiempo real pero usando tiempo simulado
    """
    # Eje temporal basado en tiempo simulado
    t_axis = np.linspace(simulated_time - win_sec, simulated_time, len(buffers[0]))

    # Parámetros de frecuencia para wavelet (deben coincidir con los de plotting.py)
    freqs = np.linspace(1, 100, 100)  # Mismo que en create_ui
    theta_mask = (freqs >= theta_band[0]) & (freqs <= theta_band[1])
    gamma_mask = (freqs >= gamma_band[0]) & (freqs <= gamma_band[1])

    # --- 1) Señales crudas ---
    for i, curve in enumerate(ui['curves_raw']):
        sig = np.array(buffers[i])
        curve.setData(t_axis, sig + i * offset)

    # Resultados agregados
    theta_pows, gamma_pows, ratios = [], [], []

    # --- Procesar cada canal ---
    for i in range(len(buffers)):
        raw_win = np.array(buffers[i])

        # Solo procesar si tenemos datos suficientes
        if len(raw_win) < win_sec * fs * 0.8:  # Esperar hasta tener al menos 80% de ventana
            theta_pows.append(0.0)
            gamma_pows.append(0.0)
            ratios.append(0.5)  # Valor neutral
            continue

        raw_win = preprocess_signal(raw_win, fs=fs)

        if mode == 'butterworth':
            # Ganancia de banda
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

            # Potencias (usando ventana completa)
            theta_power = np.mean(theta_env ** 2)
            gamma_power = np.mean(gamma_env ** 2)

            ui["p_filt"].setLabel('left', 'Amplitud (µV)')

        else:  # === mode == 'wavelet' ===
            power_norm = compute_wavelet(raw_win, fs, freqs)
            theta_env = np.sqrt(np.mean(power_norm[theta_mask, :], axis=0))
            gamma_env = np.sqrt(np.mean(power_norm[gamma_mask, :], axis=0))

            theta_power = np.mean(theta_env ** 2)
            gamma_power = np.mean(gamma_env ** 2)

            ui["p_filt"].setLabel('left', 'Amplitud Media (µV)')

        # Guardar para barras y ratio
        theta_pows.append(theta_power)
        gamma_pows.append(gamma_power)
        ratio_val = compute_tg_ratio(theta_power, gamma_power, eps)
        ratios.append(ratio_val)

        # --- Señal filtrada del canal seleccionado ---
        if i == ch_idx:
            ui["p_filt"].setTitle(f"Señal filtrada {mode} (Canal {ch_idx + 1})")
            if mode == 'butterworth':
                ui['curve_theta'].setData(t_axis, theta_filt)
                ui['curve_gamma'].setData(t_axis, gamma_filt)
            else:
                ui['curve_theta'].setData(t_axis, theta_env)
                ui['curve_gamma'].setData(t_axis, gamma_env)

            # Espectrograma wavelet
            if mode == 'wavelet':
                spec_db = 10 * np.log10(np.clip(power_norm.T, 1e-18, None)).astype(np.float32)
                ui['p_cwt'].setTitle(f"Espectrograma Wavelet (Canal {ch_idx + 1})")
                update_wavelet_plot(ui, spec_db, freqs, win_sec)

    # --- 2) Barras de potencia ---
    ui['bar_theta'].setOpts(height=np.array(theta_pows))
    ui['bar_gamma'].setOpts(height=np.array(gamma_pows))

    # --- 3) Ratio global ---
    current_ratio = np.median(ratios)

    # Actualizar buffer de ratio (igual que en tiempo real)
    ui['ratio_t'].append(simulated_time)
    ui['ratio_y'].append(current_ratio)

    # Mantener solo últimos 30 segundos en el plot
    keep_mask = np.array(ui['ratio_t']) >= max(0, simulated_time - 30)
    ui['ratio_t'] = list(np.array(ui['ratio_t'])[keep_mask])
    ui['ratio_y'] = list(np.array(ui['ratio_y'])[keep_mask])

    ui['curve_ratio'].setData(ui['ratio_t'], ui['ratio_y'])
    ui['p_ratio'].setXRange(max(0, simulated_time - 30), simulated_time)
    ui['p_ratio'].setYRange(0, 1)

    return current_ratio


def main():
    app = QtWidgets.QApplication(sys.argv)

    # Seleccionar archivo
    filename, _ = QtWidgets.QFileDialog.getOpenFileName(
        None, "Seleccionar archivo de grabación", "recordings", "NPZ files (*.npz)")

    if not filename:
        print("No se seleccionó ningún archivo")
        sys.exit(0)

    # Inicializar reproductor
    player = NPZPlayer(filename)

    # UI idéntica a main.py
    pg.setConfigOptions(antialias=True, background='#111218', foreground='w')
    main_win, ui = create_ui(player.n_ch, player.win_sec, player.offset, player.fs)

    # Control de canales
    ch_sel = {"idx": 0}
    connect_channel_controls(ui, player.n_ch, lambda new_idx: ch_sel.update(idx=new_idx))

    # Juego Corsi
    game = CorsiGame(grid_size=3, sequence_len=5)

    # Timer para updates
    timer = QtCore.QTimer()

    def update():
        if not player.is_playing or player.cursor_pos >= player.eeg.shape[1]:
            if player.cursor_pos >= player.eeg.shape[1]:
                print("✓ Fin de la grabación")
                player.is_playing = False
                if hasattr(ui, 'time_label'):
                    ui.time_label.setText("✓ Grabación completada")
            timer.stop()
            return

        # Obtener chunk actual
        end_pos = player.cursor_pos + player.chunk_size
        if end_pos > player.eeg.shape[1]:
            end_pos = player.eeg.shape[1]
            player.is_playing = False

        chunk = player.eeg[:, player.cursor_pos:end_pos]

        # Actualizar buffers (EXACTAMENTE como en main.py)
        for i in range(player.n_ch):
            player.buffers[i].extend(chunk[i])

        # Calcular tiempo simulado
        player.simulated_time = player.cursor_pos / player.fs

        # Procesamiento offline específico
        ratio = update_loop_offline(
            player.buffers, player.fs, player.theta_band, player.gamma_band,
            player.eps, ui, ch_sel["idx"], player.win_sec,
            player.offset, player.mode, player.simulated_time
        )

        # Enviar ratio al juego (igual que en main.py)
        if ratio is not None and np.isfinite(ratio):
            game.set_brain_ratio(ratio)

        # Actualizar UI
        player.cursor_pos = end_pos
        progress = (player.cursor_pos / player.eeg.shape[1]) * 100

        if hasattr(ui, 'time_label'):
            ui.time_label.setText(
                f"Tiempo: {player.simulated_time:.1f}s | "
                f"Progreso: {progress:.1f}% | "
                f"Ratio: {ratio:.3f}" if ratio is not None else "Calculando..."
            )

    # Configurar timer
    timer.timeout.connect(update)
    timer.start(player.update_ms)

    # Botones de control
    control_widget = QtWidgets.QWidget()
    control_layout = QtWidgets.QHBoxLayout(control_widget)

    btn_play = QtWidgets.QPushButton("Reproducir")
    btn_pause = QtWidgets.QPushButton("Pausar")
    btn_restart = QtWidgets.QPushButton("Reiniciar")

    def play():
        player.is_playing = True
        timer.start(player.update_ms)

    def pause():
        player.is_playing = False
        timer.stop()

    def restart():
        player.cursor_pos = 0
        player.simulated_time = 0.0
        # Reiniciar buffers
        player.buffers = [deque([0.0] * player.win_samples, maxlen=player.win_samples)
                          for _ in range(player.n_ch)]
        # Reiniciar buffers de ratio en UI
        ui['ratio_t'].clear()
        ui['ratio_y'].clear()
        player.is_playing = True
        timer.start(player.update_ms)

    btn_play.clicked.connect(play)
    btn_pause.clicked.connect(pause)
    btn_restart.clicked.connect(restart)

    control_layout.addWidget(btn_play)
    control_layout.addWidget(btn_pause)
    control_layout.addWidget(btn_restart)

    # Layout principal
    main_layout = QtWidgets.QVBoxLayout()
    main_layout.addWidget(control_widget)
    main_layout.addWidget(main_win)

    container = QtWidgets.QWidget()
    container.setLayout(main_layout)

    def start_processing():
        container.show()
        QtCore.QTimer.singleShot(2000, game.run)

    QtCore.QTimer.singleShot(100, start_processing)

    try:
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()