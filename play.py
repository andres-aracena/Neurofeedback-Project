import sys, time, os, csv
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from collections import deque
import numpy as np
import pandas as pd

# Importamos TUS módulos locales
try:
    from processing import update_loop
    from plotting import create_ui, connect_channel_controls
except ImportError as e:
    print(f"❌ Error crítico: Faltan módulos (processing.py, plotting.py).")
    sys.exit(1)


class ReplayPlayer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Configuración Base
        self.fs = 250
        self.n_ch = 8
        self.win_sec = 10
        self.update_ms = 30  # 25 FPS para fluidez

        self.theta_band = (4.0, 8.0)
        self.gamma_band = (30.0, 80.0)
        self.eps = 1e-12
        self.offset = 250
        self.mode = 'wavelet'

        # Estado
        self.playing = False
        self.current_idx = 0
        self.data_len = 0
        self.df = None
        self.raw_data = None

        self.setup_ui_structure()
        self.load_file()

    def setup_ui_structure(self):
        self.setWindowTitle("Neurofeedback Replay System - Universal Player")
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # 1. Área de Gráficos
        self.plot_container = QtWidgets.QWidget()
        self.layout.addWidget(self.plot_container, stretch=1)

        # 2. Controles
        control_panel = QtWidgets.QWidget()
        control_panel.setStyleSheet("background-color: #2b2b2b; color: white;")
        cp_layout = QtWidgets.QHBoxLayout(control_panel)

        self.btn_play = QtWidgets.QPushButton("▶ Reproducir")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setStyleSheet("background-color: #4CAF50; font-weight: bold; padding: 6px;")

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.sliderPressed.connect(self.slider_pressed)
        self.slider.sliderReleased.connect(self.slider_released)
        self.slider.valueChanged.connect(self.slider_moved)

        self.lbl_time = QtWidgets.QLabel("00:00 / 00:00")
        self.lbl_info = QtWidgets.QLabel("Esperando archivo...")
        self.lbl_info.setStyleSheet("color: #aaaaaa; font-size: 11px; margin-left: 10px;")

        cp_layout.addWidget(self.btn_play)
        cp_layout.addWidget(self.lbl_time)
        cp_layout.addWidget(self.slider)

        self.layout.addWidget(control_panel)
        self.layout.addWidget(self.lbl_info)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

    def load_file(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Abrir Grabación", "", "Archivos de Datos (*.csv *.npz)"
        )
        if not filename: sys.exit(0)

        print(f"📂 Cargando: {os.path.basename(filename)}...")

        try:
            ext = os.path.splitext(filename)[1].lower()
            if ext == '.npz':
                self._process_npz(filename)
            else:
                self._process_csv(filename)

            # Post-Carga
            if self.df is None or self.raw_data is None:
                raise ValueError("No se pudieron cargar datos válidos.")

            self.data_len = len(self.df)
            self.n_ch = self.raw_data.shape[1]

            # Estimar FS
            if len(self.df) > 1:
                diffs = np.diff(self.df['timestamp'].values)
                valid = diffs[diffs > 0]
                if len(valid) > 0:
                    self.fs = int(round(1.0 / np.median(valid)))

            print(f"✅ Listo: {self.n_ch} canales, {self.fs} Hz, {self.data_len} muestras.")

            # Reset Buffers
            self.win_samples = self.win_sec * self.fs
            self.buffers = [deque([0.0] * self.win_samples, maxlen=self.win_samples) for _ in range(self.n_ch)]

            self.init_plotting_ui()
            self.slider.setRange(0, self.data_len)
            self.update_info_label()
            self.t0 = time.time()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Fallo al cargar:\n{e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # ==========================================
    # CARGA ROBUSTA (De-duplicación y Anti-Zero)
    # ==========================================

    def _process_csv(self, filepath):
        data_rows = []
        last_ts = -1.0
        repaired_zeros = 0
        dropped_duplicates = 0

        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            ch_cols = [c for c in headers if c.startswith('channel_')]
            meta_cols = [c for c in headers if not c.startswith('channel_')]

            last_valid_eeg = [0.0] * len(ch_cols)

            for row in reader:
                try:
                    ts = float(row['timestamp'])

                    # 1. Filtro de duplicados
                    if ts <= last_ts:
                        dropped_duplicates += 1
                        continue

                    # 2. Reparación de ceros (Sample & Hold)
                    current_eeg = []
                    is_zero = True
                    for col in ch_cols:
                        val = float(row[col])
                        current_eeg.append(val)
                        if abs(val) > 1e-6: is_zero = False

                    if is_zero and len(ch_cols) > 0:
                        current_eeg = list(last_valid_eeg)
                        repaired_zeros += 1
                    else:
                        last_valid_eeg = list(current_eeg)

                    # Construir fila limpia
                    clean_row = {'timestamp': ts, **{k: row.get(k, '') for k in meta_cols if k != 'timestamp'}}
                    for i, col in enumerate(ch_cols): clean_row[col] = current_eeg[i]

                    data_rows.append(clean_row)
                    last_ts = ts
                except ValueError:
                    continue

        self.df = pd.DataFrame(data_rows)
        self._finalize_processing(ch_cols, dropped_duplicates, repaired_zeros)

    def _process_npz(self, filepath):
        try:
            data = np.load(filepath, allow_pickle=True)
        except:
            data = np.load(filepath, allow_pickle=False)

        if 'eeg_data' in data:
            eeg = data['eeg_data']
        elif 'eeg' in data:
            eeg = data['eeg']
        else:
            raise ValueError("NPZ sin datos EEG")

        if eeg.shape[0] < eeg.shape[1]: eeg = eeg.T

        min_len = eeg.shape[0]

        # Reconstruir DataFrame
        df_dict = {}
        if 'timestamps' in data and len(data['timestamps']) >= min_len:
            df_dict['timestamp'] = data['timestamps'][:min_len]
        else:
            fs_est = int(data.get('fs', 250)) if 'fs' in data else 250
            df_dict['timestamp'] = np.arange(min_len) / fs_est

        ch_cols = []
        for i in range(eeg.shape[1]):
            c = f'channel_{i + 1}'
            df_dict[c] = eeg[:, i]
            ch_cols.append(c)

        r = data.get('neurofeedback_ratios', data.get('ratios', np.zeros(min_len)))
        if len(r) > min_len:
            r = r[:min_len]
        elif len(r) < min_len:
            r = np.pad(r, (0, min_len - len(r)), 'edge')
        df_dict['ratio'] = r

        states = np.full(min_len, 'unknown', dtype=object)
        energies = np.zeros(min_len)
        missions = np.full(min_len, '', dtype=object)

        if 'game_states' in data:
            gs = data['game_states']
            if len(gs) > 0:
                factor = min_len / len(gs)
                try:
                    indices = (np.arange(min_len) / factor).astype(int)
                    indices = np.clip(indices, 0, len(gs) - 1)
                    if isinstance(gs[0], dict):
                        s_list = [g.get('game_state', 'unknown') for g in gs]
                        e_list = [g.get('energy', 0) for g in gs]
                        m_list = [g.get('mission_state', '') for g in gs]
                        states = np.array(s_list)[indices]
                        energies = np.array(e_list)[indices]
                        missions = np.array(m_list)[indices]
                except:
                    pass

        df_dict['game_state'] = states
        df_dict['energy'] = energies
        df_dict['mission_state'] = missions

        self.df = pd.DataFrame(df_dict)
        self.mode = str(data.get('mode', 'wavelet'))

        self._finalize_processing(ch_cols, 0, 0)

    def _finalize_processing(self, ch_cols, dups, zeros):
        # 1. Limpieza de señal (RAW -> uV y DC Removal)
        eeg_matrix = self.df[ch_cols].values

        if np.max(np.abs(eeg_matrix)) > 50000:
            print("   -> Detectado formato RAW. Escalando...")
            eeg_matrix = eeg_matrix * 0.02235

        eeg_matrix = eeg_matrix - np.mean(eeg_matrix, axis=0)

        self.raw_data = eeg_matrix
        self.n_ch = len(ch_cols)
        self.data_len = len(self.df)

        # Detectar modo del CSV
        if 'filter_mode' in self.df.columns:
            val = self.df['filter_mode'].iloc[0]
            if isinstance(val, str) and len(val) > 2:
                self.mode = val

        print(f"   🔧 Reparaciones: {dups} duplicados, {zeros} ceros.")

    # ==========================================
    # VISUALIZACIÓN
    # ==========================================

    def init_plotting_ui(self):
        if self.plot_container.layout():
            QtWidgets.QWidget().setLayout(self.plot_container.layout())

        layout = QtWidgets.QVBoxLayout(self.plot_container)
        layout.setContentsMargins(0, 0, 0, 0)
        pg.setConfigOptions(antialias=True, background='#111218', foreground='w')

        self.main_plot_widget, self.ui_dict = create_ui(self.n_ch, self.win_sec, self.offset, self.fs)
        layout.addWidget(self.main_plot_widget)

        self.ch_sel = {"idx": 0}
        connect_channel_controls(self.ui_dict, self.n_ch, lambda new_idx: self.ch_sel.update(idx=new_idx))

        # --- FIX: OVERLAY PARENTING ---
        # Emparentamos el Label a self (la ventana principal) para que flote encima
        self.lbl_overlay = QtWidgets.QLabel(self)
        self.lbl_overlay.setStyleSheet(
            "color: #00FF00; font-size: 14px; font-weight: bold; background-color: rgba(0,0,0,180); padding: 8px; border-radius: 4px;")
        self.lbl_overlay.move(40, 40)  # Margen seguro desde el borde
        self.lbl_overlay.raise_()  # Forzar al frente
        self.lbl_overlay.show()

    def update_frame(self):
        if not self.playing or self.current_idx >= self.data_len:
            if self.current_idx >= self.data_len:
                self.playing = False
                self.btn_play.setText("↺ Reiniciar")
                self.timer.stop()
            return

        samples_to_read = int(self.fs * (self.update_ms / 1000.0))
        end_idx = min(self.current_idx + samples_to_read, self.data_len)

        # Validación de seguridad
        if end_idx <= self.current_idx: return

        chunk = self.raw_data[self.current_idx: end_idx]
        if len(chunk) == 0: return

        chunk_T = chunk.T
        for i in range(self.n_ch):
            self.buffers[i].extend(chunk_T[i])

        # Procesar
        try:
            ratio = update_loop(
                self.buffers, self.fs, self.theta_band, self.gamma_band,
                self.eps, self.ui_dict, self.t0, self.ch_sel["idx"],
                self.win_sec, self.offset, self.mode
            )
        except Exception as e:
            # Fallback silencioso si el procesamiento falla en un frame
            ratio = 0.5

        # Actualizar Texto Overlay
        try:
            row = self.df.iloc[end_idx - 1]
            state = row.get('game_state', 'N/A')
            nrg = row.get('energy', 0)
            mission = row.get('mission_state', '')
            r_rec = row.get('ratio', 0)

            txt = f"ESTADO: {state} | ENERGÍA: {nrg}\nMISIÓN: {mission}\n"
            txt += f"RATIO (Grabado): {float(r_rec):.2f} | (Calc): {float(ratio):.2f}"
            self.lbl_overlay.setText(txt)
            self.lbl_overlay.adjustSize()
        except Exception as e:
            # print(f"Error UI Text: {e}") # Debug
            pass

        self.current_idx = end_idx

        self.slider.blockSignals(True)
        self.slider.setValue(self.current_idx)
        self.slider.blockSignals(False)

        cur = self.current_idx / self.fs
        tot = self.data_len / self.fs
        self.lbl_time.setText(f"{int(cur // 60):02d}:{int(cur % 60):02d} / {int(tot // 60):02d}:{int(tot % 60):02d}")

    def toggle_play(self):
        if self.current_idx >= self.data_len:
            self.current_idx = 0
            self.buffers = [deque([0.0] * self.win_samples, maxlen=self.win_samples) for _ in range(self.n_ch)]

        self.playing = not self.playing
        if self.playing:
            self.btn_play.setText("⏸ Pausar")
            self.btn_play.setStyleSheet("background-color: #FF9800; font-weight: bold; padding: 6px;")
            self.timer.start(self.update_ms)
        else:
            self.btn_play.setText("▶ Reproducir")
            self.btn_play.setStyleSheet("background-color: #4CAF50; font-weight: bold; padding: 6px;")
            self.timer.stop()

    def slider_pressed(self):
        self.was_playing = self.playing
        self.playing = False
        self.timer.stop()

    def slider_released(self):
        if self.was_playing:
            self.playing = True
            self.timer.start(self.update_ms)

    def slider_moved(self, val):
        self.current_idx = val
        start = max(0, self.current_idx - self.win_samples)
        chunk = self.raw_data[start: self.current_idx]
        if len(chunk) > 0:
            chunk_T = chunk.T
            for i in range(self.n_ch):
                self.buffers[i].clear()
                if len(chunk) < self.win_samples:
                    self.buffers[i].extend([0.0] * (self.win_samples - len(chunk)))
                self.buffers[i].extend(chunk_T[i])

        self.playing = True
        self.update_frame()
        self.playing = False

    def update_info_label(self):
        dur = self.data_len / self.fs
        self.lbl_info.setText(f"Archivo: {self.n_ch} canales | {self.fs}Hz | {dur:.1f}s | Modo: {self.mode.upper()}")

    # Evento para asegurar que el overlay se mantenga encima al redimensionar
    def resizeEvent(self, event):
        if hasattr(self, 'lbl_overlay'):
            self.lbl_overlay.raise_()
        super().resizeEvent(event)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    player = ReplayPlayer()
    player.show()
    player.resize(1200, 800)
    sys.exit(app.exec_())