import sys, time, os, csv
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from collections import deque
import numpy as np
import pandas as pd

# Importamos TUS módulos locales para asegurar identidad visual y matemática
try:
    from processing import update_loop
    from plotting import create_ui, connect_channel_controls
except ImportError as e:
    print(f"❌ Error crítico: No se encuentran los módulos del sistema (processing.py, plotting.py).")
    print(f"   Detalle: {e}")
    print("   Asegúrate de ejecutar este script en la misma carpeta que tu main.py")
    sys.exit(1)


class ReplayPlayer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Configuración por defecto (se sobrescribirá al cargar el archivo)
        self.fs = 250
        self.n_ch = 8
        self.win_sec = 10  # Ventana visual igual que en tiempo real
        self.update_ms = 80  # Velocidad de actualización igual que en main.py

        # Bandas de frecuencia (Idénticas a main.py)
        self.theta_band = (4.0, 8.0)
        self.gamma_band = (30.0, 90.0)
        self.eps = 1e-12
        self.offset = 250
        self.mode = 'wavelet'  # Por defecto, se leerá del CSV si existe

        # Estado de reproducción
        self.playing = False
        self.current_idx = 0
        self.data_len = 0
        self.df = None
        self.raw_data = None

        # Interfaz
        self.setup_ui_structure()

        # Cargar archivo al inicio
        self.load_file()

    def setup_ui_structure(self):
        self.setWindowTitle("Neurofeedback Replay System")
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # 1. Área de Gráficos (Placeholder, se crea al cargar datos)
        self.plot_container = QtWidgets.QWidget()
        self.layout.addWidget(self.plot_container, stretch=1)

        # 2. Panel de Control de Reproducción
        control_panel = QtWidgets.QWidget()
        control_panel.setStyleSheet("background-color: #2b2b2b; color: white;")
        cp_layout = QtWidgets.QHBoxLayout(control_panel)

        self.btn_play = QtWidgets.QPushButton("▶ Reproducir")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setStyleSheet("background-color: #4CAF50; font-weight: bold; padding: 5px;")

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.sliderPressed.connect(self.slider_pressed)
        self.slider.sliderReleased.connect(self.slider_released)
        self.slider.valueChanged.connect(self.slider_moved)

        self.lbl_time = QtWidgets.QLabel("00:00 / 00:00")
        self.lbl_info = QtWidgets.QLabel("Esperando archivo...")
        self.lbl_info.setStyleSheet("color: #aaaaaa; font-size: 10px;")

        cp_layout.addWidget(self.btn_play)
        cp_layout.addWidget(self.lbl_time)
        cp_layout.addWidget(self.slider)

        self.layout.addWidget(control_panel)
        self.layout.addWidget(self.lbl_info)

        # Timer de actualización (simula el loop de main.py)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

    def load_file(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Seleccionar Grabación CSV", "", "CSV Files (*.csv)"
        )

        if not filename:
            sys.exit(0)

        print(f"📂 Cargando: {os.path.basename(filename)}...")

        try:
            # Carga rápida con Pandas
            self.df = pd.read_csv(filename)

            # Detectar columnas de canales
            ch_cols = [c for c in self.df.columns if c.startswith('channel_')]
            self.n_ch = len(ch_cols)

            if self.n_ch == 0:
                raise ValueError("No se encontraron columnas 'channel_X'")

            # Extraer matriz de datos EEG (Muestras x Canales)
            self.raw_data = self.df[ch_cols].values
            self.data_len = len(self.df)

            # Estimar Frecuencia de Muestreo (FS)
            timestamps = self.df['timestamp'].values
            if len(timestamps) > 1:
                diffs = np.diff(timestamps)
                # Mediana para ignorar saltos
                median_diff = np.median(diffs[diffs > 0])
                self.fs = int(round(1.0 / median_diff))
            else:
                self.fs = 250  # Default

            # Detectar modo de filtro usado
            if 'filter_mode' in self.df.columns:
                self.mode = self.df['filter_mode'].iloc[0]

            print(f"✅ Configuración detectada: {self.n_ch} canales, {self.fs} Hz, Modo: {self.mode}")

            # Inicializar Buffers (Deque, igual que en main.py)
            self.win_samples = self.win_sec * self.fs
            self.buffers = [deque([0.0] * self.win_samples, maxlen=self.win_samples) for _ in range(self.n_ch)]

            # Inicializar UI de Plotting.py
            self.init_plotting_ui()

            # Configurar Slider
            self.slider.setRange(0, self.data_len)
            self.update_info_label()

            # Tiempo de inicio para gráficas relativas
            self.t0 = time.time()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error leyendo archivo:\n{e}")
            sys.exit(1)

    def init_plotting_ui(self):
        # Limpiar contenedor anterior si existe
        if self.plot_container.layout():
            QtWidgets.QWidget().setLayout(self.plot_container.layout())

        # Usar create_ui de TU plotting.py para que se vea IDÉNTICO
        # Necesitamos un layout vertical para meter el widget que devuelve create_ui
        layout = QtWidgets.QVBoxLayout(self.plot_container)
        layout.setContentsMargins(0, 0, 0, 0)

        pg.setConfigOptions(antialias=True, background='#111218', foreground='w')

        # Llamada a tu función original
        self.main_plot_widget, self.ui_dict = create_ui(self.n_ch, self.win_sec, self.offset, self.fs)

        layout.addWidget(self.main_plot_widget)

        # Conectar controles de canales
        self.ch_sel = {"idx": 0}
        connect_channel_controls(self.ui_dict, self.n_ch, lambda new_idx: self.ch_sel.update(idx=new_idx))

        # Label extra para estado del juego (Overlay)
        self.lbl_overlay = QtWidgets.QLabel(self.main_plot_widget)
        self.lbl_overlay.setStyleSheet(
            "color: lime; font-size: 14px; font-weight: bold; background-color: rgba(0,0,0,150); padding: 5px;")
        self.lbl_overlay.move(20, 20)  # Esquina superior izquierda
        self.lbl_overlay.show()

    def update_frame(self):
        if not self.playing or self.current_idx >= self.data_len:
            if self.current_idx >= self.data_len:
                self.playing = False
                self.btn_play.setText("↺ Reiniciar")
                self.timer.stop()
            return

        # Calcular cuántas muestras procesar en este frame
        # En main.py se procesan bloques cada 80ms.
        # samples_per_frame = FS * (UPDATE_MS / 1000)
        samples_to_read = int(self.fs * (self.update_ms / 1000.0))

        # Asegurar que no nos pasamos del final
        end_idx = min(self.current_idx + samples_to_read, self.data_len)

        # Extraer chunk de datos
        chunk = self.raw_data[self.current_idx: end_idx]

        # Si no hay datos, salir
        if len(chunk) == 0: return

        # 1. Alimentar Buffers (Igual que main.py update)
        # Transponer chunk para iterar por canales
        chunk_T = chunk.T
        for i in range(self.n_ch):
            self.buffers[i].extend(chunk_T[i])

        # 2. Llamar a TU update_loop original (Processing.py)
        # Esto garantiza que el cálculo matemático y el dibujado sean idénticos
        ratio = update_loop(
            self.buffers,
            self.fs,
            self.theta_band,
            self.gamma_band,
            self.eps,
            self.ui_dict,
            self.t0,
            self.ch_sel["idx"],
            self.win_sec,
            self.offset,
            self.mode
        )

        # 3. Actualizar UI con Metadatos Grabados
        # Usamos la última fila del chunk para mostrar el estado actual
        current_row = self.df.iloc[end_idx - 1]

        # Recuperar info guardada
        game_state = current_row.get('game_state', 'N/A')
        energy = current_row.get('energy', 0)
        mission = current_row.get('mission_state', '')
        ratio_rec = current_row.get('ratio', 0)

        status_text = (
            f"Juego: {game_state} | Energía: {energy} | Misión: {mission}\n"
            f"Ratio Grabado: {ratio_rec:.2f} | Ratio Recalculado: {ratio:.2f}"
        )
        self.lbl_overlay.setText(status_text)
        self.lbl_overlay.adjustSize()

        # Avanzar puntero
        self.current_idx = end_idx

        # Actualizar Slider y Tiempo (sin bloquear)
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_idx)
        self.slider.blockSignals(False)

        cur_sec = self.current_idx / self.fs
        tot_sec = self.data_len / self.fs
        self.lbl_time.setText(
            f"{int(cur_sec // 60):02d}:{int(cur_sec % 60):02d} / {int(tot_sec // 60):02d}:{int(tot_sec % 60):02d}")

    # --- Controles de Reproducción ---
    def toggle_play(self):
        if self.current_idx >= self.data_len:
            self.current_idx = 0  # Reiniciar si llegó al final
            # Limpiar buffers
            self.buffers = [deque([0.0] * self.win_samples, maxlen=self.win_samples) for _ in range(self.n_ch)]

        self.playing = not self.playing
        if self.playing:
            self.btn_play.setText("⏸ Pausar")
            self.btn_play.setStyleSheet("background-color: #FF9800; font-weight: bold; padding: 5px;")
            self.timer.start(self.update_ms)
        else:
            self.btn_play.setText("▶ Reproducir")
            self.btn_play.setStyleSheet("background-color: #4CAF50; font-weight: bold; padding: 5px;")
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
        # Cuando saltamos en el tiempo, necesitamos rellenar los buffers
        # con los datos anteriores a ese punto para que los filtros no exploten
        start_fill = max(0, self.current_idx - self.win_samples)
        fill_chunk = self.raw_data[start_fill: self.current_idx]

        # Rellenar buffers limpiamente
        if len(fill_chunk) > 0:
            fill_T = fill_chunk.T
            for i in range(self.n_ch):
                # Limpiar y llenar
                self.buffers[i].clear()
                # Si falta data al inicio, rellenar con ceros
                if len(fill_chunk) < self.win_samples:
                    self.buffers[i].extend([0.0] * (self.win_samples - len(fill_chunk)))
                self.buffers[i].extend(fill_T[i])

        # Actualizar un frame estático para ver dónde estamos
        self.playing = True  # Hack temporal para que update_frame dibuje
        self.update_frame()
        self.playing = False

    def update_info_label(self):
        duration = self.data_len / self.fs
        self.lbl_info.setText(
            f"Archivo: {self.n_ch} canales | {self.fs}Hz | "
            f"Duración: {duration:.1f}s | Modo: {self.mode.upper()}"
        )


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    player = ReplayPlayer()
    player.show()
    player.resize(1200, 800)
    sys.exit(app.exec_())