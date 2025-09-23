# plotting.py
import json
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from collections import deque

# -------------------------
# Ventana de configuración inicial
# -------------------------
class ConfigDialog(QtWidgets.QDialog):
    def __init__(self, fs_values, n_ch_values, win_sec_values, mode_values,
                 default_fs=250, default_n_ch=8, default_win=10, default_mode="wavelet"):
        super().__init__()
        self.setWindowTitle("Configuración de sesión")
        self.setStyleSheet("background-color: #111218; color: white; font-size:24px;")
        layout = QtWidgets.QFormLayout(self)

        # ComboBoxes
        self.fs_cb = QtWidgets.QComboBox(); [self.fs_cb.addItem(str(v)) for v in fs_values]
        self.fs_cb.setCurrentText(str(default_fs))

        self.nch_cb = QtWidgets.QComboBox(); [self.nch_cb.addItem(str(v)) for v in n_ch_values]
        self.nch_cb.setCurrentText(str(default_n_ch))

        self.win_cb = QtWidgets.QComboBox(); [self.win_cb.addItem(str(v)) for v in win_sec_values]
        self.win_cb.setCurrentText(str(default_win))

        self.mode_cb = QtWidgets.QComboBox(); [self.mode_cb.addItem(v) for v in mode_values]
        self.mode_cb.setCurrentText(default_mode)

        layout.addRow("Frecuencia de muestreo (Hz):", self.fs_cb)
        layout.addRow("Número de canales:", self.nch_cb)
        layout.addRow("Ventana (s):", self.win_cb)
        layout.addRow("Modo de procesamiento:", self.mode_cb)

        # Botones
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def get_config(self):
        return {
            "FS": int(self.fs_cb.currentText()),
            "N_CH": int(self.nch_cb.currentText()),
            "WIN_SEC": int(self.win_cb.currentText()),
            "MODE": self.mode_cb.currentText()
        }

# -------------------------
# Helper: botón de ayuda en cada plot
# -------------------------
def add_help_button(plot_item, description):
    """
    Inserta un botón '?' dentro del PlotItem usando QGraphicsProxyWidget.
    El botón se reposiciona cuando cambia el tamaño del ViewBox/scene.
    """
    # botón Qt (sin parent)
    btn = QtWidgets.QToolButton()
    btn.setText("?")
    btn.setStyleSheet("""
        QToolButton {
            background-color: #333;
            color: white;
            font-size: 12px;
            border-radius: 8px;
            padding: 2px;
        }
        QToolButton::hover {
            background-color: #555;
        }
    """)
    btn.setFixedSize(18, 18)

    # proxy para insertar widget Qt en la escena de pyqtgraph
    proxy = QtWidgets.QGraphicsProxyWidget()
    proxy.setWidget(btn)

    # añadir proxy a la escena del plot_item
    scene = plot_item.scene()
    if scene is None:
        # si por alguna razón no está aún en la escena, intentamos añadir más tarde
        raise RuntimeError("PlotItem no está aún en la escena gráfica.")
    scene.addItem(proxy)

    # función para posicionar el proxy en la esquina superior derecha del ViewBox
    vb = plot_item.getViewBox()

    def update_position(*args):
        try:
            rect = vb.sceneBoundingRect()  # rect en coordenadas de la escena
            # ajustar para que el botón quede dentro del rect (debe restar el ancho del proxy)
            w = proxy.boundingRect().width() if proxy.boundingRect().width() > 0 else 18
            x = rect.right() - w - 6
            y = rect.top() + 6
            proxy.setPos(x, y)
        except Exception:
            pass  # fallbacks silenciosos si hay problemas momentáneos

    # conectar señales que normalmente se emiten en cambio de tamaño / layout
    try:
        vb.sigResized.connect(update_position)   # ViewBox resize (pyqtgraph)
    except Exception:
        # señal puede no existir según versión; no crítico
        pass

    try:
        scene.sigSceneRectChanged.connect(update_position)
    except Exception:
        # fallback: si no existe, no hacemos nada
        pass

    # también actualizamos inmediatamente
    update_position()

    # acción del botón: mostrar cuadro de información
    def show_info():
        QtWidgets.QMessageBox.information(None, "Información del gráfico", description)

    btn.clicked.connect(show_info)

    # regresar proxy por si queremos manipularlo (no necesario)
    return proxy

# -------------------------
# create_ui
# -------------------------
def create_ui(N_CH, WIN_SEC, OFFSET, FS):
    """
    Crea la interfaz y devuelve (main_widget, ui_dict).

    - N_CH: número de canales a mostrar
    - WIN_SEC: segundos visibles (display). Nota: el procesamiento puede usar 6s internamente.
    - OFFSET: separación vertical entre canales (en µV visual)
    - FS: sampling rate (por defecto 250)
    """

    # cargar descripciones (info.json debe estar en el mismo directorio)
    try:
        with open("info.json", "r", encoding="utf-8") as f:
            INFO = json.load(f)
    except Exception:
        # texto por defecto si no existe info.json
        INFO = {
            "ratio": "Relación Theta/Gamma: mediana entre canales.",
            "raw": "Señal EEG cruda. Cada línea corresponde a un canal desplazado verticalmente.",
            "filt": "Señal filtrada: theta y gamma del canal seleccionado.",
            "env": "Envolventes: amplitud media por canal en las bandas theta/gamma.",
            "wavelet": "Espectrograma Wavelet (CWT): energía por frecuencia en el tiempo."
        }

    # ventana principal
    main = QtWidgets.QWidget()
    main.setWindowTitle("Neurofeedback - Filtros + Wavelet")
    main.setStyleSheet("background-color: #111218; color: #e6e6e6;")
    vlayout = QtWidgets.QVBoxLayout(main)

    # controles (botones prev/next y label canal)
    ctrl_layout = QtWidgets.QHBoxLayout()
    ctrl_layout.addStretch()
    btn_prev = QtWidgets.QToolButton()
    btn_prev.setArrowType(QtCore.Qt.LeftArrow)
    btn_prev.setStyleSheet("QToolButton { color: white; background: transparent; font-size:20px; }")
    lbl_channel = QtWidgets.QLabel("Canal 1")
    lbl_channel.setAlignment(QtCore.Qt.AlignCenter)
    lbl_channel.setStyleSheet("font-size:18px; color:white; padding:6px;")
    btn_next = QtWidgets.QToolButton()
    btn_next.setArrowType(QtCore.Qt.RightArrow)
    btn_next.setStyleSheet("QToolButton { color: white; background: transparent; font-size:20px; }")
    ctrl_layout.addWidget(btn_prev)
    ctrl_layout.addWidget(lbl_channel)
    ctrl_layout.addWidget(btn_next)
    ctrl_layout.addStretch()
    vlayout.addLayout(ctrl_layout)

    # graphics layout
    graph_area = pg.GraphicsLayoutWidget()
    vlayout.addWidget(graph_area)
    main.resize(2000, 1500)

    # procesamiento vs display: (processing debe usar 6s; aquí mostramos WIN_SEC s)
    disp_sec = WIN_SEC
    n_disp = int(disp_sec * FS)
    t_axis = np.linspace(-disp_sec, 0, n_disp)

    # -------------------------
    # Gráfico Ratio Theta/Gamma
    # =========================
    p_ratio = graph_area.addPlot(row=0, col=0, colspan=2)
    p_ratio.setTitle("Relación Theta/Gamma (Mediana 8 canales)")
    p_ratio.setLabel('bottom', 'Tiempo', units='s')
    p_ratio.setLabel('left', 'Relación')
    p_ratio.setYRange(0, 1)
    p_ratio.showGrid(x=True, y=True)
    ratio_t, ratio_y = [], []
    curve_ratio = p_ratio.plot([], [], pen=pg.mkPen('c', width=2))
    add_help_button(p_ratio, INFO["ratio"])

    # =========================
    # Gráfico Señales crudas
    # =========================
    p_raw = graph_area.addPlot(row=1, col=0)
    p_raw.setTitle("EEG Crudo (8 canales)")
    p_raw.setLabel('bottom', 'Tiempo', units='s')
    yticks = [(i * OFFSET, f"Canal {i + 1}") for i in range(N_CH)]
    p_raw.getAxis('left').setTicks([yticks])
    p_raw.showGrid(x=True, y=True)
    t_axis = np.linspace(-WIN_SEC, 0, WIN_SEC*250)
    curves_raw = [
        p_raw.plot(t_axis, np.zeros(WIN_SEC*250), pen=pg.mkPen((i*30, 200, 120), width=1))
        for i in range(N_CH)
    ]
    add_help_button(p_raw, INFO["raw"])

    # =========================
    # Gráfico Señal filtrada
    # =========================
    p_filt = graph_area.addPlot(row=2, col=0)
    p_filt.setTitle("Señal filtrada (Canal 1)")
    p_filt.setLabel('bottom', 'Tiempo', units='s')
    p_filt.setLabel('left', 'Amplitud (µV)')
    p_filt.showGrid(x=True, y=True)
    curve_theta = p_filt.plot(t_axis, np.zeros(WIN_SEC*250), pen=pg.mkPen('#99FF00', width=2), name="Theta")
    curve_gamma = p_filt.plot(t_axis, np.zeros(WIN_SEC*250), pen=pg.mkPen('r', width=2), name="Gamma")
    p_filt.addLegend()
    add_help_button(p_filt, INFO["filt"])

    # =========================
    # Gráfico Envolventes
    # =========================
    p_env = graph_area.addPlot(row=1, col=1)
    p_env.setTitle("Envolventes por canal (Theta / Gamma)")
    p_env.setLabel('bottom', 'Canal')
    p_env.setLabel('left', 'Amplitud (µV)')
    p_env.showGrid(x=True, y=True)
    x_idx = (np.arange(N_CH))+1
    bar_theta = pg.BarGraphItem(x=x_idx-0.2, height=np.zeros(N_CH), width=0.4, brush=(150, 255, 0))
    bar_gamma = pg.BarGraphItem(x=x_idx+0.2, height=np.zeros(N_CH), width=0.4, brush=(200, 50, 50))
    p_env.addItem(bar_theta)
    p_env.addItem(bar_gamma)
    add_help_button(p_env, INFO["env"])

    # =========================
    # Gráfico Wavelet
    # =========================
    p_cwt = graph_area.addPlot(row=2, col=1)
    p_cwt.setTitle("Espectrograma Wavelet (Canal 1)")
    p_cwt.setLabel('bottom','Tiempo', units='s')
    p_cwt.setLabel('left','Frecuencia', units='Hz')
    img_cwt = pg.ImageItem()
    p_cwt.addItem(img_cwt)
    p_cwt.setYRange(0, 100)
    freqs = np.linspace(1, 100, 40)
    t_cwt = np.linspace(-WIN_SEC, 0, WIN_SEC*250)
    colormap = pg.colormap.get("viridis")
    lut = colormap.getLookupTable(0.0, 1.0, 256)
    cbar = pg.ColorBarItem(values=(0,1), colorMap=colormap)
    cbar.setImageItem(img_cwt, insert_in=p_cwt)
    try:
        add_help_button(p_cwt, INFO.get("wavelet", "Espectrograma Wavelet"))
    except Exception:
        pass

    # -------------------------
    # Diccionario UI (retorno)
    # -------------------------
    ui = {
        "btn_prev": btn_prev, "btn_next": btn_next, "lbl_channel": lbl_channel,
        "p_ratio": p_ratio, "curve_ratio": curve_ratio,
        "ratio_t": deque(maxlen=30 * 1000 // 100), "ratio_y": deque(maxlen=30 * 1000 // 100),
        "curves_raw": curves_raw, "curve_theta": curve_theta, "curve_gamma": curve_gamma,
        "bar_theta": bar_theta, "bar_gamma": bar_gamma,
        "p_raw": p_raw, "p_filt": p_filt, "p_cwt": p_cwt, "p_env": p_env,
        "img_cwt": img_cwt, "cbar": cbar, "freqs": freqs, "lut": lut, "t_cwt": t_cwt,
        "FS": FS, "OFFSET": OFFSET, "disp_sec": disp_sec
    }

    return main, ui

# -------------------------
# Helpers extra
# -------------------------
def connect_channel_controls(ui, N_CH, on_change):
    """
    Conecta botones prev/next a una función de cambio de canal.
    on_change recibe el nuevo índice.
    """
    ch_sel = {"idx": 0}  # mutable

    def set_channel(idx):
        ch_sel["idx"] = idx % N_CH
        ui['lbl_channel'].setText(f"Canal {ch_sel['idx']+1}")
        on_change(ch_sel["idx"])

    ui['btn_prev'].clicked.connect(lambda: set_channel(ch_sel["idx"]-1))
    ui['btn_next'].clicked.connect(lambda: set_channel(ch_sel["idx"]+1))

    return ch_sel
