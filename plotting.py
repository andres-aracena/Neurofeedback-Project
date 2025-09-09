import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from collections import deque

def create_ui(N_CH, WIN_SEC):
    main = QtWidgets.QWidget()
    main.setWindowTitle("Neurofeedback - Filtros + Wavelet")
    main.setStyleSheet("background-color: #111218; color: #e6e6e6;")
    layout = QtWidgets.QVBoxLayout(main)

    # --- Botones ---
    ctrl_layout = QtWidgets.QHBoxLayout()
    ctrl_layout.addStretch()
    btn_prev = QtWidgets.QToolButton()
    btn_prev.setArrowType(QtCore.Qt.LeftArrow)
    btn_prev.setStyleSheet("QToolButton { color: white; background: transparent; font-size:24px; }")
    lbl_channel = QtWidgets.QLabel("Canal 1")
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

    # --- Área de gráficos ---
    graph_area = pg.GraphicsLayoutWidget()
    layout.addWidget(graph_area)
    main.resize(1500, 900)

    # Ratio
    p_ratio = graph_area.addPlot(row=0, col=0, colspan=2)
    p_ratio.setTitle("Relación Theta/Gamma (mediana entre canales)")
    p_ratio.setLabel('bottom', 'Tiempo', units='s')
    p_ratio.setLabel('left', 'Relación')
    p_ratio.showGrid(x=True, y=True)
    ratio_t, ratio_y = [], []
    curve_ratio = p_ratio.plot([], [], pen=pg.mkPen('c', width=2))

    # Señales crudas
    p_raw = graph_area.addPlot(row=1, col=0)
    p_raw.setTitle("EEG Crudo (8 canales)")
    p_raw.setLabel('bottom', 'Tiempo', units='s')
    p_raw.setLabel('left', 'Amplitud (µV)')
    p_raw.showGrid(x=True, y=True)
    t_axis = np.linspace(-WIN_SEC, 0, WIN_SEC*250)
    curves_raw = [p_raw.plot(t_axis, np.zeros(WIN_SEC*250), pen=pg.mkPen((i*30, 200, 120), width=1)) for i in range(N_CH)]

    # Señal filtrada
    p_filt = graph_area.addPlot(row=2, col=0)
    p_filt.setTitle("Señal filtrada (canal seleccionado)")
    p_filt.setLabel('bottom', 'Tiempo', units='s')
    p_filt.setLabel('left', 'Amplitud (µV)')
    p_filt.showGrid(x=True, y=True)
    curve_theta = p_filt.plot(t_axis, np.zeros(WIN_SEC*250), pen=pg.mkPen('#99FF00', width=2))
    curve_gamma = p_filt.plot(t_axis, np.zeros(WIN_SEC*250), pen=pg.mkPen('r', width=2))

    # Envolventes
    p_env = graph_area.addPlot(row=1, col=1)
    p_env.setTitle("Envolventes por canal (Theta / Gamma)")
    p_env.setLabel('bottom', 'Canal')
    p_env.setLabel('left', 'Amplitud de envolvente (µV)')
    p_env.showGrid(x=True, y=True)
    x_idx = np.arange(N_CH)
    bar_theta = pg.BarGraphItem(x=x_idx-0.2, height=np.zeros(N_CH), width=0.4, brush=(150, 255, 0))
    bar_gamma = pg.BarGraphItem(x=x_idx+0.2, height=np.zeros(N_CH), width=0.4, brush=(200, 50, 50))
    p_env.addItem(bar_theta)
    p_env.addItem(bar_gamma)

    # Wavelet
    p_cwt = graph_area.addPlot(row=2, col=1)
    p_cwt.setTitle("Espectrograma Wavelet (canal seleccionado)")
    p_cwt.setLabel('bottom','Tiempo', units='s')
    p_cwt.setLabel('left','Frecuencia', units='Hz')
    img_cwt = pg.ImageItem()
    p_cwt.addItem(img_cwt)
    freqs = np.linspace(2, 80, 40)
    t_cwt = np.linspace(-WIN_SEC, 0, WIN_SEC*250)
    colormap = pg.colormap.get("viridis")
    lut = colormap.getLookupTable(0.0, 1.0, 256)
    cbar = pg.ColorBarItem(values=(0,1), colorMap=colormap)
    cbar.setImageItem(img_cwt, insert_in=p_cwt)

    ui = {
        "btn_prev": btn_prev, "btn_next": btn_next, "lbl_channel": lbl_channel,
        "p_ratio": p_ratio, "curve_ratio": curve_ratio,
        "ratio_t": deque(maxlen=30*1000//100), "ratio_y": deque(maxlen=30*1000//100),
        "curves_raw": curves_raw, "curve_theta": curve_theta, "curve_gamma": curve_gamma,
        "bar_theta": bar_theta, "bar_gamma": bar_gamma,
        "img_cwt": img_cwt, "cbar": cbar, "freqs": freqs, "lut": lut, "t_cwt": t_cwt
    }

    return main, ui
