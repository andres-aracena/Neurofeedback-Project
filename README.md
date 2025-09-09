# Neurofeedback Project

Este proyecto implementa un sistema de **neurofeedback con EEG** usando:
- OpenBCI / BrainFlow
- Filtros de señal (Butterworth, Notch, High-pass, Savitzky-Golay)
- Análisis en tiempo real de bandas **Theta** y **Gamma**
- Visualización con **PyQtGraph**
- Espectrograma Wavelet

---

## 📂 Estructura

- main.py                 # Punto de entrada principal
- filters.py              # Filtros de preprocesamiento y bandas
- plotting.py             # Configuración de PyQtGraph y funciones gráficas
- processing.py           # Lógica de análisis (envolventes, ratio, wavelet)
- board_manager.py        # Conexión y gestión de BrainFlow
- requirements.txt        # Dependencias del proyecto
- README.md               # Guía del proyecto

---

## 🚀 Uso

1. Clonar el repositorio
   ```bash
   git clone https://github.com/andres-aracena/neurofeedback-project.git
   cd neurofeedback-project

2. Instalar dependencias

   ```bash
   pip install -r requirements.txt

3. Ejecutar

   ```bash
   python main.py

---

## ⚙️ Configuración

Por defecto usa BoardIds.SYNTHETIC_BOARD (simulación).
Para usar Cyton cambia en main.py:

   ```bash
   board = init_board(BoardIds.CYTON_BOARD.value)
   ```
---

## ✨ Funcionalidades

- Señales crudas en tiempo real (8 canales).
- Señales filtradas (Theta/Gamma).
- Envolventes por canal.
- Relación Theta/Gamma (mediana).
- Espectrograma Wavelet (mapa de calor).
- Selector de canal con botones de flechas.