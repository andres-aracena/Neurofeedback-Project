import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, sosfiltfilt
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# === CONFIGURACIÓN DE BRAINFLOW CON SEÑAL ARTIFICIAL ===
BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
board_id = BoardIds.SYNTHETIC_BOARD.value
board = BoardShim(board_id, params)
board.prepare_session()
board.start_stream()
print("Board simulada conectada.")

# === CONFIGURACIÓN GENERAL ===
fs = board.get_sampling_rate(board_id)
eeg_channels = board.get_eeg_channels(board_id)
num_channels = len(eeg_channels)
window_sec = 2
window_samples = fs * window_sec
overlap_sec = 0.25
update_samples = int(fs * overlap_sec)
nyquist = 0.5 * fs
epsilon = 1e-10  # para evitar división por cero

# === FUNCIONES DE FILTRADO ===
def bandpass_filter_sos(data, lowcut, highcut, fs, order=6):
    sos = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band', output='sos')
    return sosfiltfilt(sos, data)

def compute_band_power(data):
    return np.mean(np.square(data))

# === GRAFICACIÓN ===
fig, axs = plt.subplots(3, 1, figsize=(12, 10))
fig.subplots_adjust(hspace=0.5)

# Subplots
ax_ratio = axs[0]
ax_bands = axs[1]
ax_signal = axs[2]

# === Gráfico de Ratio ===
x_vals = []
y_vals = []
line_ratio, = ax_ratio.plot([], [], label="Theta/Gamma Ratio", color='blue')
ax_ratio.set_title("Ratio Theta/Gamma en Tiempo Real")
ax_ratio.set_xlabel("Tiempo (s)")
ax_ratio.set_ylabel("Ratio")
ax_ratio.set_ylim(0, 1)
ax_ratio.grid(True)
ax_ratio.legend()

# === Gráfico de Potencia por Canal ===
bar_theta = ax_bands.bar(np.arange(num_channels) - 0.15, [0]*num_channels, width=0.3, label='Theta', color='green')
bar_gamma = ax_bands.bar(np.arange(num_channels) + 0.15, [0]*num_channels, width=0.3, label='Gamma', color='red')
ax_bands.set_title("Potencia Theta y Gamma por Canal")
ax_bands.set_xlabel("Canal EEG")
ax_bands.set_ylabel("Potencia")
ax_bands.set_xticks(np.arange(num_channels))
ax_bands.set_ylim(0, 50)
ax_bands.legend()

# === Gráfico de señal cruda y filtrada (primer canal) ===
t_signal = np.linspace(0, window_sec, window_samples)
line_raw, = ax_signal.plot(t_signal, np.zeros_like(t_signal), label="Raw", color='gray')
line_theta, = ax_signal.plot(t_signal, np.zeros_like(t_signal), label="Theta", color='green')
line_gamma, = ax_signal.plot(t_signal, np.zeros_like(t_signal), label="Gamma", color='red')
ax_signal.set_title("Señal EEG (cruda y filtrada) - Canal 1")
ax_signal.set_xlabel("Tiempo (s)")
ax_signal.set_ylabel("uV")
ax_signal.set_ylim(-50, 50)
ax_signal.set_xlim(0, window_sec)
ax_signal.legend()
ax_signal.grid(True)

start_time = time.time()

# === FUNCIÓN DE ACTUALIZACIÓN ===
def update(frame):
    global x_vals, y_vals

    # Obtener datos de ventana completa
    data = board.get_current_board_data(window_samples)
    theta_powers = []
    gamma_powers = []

    # Procesar canales EEG
    for i, ch in enumerate(eeg_channels):
        raw = data[ch]
        if len(raw) < window_samples:
            theta_powers.append(0)
            gamma_powers.append(0)
            continue

        theta = bandpass_filter_sos(raw, 4, 8, fs)
        gamma = bandpass_filter_sos(raw, 30, 100, fs)

        theta_power = compute_band_power(theta) / (8 - 4)  # normalizar
        gamma_power = compute_band_power(gamma) / (100 - 30)  # normalizar

        theta_powers.append(theta_power)
        gamma_powers.append(gamma_power)

        # Actualizar señal solo para canal 1
        if i == 7:
            line_raw.set_ydata(raw)
            line_theta.set_ydata(theta)
            line_gamma.set_ydata(gamma)

    # Calcular ratio promedio con protección
    mean_theta = np.mean(theta_powers)
    mean_gamma = np.mean(gamma_powers)
    ratio = mean_theta / (mean_gamma + epsilon)
    ratio = min(ratio, 50)  # Cap para gráfica

    # Actualizar gráfico de ratio
    t_now = time.time() - start_time
    x_vals.append(t_now)
    y_vals.append(ratio)
    if len(x_vals) > 100:
        x_vals = x_vals[-100:]
        y_vals = y_vals[-100:]

    line_ratio.set_data(x_vals, y_vals)
    ax_ratio.set_xlim(max(0, t_now - 30), t_now)

    # Actualizar gráfico de barras
    for i, bar in enumerate(bar_theta):
        bar.set_height(theta_powers[i])
    for i, bar in enumerate(bar_gamma):
        bar.set_height(gamma_powers[i])

    return line_ratio, bar_theta, bar_gamma, line_raw, line_theta, line_gamma

ani = FuncAnimation(fig, update, interval=int(overlap_sec * 1000))  # ~250ms
plt.show()

# === Limpiar sesión al cerrar ===
board.stop_stream()
board.release_session()
