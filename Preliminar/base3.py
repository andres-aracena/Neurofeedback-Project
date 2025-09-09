import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, sosfiltfilt, hilbert
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# === Configuración del board simulado ===
BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
board.prepare_session()
board.start_stream()
print("Board simulada conectada.")

# === Parámetros generales ===
fs = board.get_sampling_rate(BoardIds.SYNTHETIC_BOARD.value)
channels = board.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
n_channels = len(channels)
win_sec = 2
win_samples = fs * win_sec
interval_ms = 250
nyquist = fs / 2
epsilon = 1e-10

# === Funciones de señal ===
def bandpass(data, low, high, order=6):
    sos = butter(order, [low / nyquist, high / nyquist], btype='band', output='sos')
    return sosfiltfilt(sos, data)

def compute_envelope_ratio(theta, gamma, epsilon=1e-10):
    env_theta = np.abs(hilbert(theta))
    env_gamma = np.abs(hilbert(gamma))
    ratio_envelope = env_theta / (env_gamma + epsilon)
    return np.mean(ratio_envelope)

# === Configurar figura ===
fig, axs = plt.subplots(3, 1, figsize=(12, 10))
fig.subplots_adjust(hspace=0.5)

# === Ratio plot ===
x_vals, y_vals = [], []
line_ratio, = axs[0].plot([], [], label="Theta/Gamma", color='blue')
axs[0].set_title("Ratio Theta/Gamma")
axs[0].set_ylabel("Ratio")
axs[0].set_ylim(0, 50)
axs[0].legend()
axs[0].grid(True)

# === Barras de potencia por canal (envolventes) ===
bar_theta = axs[1].bar(np.arange(n_channels) - 0.15, [0]*n_channels, 0.3, label='Theta (Env)', color='green')
bar_gamma = axs[1].bar(np.arange(n_channels) + 0.15, [0]*n_channels, 0.3, label='Gamma (Env)', color='red')
axs[1].set_title("Potencia Instantánea (Envelope)")
axs[1].set_xlabel("Canal EEG")
axs[1].set_ylabel("Potencia")
axs[1].set_ylim(0, 50)
axs[1].set_xticks(np.arange(n_channels))
axs[1].legend()

# === Señal cruda y filtrada (solo canal 0) ===
t_signal = np.linspace(0, win_sec, win_samples)
line_raw, = axs[2].plot(t_signal, np.zeros(win_samples), label="Raw", color='gray')
line_theta, = axs[2].plot(t_signal, np.zeros(win_samples), label="Theta", color='green')
line_gamma, = axs[2].plot(t_signal, np.zeros(win_samples), label="Gamma", color='red')
axs[2].set_title("Señal EEG - Canal 1")
axs[2].set_xlabel("Tiempo (s)")
axs[2].set_ylabel("uV")
axs[2].set_xlim(0, win_sec)
axs[2].set_ylim(-100, 100)
axs[2].legend()
axs[2].grid(True)

start_time = time.time()

# === Loop de actualización ===
def update(frame):
    global x_vals, y_vals

    data = board.get_current_board_data(win_samples)
    ratios = []
    theta_envs = []
    gamma_envs = []

    for i, ch in enumerate(channels):
        raw = data[ch]
        if len(raw) < win_samples:
            continue

        theta = bandpass(raw, 4, 8, order=6)
        gamma = bandpass(raw, 30, 80, order=10)
        gamma *= 0.5  # corregir sobreestimación de gamma

        # Envolventes
        env_theta = np.abs(hilbert(theta))
        env_gamma = np.abs(hilbert(gamma))

        # Ratio por canal
        ratio = compute_envelope_ratio(theta, gamma, epsilon)
        ratios.append(ratio)

        theta_envs.append(np.mean(env_theta))
        gamma_envs.append(np.mean(env_gamma))

        # Solo para canal 0: mostrar señales
        if i == 0:
            line_raw.set_ydata(raw)
            line_theta.set_ydata(theta)
            line_gamma.set_ydata(gamma)

    # Promedio del ratio de todos los canales
    ratio_global = np.mean(ratios) if ratios else 0
    ratio_global = min(ratio_global, 50)

    # Actualizar gráfico de ratio
    t_now = time.time() - start_time
    x_vals.append(t_now)
    y_vals.append(ratio_global)
    x_vals = x_vals[-200:]
    y_vals = y_vals[-200:]
    line_ratio.set_data(x_vals, y_vals)
    axs[0].set_xlim(max(0, t_now - 30), t_now)

    # Actualizar barras de potencia (envelope)
    for i, bar in enumerate(bar_theta):
        bar.set_height(theta_envs[i] if i < len(theta_envs) else 0)
    for i, bar in enumerate(bar_gamma):
        bar.set_height(gamma_envs[i] if i < len(gamma_envs) else 0)

    return line_ratio, bar_theta, bar_gamma, line_raw, line_theta, line_gamma

ani = FuncAnimation(fig, update, interval=interval_ms)
plt.show()

# === Limpieza al cerrar ===
board.stop_stream()
board.release_session()
