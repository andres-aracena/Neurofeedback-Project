import numpy as np
import time
from scipy.signal import butter, lfilter
from pylsl import StreamInlet, resolve_byprop
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# === Configuración ===
fs = 250  # Hz
window_size_sec = 2
window_length = fs * window_size_sec
num_channels = 8  # Cyton tiene 8 canales

# === Filtro bandpass ===
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def compute_band_power(filtered_data):
    return np.mean(np.square(filtered_data))

# === Conexión LSL ===
print("Buscando stream EEG...")
streams = resolve_byprop('type', 'EEG', timeout=5)
if len(streams) == 0:
    print("No se encontró ningún stream EEG.")
    exit()

inlet = StreamInlet(streams[0])
print("Conectado a stream EEG.")

# === Buffers por canal ===
channel_buffers = [[] for _ in range(num_channels)]
ratio_history = []

# === Setup del gráfico ===
fig, (ax_ratio, ax_bands) = plt.subplots(2, 1, figsize=(10, 8))

# Gráfico 1: Ratio theta/gamma en el tiempo
x_vals = []
y_vals = []

line_ratio, = ax_ratio.plot([], [], label="Theta/Gamma Ratio", color='blue')
ax_ratio.set_title("Ratio Theta/Gamma en Tiempo Real")
ax_ratio.set_xlabel("Tiempo (s)")
ax_ratio.set_ylabel("Ratio")
ax_ratio.set_ylim(0, 1)
ax_ratio.legend()
ax_ratio.grid(True)

# Gráfico 2: Potencias theta y gamma por canal (barra)
bar_theta = ax_bands.bar(np.arange(num_channels) - 0.15, [0]*num_channels, width=0.3, label='Theta (4-8Hz)', color='green')
bar_gamma = ax_bands.bar(np.arange(num_channels) + 0.15, [0]*num_channels, width=0.3, label='Gamma (30-80Hz)', color='red')
ax_bands.set_title("Potencia Theta y Gamma por Canal")
ax_bands.set_xlabel("Canal EEG")
ax_bands.set_ylabel("Potencia")
ax_bands.set_xticks(range(num_channels))
ax_bands.legend()
ax_bands.set_ylim(0, 100)

# === Función de actualización gráfica ===
start_time = time.time()

def update(frame):
    global x_vals, y_vals

    # Recolectar nuevas muestras
    for _ in range(fs):  # 1 segundo de datos
        sample, _ = inlet.pull_sample()
        for ch in range(num_channels):
            channel_buffers[ch].append(sample[ch])
            if len(channel_buffers[ch]) > window_length:
                channel_buffers[ch] = channel_buffers[ch][-window_length:]

    # Calcular potencia theta y gamma por canal
    theta_powers = []
    gamma_powers = []

    for ch in range(num_channels):
        data = np.array(channel_buffers[ch])
        if len(data) < window_length:
            theta_powers.append(0)
            gamma_powers.append(0)
            continue

        theta = bandpass_filter(data, 4, 8, fs)
        gamma = bandpass_filter(data, 30, 80, fs)

        theta_power = compute_band_power(theta)
        gamma_power = compute_band_power(gamma)

        theta_powers.append(theta_power)
        gamma_powers.append(gamma_power)

    # Calcular ratio global como promedio de todos los canales
    total_theta = np.mean(theta_powers)
    total_gamma = np.mean(gamma_powers)
    ratio = total_theta / total_gamma if total_gamma > 0 else 0

    # Actualizar datos del gráfico 1
    current_time = time.time() - start_time
    x_vals.append(current_time)
    y_vals.append(ratio)
    if len(x_vals) > 30:
        x_vals = x_vals[-30:]
        y_vals = y_vals[-30:]
    line_ratio.set_data(x_vals, y_vals)
    ax_ratio.set_xlim(max(0, current_time - 30), current_time)

    # Actualizar gráfico de barras
    for i, bar in enumerate(bar_theta):
        bar.set_height(theta_powers[i])
    for i, bar in enumerate(bar_gamma):
        bar.set_height(gamma_powers[i])

    return line_ratio, bar_theta, bar_gamma

# === Iniciar visualización ===
ani = FuncAnimation(fig, update, interval=1000)
plt.tight_layout()
plt.show()
