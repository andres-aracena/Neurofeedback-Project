import time
import numpy as np
from scipy.signal import butter, lfilter
from pylsl import StreamInlet, resolve_byprop

# === CONFIGURACIÓN ===
fs = 250  # Frecuencia de muestreo de Cyton
buffer = []

# === FUNCIONES DE FILTRADO Y POTENCIA ===
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def compute_band_power(filtered_data):
    return np.mean(np.square(filtered_data))  # Potencia = RMS²

# === BUSCAR Y CONECTAR AL STREAM EEG ===
print("Buscando stream EEG...")
streams = resolve_byprop('type', 'EEG', timeout=5)
if len(streams) == 0:
    print("No se encontró ningún stream EEG.")
    exit()

inlet = StreamInlet(streams[0])
print("Stream EEG conectado correctamente.")

# === BUCLE DE ADQUISICIÓN Y ANÁLISIS EN TIEMPO REAL ===
print("Iniciando adquisición en tiempo real...\n")

try:
    while True:
        sample, _ = inlet.pull_sample()
        buffer.append(sample[0])  # Canal 1 (puedes cambiar el índice)

        if len(buffer) >= fs * 2:  # Ventana de 2 segundos
            window = np.array(buffer[-fs*2:])  # Últimos 2 segundos
            theta = bandpass_filter(window, 4, 8, fs)
            gamma = bandpass_filter(window, 30, 80, fs)
            theta_power = compute_band_power(theta)
            gamma_power = compute_band_power(gamma)

            # Evitar división por cero
            if gamma_power > 0:
                ratio = theta_power / gamma_power
                print(f"Theta/Gamma ratio: {ratio:.2f}")
            else:
                print("Potencia gamma insuficiente para calcular ratio.")

        time.sleep(0.01)  # Evita sobrecargar la CPU

except KeyboardInterrupt:
    print("\nAdquisición finalizada por el usuario.")
