# main.py
import sys, time, os, socket, json, threading
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from collections import deque
import numpy as np
from datetime import datetime
import csv

# Importamos tus módulos
from board_manager import init_board, get_eeg_channels
from processing import update_loop
from plotting import create_ui, connect_channel_controls, ConfigDialog

# =========================
# Configuración de comunicación UDP con Godot
# =========================
GODOT_UDP_PORT = 9080
GODOT_UDP_IP = "127.0.0.1"
PYTHON_UDP_PORT = 9081

class GodotUDPCommunicator:
    def __init__(self):
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind(('127.0.0.1', PYTHON_UDP_PORT))
        self.udp_socket.settimeout(0.001)

        self.connected = False
        self.latest_game_state = "disconnected"
        self.latest_energy = 0
        self.latest_mission_state = "unknown"
        self.latest_minigame_type = "none"
        self.latest_minigame_state = "none"

        self.corsi_state = "INTRO"
        self.nback_state = "INTRO"

        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()

        print(f"Comunicador UDP iniciado en puerto {PYTHON_UDP_PORT}")

    def _receive_loop(self):
        while True:
            try:
                data, addr = self.udp_socket.recvfrom(1024)
                message = data.decode('utf-8')
                self._handle_godot_message(message)
            except socket.timeout:
                continue
            except Exception as e:
                time.sleep(0.01)

    def _handle_godot_message(self, message):
        try:
            data = json.loads(message)
            message_type = data.get("type", "")

            if message_type == "game_state":
                self.latest_game_state = data.get("state", "unknown")
                self.latest_energy = data.get("energy", 0)
                self.latest_mission_state = data.get("mission_state", "unknown")

            elif message_type == "minigame_state":
                minigame_type = data.get("minigame_type", "")
                state = data.get("state", "")

                if minigame_type == "corsi":
                    self.corsi_state = state
                    self.latest_minigame_type = "corsi"
                    self.latest_minigame_state = state

                elif minigame_type == "nback":
                    self.nback_state = state
                    self.latest_minigame_type = "nback"
                    self.latest_minigame_state = state

        except Exception as e:
            pass

    def send_ratio_to_godot(self, ratio):
        try:
            message = {
                "type": "neurofeedback",
                "ratio": float(ratio),
                "timestamp": time.time()
            }
            data = json.dumps(message).encode('utf-8')
            self.udp_socket.sendto(data, (GODOT_UDP_IP, GODOT_UDP_PORT))
        except Exception as e:
            pass

    def get_current_state(self):
        return {
            "game_state": self.latest_game_state,
            "energy": self.latest_energy,
            "mission_state": self.latest_mission_state,
            "minigame_type": self.latest_minigame_type,
            "minigame_state": self.latest_minigame_state,
            "corsi_state": self.corsi_state,
            "nback_state": self.nback_state
        }


# Instancia global del comunicador UDP
godot_udp = GodotUDPCommunicator()

# =========================
# Configuración inicial con ventana
# =========================
app = QtWidgets.QApplication(sys.argv)

fs_values = [125, 250]
n_ch_values = [4, 8, 16]
win_sec_values = [5, 10, 15]
mode_values = ["butterworth", "wavelet"]


class EnhancedConfigDialog(ConfigDialog):
    def __init__(self, fs_values, n_ch_values, win_sec_values, mode_values):
        super().__init__(fs_values, n_ch_values, win_sec_values, mode_values)
        self.initials_label = QtWidgets.QLabel("Iniciales del voluntario:")
        self.initials_edit = QtWidgets.QLineEdit()
        self.initials_edit.setMaxLength(3)
        self.initials_edit.setPlaceholderText("ABC")
        self.layout().insertRow(4, self.initials_label, self.initials_edit)

    def get_config(self):
        config = super().get_config()
        config["INITIALS"] = self.initials_edit.text().strip().upper() or "UNK"
        return config


dlg = EnhancedConfigDialog(fs_values, n_ch_values, win_sec_values, mode_values)
if dlg.exec_() == QtWidgets.QDialog.Rejected:
    sys.exit(0)
cfg = dlg.get_config()

FS, N_CH, WIN_SEC, MODE, INITIALS = cfg["FS"], cfg["N_CH"], cfg["WIN_SEC"], cfg["MODE"], cfg["INITIALS"]

# =========================
# CONFIGURACIÓN PROCESAMIENTO
# =========================
UPDATE_MS = 80
OFFSET = 250
THETA_BAND = (4.0, 8.0)
GAMMA_BAND = (30.0, 80.0)
EPS = 1e-12

# =========================
# Inicializar Board
# =========================
board = init_board()
print('init')
eeg_channels = get_eeg_channels(board, N_CH)
print('channels')

# =========================
# Buffers
# =========================
WIN_SAMPLES = WIN_SEC * FS
buffers = [deque(np.zeros(WIN_SAMPLES), maxlen=WIN_SAMPLES) for _ in range(N_CH)]
data_buffer = []

# =========================
# Interfaz gráfica
# =========================
pg.setConfigOptions(antialias=True, background='#111218', foreground='w')
main, ui = create_ui(N_CH, WIN_SEC, OFFSET, FS)
ch_sel = {"idx": 0}
connect_channel_controls(ui, N_CH, lambda new_idx: ch_sel.update(idx=new_idx))

# =========================
# Update loop
# =========================
t0 = time.time()
session_start_time = t0
sample_counter = 0


def update():
    global sample_counter

    # Usamos get_board_data() para evitar duplicados en CSV
    data = board.get_board_data()

    if data.shape[1] == 0:
        return

    # 1. Actualizar buffers circulares (Deque)
    for i, ch in enumerate(eeg_channels):
        buffers[i].extend(data[ch])

    # 2. Procesamiento Neurofeedback
    ratio = update_loop(buffers, FS, THETA_BAND, GAMMA_BAND, EPS,
                        ui, t0, ch_sel["idx"], WIN_SEC, OFFSET, MODE)

    # 3. Enviar a Godot
    godot_udp.send_ratio_to_godot(ratio)

    # 4. Guardar datos en Buffer CSV
    eeg_data_new = data[eeg_channels].T
    current_state = godot_udp.get_current_state()
    num_new_samples = data.shape[1]

    # Pre-calculamos la metadata que es igual para este bloque de datos
    # (Optimización para no recrear la lista en cada micro-iteración)
    metadata_block = [
        ratio,
        current_state["game_state"],
        current_state["energy"],
        current_state["mission_state"],
        current_state["minigame_type"],
        current_state["minigame_state"],
        current_state["corsi_state"],
        current_state["nback_state"],
        MODE
    ]

    for i in range(num_new_samples):
        # Timestamp relativo exacto basado en conteo de muestras
        sample_time = (sample_counter / FS)

        row = [sample_time, sample_counter]
        # Añadir EEG del canal i
        row.extend(eeg_data_new[i])
        # Añadir metadata
        row.extend(metadata_block)

        data_buffer.append(row)
        sample_counter += 1


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(UPDATE_MS)

# =========================
# Guardado periódico
# =========================
SAVE_INTERVAL = 5
last_save = time.time()
save_dir = "recordings"
os.makedirs(save_dir, exist_ok=True)
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

csv_filename = os.path.join(save_dir, f"session_{INITIALS}_{session_id}_{MODE}.csv")

headers = [
              "timestamp",
              "sample_index"
          ] + [f"channel_{i + 1}" for i in range(N_CH)] + [
              "ratio",
              "game_state",
              "energy",
              "mission_state",
              "minigame_type",
              "minigame_state",
              "corsi_state",
              "nback_state",
              "filter_mode"
          ]

with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)

print(f"Archivo CSV creado: {csv_filename}")
print(f"Voluntario: {INITIALS}, Modo: {MODE}")


def save_data():
    global data_buffer
    if not data_buffer:
        return

    try:
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data_buffer)

        data_buffer = []

    except Exception as e:
        print(f"Error guardando datos: {e}")


def periodic_save():
    global last_save
    if time.time() - last_save >= SAVE_INTERVAL:
        save_data()
        last_save = time.time()


save_timer = QtCore.QTimer()
save_timer.timeout.connect(periodic_save)
save_timer.start(1000)


# =========================
# Cierre seguro
# =========================
def close_application():
    print("Cerrando aplicación...")
    timer.stop()
    save_timer.stop()

    save_data()

    try:
        if board.is_prepared():
            board.stop_stream()
            board.release_session()
    except:
        pass

    godot_udp.udp_socket.close()


def close_event(event):
    close_application()
    event.accept()


main.closeEvent = close_event

# =========================
# Run
# =========================
if __name__ == '__main__':
    try:
        main.show()
        print(f"Sistema iniciado - Voluntario: {INITIALS}")
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error crítico: {e}")
    finally:
        close_application()