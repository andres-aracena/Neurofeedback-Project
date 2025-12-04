import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.signal import welch, butter, sosfiltfilt, iirnotch, filtfilt
import os
import csv
import pywt # Necesario para wavelet


# ==========================================
# FUNCIONES DE FILTRADO (TUS FUNCIONES EXACTAS)
# ==========================================

def highpass_sos(x, cutoff=0.5, order=4, fs=250):
    """Filtro paso alto para eliminar deriva (DC)"""
    sos = butter(order, cutoff / (fs / 2), btype='highpass', output='sos')
    return sosfiltfilt(sos, x)

def bandpass_sos(x, low, high, order=6, fs=250, padlen=None):
    """
    Filtro Pasa-Banda robusto (Tu función).
    """
    nyq = fs / 2
    lown, highn = low / nyq, high / nyq
    sos = butter(order, [lown, highn], btype='band', output='sos')
    pad = padlen if padlen is not None else 3 * (max(len(sos), 1))
    return sosfiltfilt(sos, x, padlen=pad)

def notch_filter(x, notch_freq=50.0, q=10.0, fs=250, aggressive=True):
    """
    Filtro Notch Reforzado.
    """
    b, a = iirnotch(notch_freq, q, fs)
    y = filtfilt(b, a, x)  # Primera pasada
    if aggressive:
        y = filtfilt(b, a, y)  # Segunda pasada (Cascada)
    return y


def preprocess_signal(x, fs=250):
    """Cadena completa de pre-procesado (TUS REGLAS)"""
    # 1. Quitar DC y deriva lenta
    x = highpass_sos(x, cutoff=0.5, fs=fs)

    # 2. Notch en 50 Hz (Fundamental)
    x = notch_filter(x, notch_freq=50.0, q=5.0, fs=fs, aggressive=False)

    # 3. Notch en 100 Hz (Primer Armónico)
    x = notch_filter(x, notch_freq=100.0, q=10.0, fs=fs, aggressive=False)

    return x


# ==========================================
# CONFIGURACIÓN DE COLORES
# ==========================================
STATE_COLORS = {
    'in_corsi_minigame': '#3498db',  # AZUL (Corsi)
    'in_nback_minigame': '#2ecc71',  # VERDE (N-Back)
    'exploring': '#95a5a6',  # Gris Medio
    'disconnected': '#7f8c8d',  # Gris Oscuro
    'unknown': '#bdc3c7',  # Gris Claro
    'paused': '#bdc3c7',
    'INTRO': '#bdc3c7'
}
DEFAULT_COLOR = '#ecf0f1'


class NeuroAnalyzerTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Neurofeedback Analyzer Pro - Research Edition")
        self.root.geometry("1500x800")

        # Estado de datos
        self.df = None
        self.df_clean = None  # DataFrame para estadísticas limpias
        self.raw_eeg = None
        self.fs = 250
        self.metadata = {}

        self.setup_ui()

    def setup_ui(self):
        style = ttk.Style()
        style.configure("Header.TLabel", font=('Segoe UI', 12, 'bold'))
        style.configure("Big.TButton", font=('Segoe UI', 10))

        # --- BARRA SUPERIOR ---
        top_bar = ttk.Frame(self.root, padding=10)
        top_bar.pack(fill=tk.X)

        ttk.Button(top_bar, text="📂 Cargar Archivo (CSV / NPZ)", command=self.load_file, style="Big.TButton").pack(
            side=tk.LEFT, padx=5)
        self.lbl_status = ttk.Label(top_bar, text="Esperando archivo...", foreground="gray")
        self.lbl_status.pack(side=tk.LEFT, padx=15)

        ttk.Button(top_bar, text="❌ Cerrar Gráficas", command=lambda: plt.close('all')).pack(side=tk.RIGHT)

        # --- CONTENEDOR PRINCIPAL ---
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # === PANEL IZQUIERDO: CONTROLES ===
        left_panel = ttk.Frame(main_container, width=350)
        main_container.add(left_panel, weight=1)

        # Grupo 1: Señal
        grp_signal = ttk.LabelFrame(left_panel, text="📡 Análisis de Señal", padding=10)
        grp_signal.pack(fill=tk.X, pady=5)
        ttk.Button(grp_signal, text="🧠 Señal EEG (Navegable)", command=self.plot_clean_eeg).pack(fill=tk.X, pady=2)
        ttk.Button(grp_signal, text="📊 Espectro Frecuencia (PSD Filtrado)", command=self.plot_psd_analysis).pack(
            fill=tk.X, pady=2)

        # Grupo 2: Contexto
        grp_game = ttk.LabelFrame(left_panel, text="🎮 Contexto y Memoria", padding=10)
        grp_game.pack(fill=tk.X, pady=5)
        ttk.Button(grp_game, text="🎨 Ratio + Contexto (Colores)", command=self.plot_combined_context).pack(fill=tk.X,
                                                                                                           pady=2)
        ttk.Button(grp_game, text="📦 Distribución (Boxplot)", command=self.plot_state_distribution).pack(fill=tk.X,
                                                                                                         pady=2)
        ttk.Button(grp_game, text="⚡ Energía del Jugador", command=self.plot_energy).pack(fill=tk.X, pady=2)

        # Grupo 3: Reportes
        grp_report = ttk.LabelFrame(left_panel, text="📋 Reportes", padding=10)
        grp_report.pack(fill=tk.X, pady=5)
        ttk.Button(grp_report, text="📑 Resumen Ejecutivo", command=self.show_summary).pack(fill=tk.X, pady=2)
        ttk.Button(grp_report, text="🔍 Informe de Calidad", command=self.show_quality_report).pack(fill=tk.X, pady=2)

        # === PANEL DERECHO: DASHBOARD ===
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=4)

        right_panel.columnconfigure(0, weight=1);
        right_panel.columnconfigure(1, weight=1)
        right_panel.rowconfigure(0, weight=1);
        right_panel.rowconfigure(1, weight=1)

        self._create_dashboard_frame(right_panel, "ℹ️ Metadatos Sesión", 0, 0, 'txt_info')
        self._create_dashboard_frame(right_panel, "📊 Memoria de Trabajo (Theta/Gamma)", 0, 1, 'txt_stats')
        self._create_dashboard_frame(right_panel, "🎮 Logs del Juego", 1, 0, 'txt_gamelog')
        self._create_dashboard_frame(right_panel, "📉 Logs Señal (uV)", 1, 1, 'txt_signallog')

    def _create_dashboard_frame(self, parent, title, r, c, attr_name):
        frm = ttk.LabelFrame(parent, text=title, padding=5)
        frm.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)
        txt = tk.Text(frm, height=10, width=40, font=('Consolas', 9))
        txt.pack(fill=tk.BOTH, expand=True)
        setattr(self, attr_name, txt)

    def load_file(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("All Supported", "*.csv *.npz"), ("CSV Files", "*.csv"), ("NPZ Files", "*.npz")])
        if not filepath: return

        self.lbl_status.config(text="Procesando...", foreground="orange")
        self.root.update()

        try:
            ext = os.path.splitext(filepath)[1].lower()
            if ext == '.npz':
                self._process_npz(filepath)
            else:
                self._process_csv(filepath)

            self.lbl_status.config(text=f"Cargado: {os.path.basename(filepath)}", foreground="green")
            self._update_dashboard()
            messagebox.showinfo("Éxito", "Sesión cargada correctamente.")
        except Exception as e:
            self.lbl_status.config(text="Error de carga", foreground="red")
            messagebox.showerror("Error", f"No se pudo leer el archivo:\n{str(e)}")

    # --- PROCESAMIENTO DE ARCHIVOS ---
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
                    if ts <= last_ts:
                        dropped_duplicates += 1
                        continue

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

                    clean_row = {'timestamp': ts, **{k: row.get(k, '') for k in meta_cols if k != 'timestamp'}}
                    for i, col in enumerate(ch_cols): clean_row[col] = current_eeg[i]

                    data_rows.append(clean_row)
                    last_ts = ts
                except ValueError:
                    continue

        self.df = pd.DataFrame(data_rows)
        self._finalize_processing(ch_cols, filepath, dropped_duplicates, repaired_zeros)

    def _process_npz(self, filepath):
        data = np.load(filepath, allow_pickle=True)
        if 'eeg_data' in data:
            eeg = data['eeg_data']
        elif 'eeg' in data:
            eeg = data['eeg']
        else:
            raise ValueError("No EEG data found")

        if eeg.shape[0] < eeg.shape[1]: eeg = eeg.T

        ts = data.get('timestamps', np.arange(eeg.shape[0]) / 250)
        min_len = min(len(ts), eeg.shape[0])

        df_dict = {'timestamp': ts[:min_len]}
        ch_cols = []
        for i in range(eeg.shape[1]):
            c = f'channel_{i + 1}'
            df_dict[c] = eeg[:min_len, i]
            ch_cols.append(c)

        r = data.get('neurofeedback_ratios', data.get('ratios', np.zeros(min_len)))
        if len(r) > min_len:
            r = r[:min_len]
        elif len(r) < min_len:
            r = np.pad(r, (0, min_len - len(r)), 'edge')
        df_dict['ratio'] = r

        states = np.full(min_len, 'unknown', dtype=object)
        energies = np.zeros(min_len)
        if 'game_states' in data:
            gs = data['game_states']
            factor = min_len / max(1, len(gs))
            for i in range(min_len):
                idx = int(i / factor)
                if idx < len(gs):
                    states[i] = gs[idx].get('game_state', 'unknown')
                    energies[i] = gs[idx].get('energy', 0)

        df_dict['game_state'] = states
        df_dict['energy'] = energies

        self.df = pd.DataFrame(df_dict)
        self._finalize_processing(ch_cols, filepath, 0, 0, is_legacy=True)

    def _finalize_processing(self, ch_cols, filepath, dups, zeros, is_legacy=False):
        for col in ['ratio', 'energy'] + ch_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

        eeg_matrix = self.df[ch_cols].values
        scale_msg = "Ninguno"
        if np.max(np.abs(eeg_matrix)) > 50000:
            eeg_matrix = eeg_matrix * 0.02235
            scale_msg = "OpenBCI Raw -> uV"

        eeg_matrix = eeg_matrix - np.mean(eeg_matrix, axis=0)
        self.raw_eeg = eeg_matrix.T
        self.ch_names = ch_cols

        if len(self.df) > 1:
            diffs = np.diff(self.df['timestamp'])
            valid = diffs[diffs > 0]
            self.fs = int(1.0 / np.median(valid)) if len(valid) > 0 else 250

        # --- APLICAR LÓGICA DE LIMPIEZA DE BLOQUES ---
        # 1. Unificar estados ruidosos
        if 'game_state' in self.df.columns:
            self.df['game_state'] = self.df['game_state'].replace(['disconnected', 'unknown', 'paused', 'INTRO'],
                                                                  'exploring')

            # 2. Eliminar el PRIMER bloque "exploring" (Ajustes iniciales)
            self.df['grp'] = (self.df['game_state'] != self.df['game_state'].shift()).cumsum()
            first_libre_grps = self.df[self.df['game_state'] == 'exploring']['grp'].unique()

            if len(first_libre_grps) > 0:
                first_grp_idx = first_libre_grps[0]
                self.df_clean = self.df[self.df['grp'] != first_grp_idx].copy()
            else:
                self.df_clean = self.df.copy()
        else:
            self.df_clean = self.df.copy()

        self.metadata = {
            'filename': os.path.basename(filepath),
            'total_rows': len(self.df),
            'dropped_duplicates': dups,
            'repaired_zeros': zeros,
            'scaling': scale_msg,
            'format': 'Legacy NPZ' if is_legacy else 'CSV',
            'duration': self.df['timestamp'].iloc[-1] - self.df['timestamp'].iloc[0]
        }

    def _update_dashboard(self):
        info = f"=== INFORMACIÓN ({self.metadata['format']}) ===\n"
        info += f"📁 {self.metadata['filename']}\n"
        info += f"⏱️ Duración: {self.metadata['duration']:.2f}s | FS: {self.fs}Hz\n"
        info += f"🔧 Reparaciones: {self.metadata['dropped_duplicates']} Dupl, {self.metadata['repaired_zeros']} Ceros\n"
        self.txt_info.delete(1.0, tk.END);
        self.txt_info.insert(1.0, info)

        # --- CALCULO DE ESTADÍSTICAS REPRODUCIBLES (Recorte 8s) ---
        stats = "=== MEMORIA DE TRABAJO ===\n"

        states_map = {
            'exploring': 'Libre',
            'in_corsi_minigame': 'Corsi',
            'in_nback_minigame': 'N-Back'
        }

        # Acumuladores de datos recortados
        valid_ratios = {k: [] for k in states_map.keys()}

        if self.df_clean is not None and not self.df_clean.empty:
            # Iterar bloques reales en df_clean
            for _, block in self.df_clean.groupby('grp'):
                state = block['game_state'].iloc[0]
                if state in states_map:
                    total_samples = len(block)
                    trim_samples = 8 * self.fs

                    filtered_data = []
                    # Regla de Recorte Estricta (Igual a Data Extractor)
                    if total_samples > (trim_samples * 2):
                        filtered_data = block['ratio'].iloc[trim_samples:-trim_samples].values
                    elif state == 'exploring' and total_samples > (2 * self.fs):
                        # Exploring corto se salva con recorte minimo
                        filtered_data = block['ratio'].iloc[self.fs:-self.fs].values

                    if len(filtered_data) > 0:
                        valid_ratios[state].extend(filtered_data)

        # Mostrar Resultados
        for k, name in states_map.items():
            data = np.array(valid_ratios[k])
            if len(data) > 0:
                stats += f"• {name}:\n"
                stats += f"  Media: {np.mean(data):.3f} ±{np.std(data):.3f}\n"
                stats += f"  Tiempo Útil: {len(data) / self.fs:.1f}s\n"
            else:
                stats += f"• {name}: Sin datos válidos\n"

        self.txt_stats.delete(1.0, tk.END);
        self.txt_stats.insert(1.0, stats)

        cols = [c for c in ['timestamp', 'game_state', 'energy', 'minigame_type'] if c in self.df.columns]
        self.txt_gamelog.delete(1.0, tk.END)
        self.txt_gamelog.insert(1.0, self.df[cols].iloc[::20].to_string(index=False))

        if self.raw_eeg is not None:
            ch_prev = self.ch_names[:4]
            prev_df = pd.DataFrame(self.raw_eeg.T[:, :4], columns=ch_prev)
            prev_df.insert(0, 't', self.df['timestamp'])
            self.txt_signallog.delete(1.0, tk.END)
            self.txt_signallog.insert(1.0, prev_df.head(5).to_string(index=False))

    # ==========================================
    # GRÁFICAS MEJORADAS (V3)
    # ==========================================

    def plot_clean_eeg(self):
        if self.raw_eeg is None: return
        n_ch = self.raw_eeg.shape[0]
        t = self.df['timestamp'].values
        fig, axes = plt.subplots(n_ch, 1, figsize=(14, 2 * n_ch), sharex=True)
        if n_ch == 1: axes = [axes]
        for i in range(n_ch):
            axes[i].plot(t, self.raw_eeg[i], lw=0.6, color='#2980b9')
            axes[i].set_ylabel(f'Ch{i + 1}', fontsize=8)
            axes[i].grid(True, alpha=0.3)
            mx = np.percentile(np.abs(self.raw_eeg[i]), 99.5) * 1.5
            axes[i].set_ylim(-mx, mx)
        axes[-1].set_xlabel('Tiempo (s)')
        if t[-1] > 60: axes[-1].set_xlim(t[0], t[0] + 60)
        plt.tight_layout()
        plt.show()

    def plot_psd_analysis(self):
        """[Image of signal processing flowchart]

        Muestra el espectro de frecuencia en 4 capas para validar el filtrado:
        1. Gris: Señal Cruda (Ruido presente)
        2. Morado: Pre-procesada (Notch 50Hz + Lowpass 100Hz)
        3. VERDE: Banda Theta Aislada (4-8 Hz)
        4. ROJO: Banda Gamma Aislada (30-80 Hz)
        """
        if self.raw_eeg is None: return

        print("Calculando PSD multicapa (Cruda, Pre, Theta, Gamma)...")

        f_axis = None
        psd_raw = None
        psd_pre = None

        # Calcular PSD promedio de todos los canales
        for i in range(self.raw_eeg.shape[0]):
            # 1. Señal Cruda
            f, Pxx_r = welch(self.raw_eeg[i], fs=self.fs, nperseg=self.fs * 2)

            # 2. Pre-procesada (Limpieza básica: Notch + Drift + LP General)
            sig_pre = preprocess_signal(self.raw_eeg[i], fs=self.fs)
            _, Pxx_p = welch(sig_pre, fs=self.fs, nperseg=self.fs * 2)

            # Acumular promedios
            if psd_raw is None:
                psd_raw = Pxx_r;
                psd_pre = Pxx_p;
                f_axis = f
            else:
                psd_raw += Pxx_r;
                psd_pre += Pxx_p;

        # Normalizar promedios
        N = self.raw_eeg.shape[0]
        psd_raw /= N;
        psd_pre /= N;

        # --- PLOTEO ---
        plt.figure(figsize=(12, 7))

        # 1. Cruda (Fondo)
        plt.semilogy(f_axis, psd_raw, color='#3c3c3b', alpha=0.3, lw=1, label='1. Original (Sin Filtro)')

        # 2. Preprocesada (Base Limpia)
        plt.semilogy(f_axis, psd_pre, color='#8e44ad', lw=1.5, alpha=0.6, label='2. Preprocesada (Sin ruido 50Hz)')

        plt.title("Análisis Espectral: Original vs Procesada (Promedio Global)", fontsize=12)
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Potencia (uV²/Hz)")
        plt.xlim(1, 100)

        # Ajuste dinámico del eje Y para que se vean bien las bandas filtradas
        # Las bandas filtradas caen mucho fuera de su rango, así que fijamos el mínimo visual
        max_pow = np.max(psd_raw)
        plt.ylim(max_pow * 1e-8, max_pow * 10)

        plt.grid(True, which="both", alpha=0.3)

        # Marcar zonas de interés
        plt.axvspan(4, 8, color='green', alpha=0.1, label='Theta (4-8Hz)')
        plt.axvspan(30, 80, color='red', alpha=0.05, label='Gamma (30-80Hz)')

        # Marcar cortes de filtro
        plt.axvline(50, color='red', ls=':', alpha=0.5, label='Notch 50Hz')

        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    def plot_combined_context(self):
        """Ratio + Contexto + TENDENCIAS"""
        if self.df is None: return
        fig, ax1 = plt.subplots(figsize=(15, 6))

        t = self.df['timestamp'].values
        ratio = self.df['ratio'].values
        states = self.df['game_state'].values if 'game_state' in self.df.columns else None

        if states is not None:
            # Agrupar por bloques para pintar y calcular tendencias
            self.df['grp'] = (self.df['game_state'] != self.df['game_state'].shift()).cumsum()

            legend_patches = {}
            for _, block in self.df.groupby('grp'):
                state_name = block['game_state'].iloc[0]
                t_block = block['timestamp'].values
                r_block = block['ratio'].values

                # Color de fondo
                c = STATE_COLORS.get(state_name, DEFAULT_COLOR)
                ax1.axvspan(t_block[0], t_block[-1], color=c, alpha=0.3, ec=None)

                if state_name not in legend_patches:
                    legend_patches[state_name] = mpatches.Patch(color=c, label=state_name, alpha=0.3)

                # --- LÍNEA DE TENDENCIA (Regresión Lineal por bloque) ---
                # Solo si el bloque tiene datos suficientes (>5s)
                if len(t_block) > 5 * self.fs:
                    try:
                        # Ajuste polinómico grado 1
                        z = np.polyfit(t_block, r_block, 1)
                        p = np.poly1d(z)
                        ax1.plot(t_block, p(t_block), color='black', linestyle='--', linewidth=1.5, alpha=0.7)
                    except:
                        pass

            # Leyenda manual para incluir la tendencia
            handles = list(legend_patches.values())
            handles.append(mlines.Line2D([], [], color='black', linestyle='--', label='Tendencia (Regresión)'))
            ax1.legend(handles=handles, loc='upper right', title="Contexto")

        ax1.plot(t, ratio, color='#2c3e50', lw=1.5, label='Ratio Instantáneo')
        ax1.set_ylabel('Ratio Theta/Gamma');
        ax1.set_ylim(0, 1.05)

        # Grid añadido
        ax1.grid(True, which='major', linestyle='-', alpha=0.5)
        ax1.minorticks_on()
        ax1.grid(True, which='minor', linestyle=':', alpha=0.2)

        plt.title("Dinámica de Memoria de Trabajo con Tendencias por Tarea");
        plt.tight_layout()
        plt.show()

    def plot_state_distribution(self):
        """Boxplot con Leyenda Explicativa"""
        # Usamos df_clean para la distribución (sin el primer bloque)
        if self.df_clean is None or 'game_state' not in self.df_clean.columns: return

        data, labels, colors = [], [], []

        # Orden preferido para visualización lógica
        unique_states = ['exploring', 'in_corsi_minigame', 'in_nback_minigame']
        # Filtrar solo los que existen en los datos
        existing_states = [s for s in unique_states if s in self.df_clean['game_state'].unique()]

        for s in existing_states:
            r = self.df_clean[self.df_clean['game_state'] == s]['ratio'].values
            if len(r) > 10:
                data.append(r);
                labels.append(s)
                colors.append(STATE_COLORS.get(s, DEFAULT_COLOR))

        if not data: return  # Evitar error si no hay datos

        fig, ax = plt.subplots(figsize=(10, 6))

        # Estilo de outliers (círculos)
        flierprops = dict(marker='o', markerfacecolor='white', markersize=5,
                          linestyle='none', markeredgecolor='gray')

        bp = ax.boxplot(data, patch_artist=True, labels=labels, flierprops=flierprops)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title("Distribución Estadística por Tarea (Post-Limpieza)")
        ax.set_ylabel("Ratio Theta/Gamma")
        ax.grid(True, axis='y', alpha=0.3)

        # --- LEYENDA EXPLICATIVA ---
        legend_elements = [
            mpatches.Patch(facecolor='gray', edgecolor='black', alpha=0.3,
                           label='Caja: Rango Intercuartil (IQR)\n(50% central de los datos)'),
            mlines.Line2D([], [], color='black', label='Línea: Mediana'),
            mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                          markerfacecolor='white', markeredgecolor='gray',
                          label='Círculos: Valores Atípicos (Outliers)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

        plt.tight_layout()
        plt.show()

    def plot_energy(self):
        if 'energy' not in self.df.columns: return
        plt.figure(figsize=(12, 5))
        plt.plot(self.df['timestamp'], self.df['energy'], color='#e67e22')
        plt.title("Energía");
        plt.show()

    def show_summary(self):
        # Similar logic to dashboard update but in popup
        if self.df is None: return
        messagebox.showinfo("Resumen", "Ver panel derecho para estadísticas detalladas coincidentes con Extractor.")

    def show_quality_report(self):
        if self.raw_eeg is None: return
        sat = np.mean(np.abs(self.raw_eeg) > 1000) * 100

        # PSD para chequear 50Hz
        f, Pxx = welch(self.raw_eeg[0], fs=self.fs, nperseg=self.fs)
        idx_50 = np.argmin(np.abs(f - 50))
        has_noise = Pxx[idx_50] > (np.mean(Pxx) * 10)

        msg = f"INFORME TÉCNICO\n{'=' * 20}\nIntegridad Señal: {100 - sat:.2f}%\n"
        msg += f"Ruido 50Hz: {'⚠️ PRESENTE' if has_noise else '✅ LIMPIO'}\n"
        msg += f"Reparaciones: {self.metadata['dropped_duplicates']} Duplicados"
        messagebox.showinfo("Calidad", msg)


if __name__ == "__main__":
    root = tk.Tk()
    app = NeuroAnalyzerTool(root)
    root.mainloop()