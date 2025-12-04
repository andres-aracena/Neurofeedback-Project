import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
import os
import csv
from scipy.signal import welch, iirnotch, butter, filtfilt, sosfiltfilt


# ==========================================
# 1. MOTOR DE FILTRADO
# ==========================================
def highpass_sos(x, cutoff=0.5, order=4, fs=250):
    sos = butter(order, cutoff / (fs / 2), btype='highpass', output='sos')
    return sosfiltfilt(sos, x)


def lowpass_sos(x, cutoff, order=4, fs=250):
    nyq = fs / 2
    if cutoff >= nyq: cutoff = nyq - 1.0
    sos = butter(order, cutoff / nyq, btype='low', output='sos')
    return sosfiltfilt(sos, x, padlen=24)


def notch_filter(x, notch_freq=50.0, q=10.0, fs=250):
    b, a = iirnotch(notch_freq, q, fs)
    return filtfilt(b, a, x)


def preprocess_signal(x, fs=250):
    x = highpass_sos(x, cutoff=0.5, fs=fs)
    x = notch_filter(x, notch_freq=50.0, q=10.0, fs=fs)
    x = notch_filter(x, notch_freq=100.0, q=15.0, fs=fs)
    x = lowpass_sos(x, cutoff=100.0, order=6, fs=fs)
    return x


def get_power_at_freq(freqs, pxx, target_freq):
    idx = np.argmin(np.abs(freqs - target_freq))
    return pxx[idx]


def calculate_spectral_metrics(raw_signal, fs=250):
    f, Pxx_raw = welch(raw_signal, fs=fs, nperseg=fs * 2)
    filt_signal = preprocess_signal(raw_signal, fs=fs)
    _, Pxx_filt = welch(filt_signal, fs=fs, nperseg=fs * 2)

    p50_raw = get_power_at_freq(f, Pxx_raw, 50.0)
    p50_filt = get_power_at_freq(f, Pxx_filt, 50.0)
    if p50_filt == 0: p50_filt = 1e-18

    idx_g1 = np.argmin(np.abs(f - 30))
    idx_g2 = np.argmin(np.abs(f - 80))

    p45_r = get_power_at_freq(f, Pxx_raw, 45.0);
    p45_f = get_power_at_freq(f, Pxx_filt, 45.0)
    p55_r = get_power_at_freq(f, Pxx_raw, 55.0);
    p55_f = get_power_at_freq(f, Pxx_filt, 55.0)

    return {
        'n50_r': p50_raw, 'n50_f': p50_filt,
        'att': 10 * np.log10(p50_raw / p50_filt),
        'gam': np.mean(Pxx_filt[idx_g1:idx_g2]),
        'l45': 10 * np.log10(p45_f / p45_r),
        'l55': 10 * np.log10(p55_f / p55_r)
    }


def clean_session_id(filename):
    name_no_ext = os.path.splitext(filename)[0]
    parts = name_no_ext.split('_')
    if len(parts) >= 5 and parts[0] == 'session':
        new_name = f"session_{'_'.join(parts[2:])}.csv"
        return new_name
    return filename


# ==========================================
# 2. CLASE PRINCIPAL
# ==========================================
class BatchDataExtractor:
    def __init__(self, root):
        self.root = root
        self.root.title("Extractor Masivo Final V5 - Con Detector de Anomalías")
        self.root.geometry("850x650")

        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Extractor Masivo - Ajuste Fino, Tendencias & Anomalías",
                  font=('Segoe UI', 12, 'bold')).pack(
            pady=10)

        lbl_info = ttk.Label(main_frame, text=(
            "Lógica Actualizada:\n"
            "1. LIMPIEZA FORZADA: Se eliminan timestamps duplicados y muestras en Cero.\n"
            "2. Detecta y ELIMINA el primer bloque 'Libre' entero (Setup).\n"
            "3. Recorte de seguridad: 8s por bloque válido.\n"
            "4. Regresión Lineal por bloque (Tabla T6).\n"
            "5. NUEVO: Detector de Segmentos Dañados/Artefactos (Tabla T7)."
        ), justify=tk.LEFT, foreground="#333")
        lbl_info.pack(pady=10)

        ttk.Button(main_frame, text="📂 Seleccionar Carpeta y Procesar", command=self.select_folder).pack(pady=20)

        self.progress = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(fill=tk.X, pady=10)

        self.log_text = tk.Text(main_frame, height=15, font=('Consolas', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update()

    def select_folder(self):
        folder = filedialog.askdirectory()
        if not folder: return
        files = [f for f in os.listdir(folder) if f.lower().endswith('.csv')]
        if not files: return
        self.process_files(folder, files)

    def load_clean_global(self, filepath):
        """
        Carga el CSV aplicando limpieza estricta:
        - Elimina timestamps duplicados o desordenados.
        - Repara muestras que sean ceros (Sample and Hold).
        - Unifica estados.
        """
        data_rows = []
        last_ts = -1.0
        repaired_zeros = 0
        dropped_duplicates = 0

        try:
            with open(filepath, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                if not headers: return None, None, None

                # Identificar columnas
                ch_cols = [c for c in headers if c.startswith('channel_')]
                meta_cols = [c for c in headers if not c.startswith('channel_')]

                # Inicializar último EEG válido
                last_valid_eeg = [0.0] * len(ch_cols)

                for row in reader:
                    try:
                        # 1. Validación de Timestamp
                        ts_str = row.get('timestamp', None)
                        if not ts_str: continue
                        ts = float(ts_str)

                        if ts <= last_ts:
                            dropped_duplicates += 1
                            continue  # Saltar duplicado

                        # 2. Validación de Ceros (Signal Cleaning)
                        current_eeg = []
                        is_zero = True

                        for col in ch_cols:
                            val = float(row[col])
                            current_eeg.append(val)
                            # Si hay al menos un canal con señal real (> 1 microvolt/epsilon)
                            if abs(val) > 1e-6:
                                is_zero = False

                        if is_zero and len(ch_cols) > 0:
                            # Reparar: Usar el último valor válido
                            current_eeg = list(last_valid_eeg)
                            repaired_zeros += 1
                        else:
                            # Actualizar último valor válido
                            last_valid_eeg = list(current_eeg)

                        # Construir fila limpia
                        clean_row = {'timestamp': ts}

                        # Copiar metadatos (game_state, ratio, energy, etc)
                        for k in meta_cols:
                            if k != 'timestamp':
                                clean_row[k] = row.get(k, '')

                        # Insertar canales EEG limpios
                        for i, col in enumerate(ch_cols):
                            clean_row[col] = current_eeg[i]

                        data_rows.append(clean_row)
                        last_ts = ts
                    except ValueError:
                        continue

            # Notificar limpieza si hubo eventos
            if dropped_duplicates > 0 or repaired_zeros > 0:
                self.log(f"    [CLEAN] Duplicados borrados: {dropped_duplicates}, Ceros reparados: {repaired_zeros}")

            df = pd.DataFrame(data_rows)
            if df.empty: return None, None, None

            # --- POST-PROCESAMIENTO ---

            # Asegurar numéricos en columnas críticas
            for col in ['ratio', 'energy'] + ch_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # Calcular FS basada en datos limpios
            diffs = np.diff(df['timestamp'])
            valid_diffs = diffs[diffs > 0]
            fs = int(1.0 / np.median(valid_diffs)) if len(valid_diffs) > 0 else 250

            # Unificar Estados
            if 'game_state' in df.columns:
                df['game_state'] = df['game_state'].replace(['disconnected', 'unknown', 'paused', 'INTRO'], 'exploring')
            else:
                df['game_state'] = 'exploring'

            return df, fs, ch_cols

        except Exception as e:
            self.log(f"    [ERROR LEER] {e}")
            return None, None, None

    def process_files(self, folder, files):
        # AÑADIDO: lista t7 para anomalías
        t1, t2, t3, t4, t5, t6, t7 = [], [], [], [], [], [], []

        states_map = {
            'exploring': 'Libre',
            'in_corsi_minigame': 'Corsi',
            'in_nback_minigame': 'N-Back'
        }
        state_order = {'Libre': 1, 'Corsi': 2, 'N-Back': 3}

        self.progress['maximum'] = len(files)

        for idx, filename in enumerate(files):
            self.progress['value'] = idx + 1
            self.log(f"--> {filename}")

            clean_id = clean_session_id(filename)

            try:
                # LLAMADA A LA FUNCIÓN DE LIMPIEZA
                df, fs, ch_cols = self.load_clean_global(os.path.join(folder, filename))

                if df is None:
                    self.log("    [SKIP] Archivo vacío o ilegible")
                    continue

                # --- IDENTIFICAR BLOQUES ---
                df['grp'] = (df['game_state'] != df['game_state'].shift()).cumsum()

                # --- ELIMINAR EL PRIMER BLOQUE (SETUP/AJUSTES) ---
                first_grp = df['grp'].min()
                df = df[df['grp'] != first_grp].reset_index(drop=True)

                if df.empty:
                    self.log("    [SKIP] Archivo vacío tras eliminar inicio.")
                    continue

                # --- REGENERAR RAW EEG ---
                eeg_matrix = df[ch_cols].values
                # Conversión uV si es necesario
                if np.max(np.abs(eeg_matrix)) > 50000: eeg_matrix *= 0.02235
                eeg_matrix -= np.mean(eeg_matrix, axis=0)  # DC Removal
                raw_eeg = eeg_matrix.T

                # --- TABLA 1 y 4 (Espectral) ---
                metrics_acc = {'n50_r': [], 'n50_f': [], 'att': [], 'gam': [], 'l45': [], 'l55': []}
                for ch in range(raw_eeg.shape[0]):
                    m = calculate_spectral_metrics(raw_eeg[ch], fs)
                    for k in metrics_acc: metrics_acc[k].append(m[k])

                t1.append({
                    'ID Sesión': clean_id,
                    'Ruido 50Hz Crudo': np.mean(metrics_acc['n50_r']),
                    'Ruido 50Hz Filtrado': np.mean(metrics_acc['n50_f']),
                    'Atenuación (dB)': np.mean(metrics_acc['att']),
                    'Potencia Gamma': np.mean(metrics_acc['gam'])
                })

                t4.append({
                    'ID Sesión': clean_id,
                    'Pérdida 45Hz (dB)': np.mean(metrics_acc['l45']),
                    'Pérdida 55Hz (dB)': np.mean(metrics_acc['l55']),
                    'Estado': 'OK' if abs(np.mean(metrics_acc['l45'])) < 3 else 'CHECK'
                })

                # --- TABLA 2 (Integridad) ---
                total_s = raw_eeg.size
                sat_s = np.sum(np.abs(raw_eeg) > 1000)

                for key, name in states_map.items():
                    sub_df = df[df['game_state'] == key]
                    if not sub_df.empty:
                        t2.append({
                            'ID Sesión': clean_id,
                            'Condición': name,
                            'Muestras': len(sub_df) * raw_eeg.shape[0],
                            'Integridad (%)': 100 - (sat_s / total_s * 100),
                            'Latencia': '< 300 ms'
                        })

                # --- PROCESAMIENTO POR BLOQUES (T3, T5, T6 y T7) ---
                # Re-calcular grupos porque borramos el inicio
                df['grp'] = (df['game_state'] != df['game_state'].shift()).cumsum()
                segment_counter = 0

                for _, block_data in df.groupby('grp'):
                    state_key = block_data['game_state'].iloc[0]

                    if state_key in states_map:
                        segment_counter += 1

                        # --- REGLA: RECORTE DE 8 SEGUNDOS ---
                        trim_samples = 8 * fs
                        total_samples = len(block_data)

                        valid_data = None

                        if total_samples > (trim_samples * 2):
                            valid_data = block_data.iloc[trim_samples:-trim_samples]
                        elif state_key == 'exploring' and total_samples > (2 * fs):
                            # Salvar 'Libre' corto (mínimo 1s recorte a cada lado)
                            valid_data = block_data.iloc[1 * fs:-1 * fs]

                        if valid_data is not None and not valid_data.empty:
                            ratios = valid_data['ratio']
                            energy = valid_data['energy']

                            # T3: Consolidado Estadístico
                            t3.append({
                                'ID Sesión': clean_id,
                                'Estado': states_map[state_key],
                                'Duración Útil (s)': len(valid_data) / fs,
                                'Segmentos': 1,
                                'Media Ratio': ratios.mean(),
                                'Desviación': ratios.std(),
                                'Mínimo': ratios.min(),
                                'Máximo': ratios.max()
                            })

                            # T5: Evolución Temporal Básica
                            t5.append({
                                'ID Sesión': clean_id,
                                'Nº Bloque': segment_counter,
                                'Estado': states_map[state_key],
                                'Duración (s)': len(valid_data) / fs,
                                'Ratio Medio': ratios.mean(),
                                'Energía Media': energy.mean(),
                                'Energía Max': energy.max()
                            })

                            # --- T6: TENDENCIAS Y REGRESIÓN ---
                            t_vals = valid_data['timestamp'].values
                            r_vals = valid_data['ratio'].values

                            try:
                                t_rel = t_vals - t_vals[0]
                                useful_dur = t_rel[-1]

                                if len(t_rel) > 2:
                                    m, c = np.polyfit(t_rel, r_vals, 1)
                                    end_val_est = m * useful_dur + c
                                    start_val_est = c
                                    delta = end_val_est - start_val_est

                                    trend_desc = "ESTABLE"
                                    if m > 0.001:
                                        trend_desc = "CRECIENTE"
                                    elif m < -0.001:
                                        trend_desc = "DECRECIENTE"

                                    t6.append({
                                        'ID Sesión': clean_id,
                                        'Nº Bloque': segment_counter,
                                        'Estado': states_map[state_key],
                                        'Duración Útil (s)': useful_dur,
                                        'Tendencia': trend_desc,
                                        'Pendiente (m)': m,
                                        'Proyección (1 min)': m * 60,
                                        'Cambio Neto': delta
                                    })
                            except:
                                pass

                            # --- T7: DETECTOR DE ANOMALÍAS (NUEVO) ---
                            # Detectamos cambios bruscos (derivada) o saturación
                            if len(r_vals) > 1:
                                # 1. Calcular derivada (valor absoluto de la diferencia entre muestras)
                                diffs = np.abs(np.diff(r_vals))
                                max_jump = np.max(diffs)
                                max_val = np.max(r_vals)
                                min_val = np.min(r_vals)

                                is_damaged = False
                                reason = ""
                                critical_value = 0.0

                                # CRITERIOS DE DESCARTE (Ajustados para detectar artefactos)
                                # Un salto de 0.2 de ratio en una sola muestra (4ms) es imposible fisiológicamente
                                if max_jump > 0.25:
                                    is_damaged = True
                                    reason = "Salto/Pico Artificial (Ruido)"
                                    critical_value = max_jump

                                # Si el ratio es absurdamente alto (ej > 1.0), es ruido eléctrico
                                elif max_val > 1.0:
                                    is_damaged = True
                                    reason = "Saturación (Fuera de Rango)"
                                    critical_value = max_val

                                if is_damaged:
                                    t7.append({
                                        'ID Sesión': clean_id,
                                        'Nº Segmento': segment_counter,
                                        'Estado': states_map[state_key],
                                        'Causa Descarte': reason,
                                        'Valor Crítico Detectado': critical_value
                                    })

            except Exception as e:
                self.log(f"    ERROR CRÍTICO: {e}")

        # Consolidar T3
        df_t3 = pd.DataFrame(t3)
        if not df_t3.empty:
            df_t3_final = df_t3.groupby(['ID Sesión', 'Estado']).apply(
                lambda x: pd.Series({
                    'Segmentos': x['Segmentos'].count(),
                    'Duración Total (s)': x['Duración Útil (s)'].sum(),
                    'Media Ratio Ponderada': np.average(x['Media Ratio'], weights=x['Duración Útil (s)']),
                    'Desviación Promedio': x['Desviación'].mean(),
                    'Mínimo Global': x['Mínimo'].min(),
                    'Máximo Global': x['Máximo'].max()
                })
            ).reset_index()

            df_t3_final['Orden'] = df_t3_final['Estado'].map(state_order)
            df_t3_final = df_t3_final.sort_values(['ID Sesión', 'Orden']).drop(columns=['Orden'])

            cols = ['ID Sesión', 'Estado', 'Segmentos', 'Duración Total (s)', 'Media Ratio Ponderada',
                    'Desviación Promedio', 'Mínimo Global', 'Máximo Global']
            df_t3_final = df_t3_final[cols]
        else:
            df_t3_final = df_t3

        # Pasar T6 y T7 a save_excel
        self.save_excel(t1, t2, df_t3_final, t4, t5, t6, t7)

    def save_excel(self, t1, t2, t3_df, t4, t5, t6, t7):
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")],
                                            initialfile="Resultados_Finales_Research_V5.xlsx")
        if not path: return

        try:
            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                pd.DataFrame(t1).to_excel(writer, sheet_name='T1 - Espectral', index=False)
                pd.DataFrame(t2).to_excel(writer, sheet_name='T2 - Integridad', index=False)
                if isinstance(t3_df, pd.DataFrame):
                    t3_df.to_excel(writer, sheet_name='T3 - Ratios (Consolidado)', index=False)
                pd.DataFrame(t4).to_excel(writer, sheet_name='T4 - Distorsión', index=False)
                pd.DataFrame(t5).to_excel(writer, sheet_name='T5 - Evolución Temporal', index=False)
                pd.DataFrame(t6).to_excel(writer, sheet_name='T6 - Tendencias', index=False)
                # NUEVA HOJA T7
                pd.DataFrame(t7).to_excel(writer, sheet_name='T7 - Anomalías Detectadas', index=False)

            self.log(f"✅ FINALIZADO. Guardado en: {path}")
            messagebox.showinfo("Fin", "Proceso completado con éxito.")
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = BatchDataExtractor(root)
    root.mainloop()