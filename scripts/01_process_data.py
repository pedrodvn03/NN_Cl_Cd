"""
01_process_data.py - Limpieza, estandarización y filtrado aerodinámico
Uso:
    python scripts/01_process_data.py --track 02 --session S01
    python scripts/01_process_data.py --track all --session all
"""
import argparse
import csv
import pandas as pd
import numpy as np
import scipy.signal as signal
from pathlib import Path

import config


def create_intermediate_csv(raw_csv_path: Path, intermediate_csv_path: Path):
    """
    ETAPA 1:
    - Lee el CSV de AiM/MoTeC manejando metadatos.
    - Calcula la frecuencia de muestreo REAL a partir del vector 'Time'.
    - Estandariza nombres de columnas y parsea marcas de baliza para vueltas.
    - Asigna LapDistance de forma segura sin propagación de NaNs.
    """
    print(f"\n--- ETAPA 1: Parseando {raw_csv_path.name} ---")

    beacon_markers = []
    headers = []
    data_rows = []
    state = "METADATA"

    if not raw_csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {raw_csv_path}")

    with open(raw_csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            row_clean = [col.strip().replace('"', '').replace("'", '') for col in row]
            if not row_clean or not row_clean[0]:
                continue

            first_col = row_clean[0]

            if state == "METADATA":
                # Buscar Beacon Markers
                if first_col == "Beacon Markers":
                    for val_str in row_clean[1:]:
                        try:
                            val = float(val_str)
                            if val > 0.0:
                                beacon_markers.append(val)
                        except ValueError:
                            pass
                # Buscar la cabecera real (fila donde primera columna es "Time")
                elif first_col == "Time" and len(row_clean) >= 20:
                    headers = row_clean
                    state = "UNITS"
                    print(f"  [i] Cabecera detectada con {len(headers)} columnas")

            elif state == "UNITS":
                # Saltar la fila de unidades
                state = "DATA"

            elif state == "DATA":
                # Solo leer filas que tengan el mismo número de columnas que la cabecera
                if len(row_clean) == len(headers):
                    data_rows.append(row_clean)

    if not headers:
        raise ValueError(f"No se encontró la cabecera 'Time' en {raw_csv_path.name}.")

    df = pd.DataFrame(data_rows, columns=headers)

    # Renombrado de columnas
    column_renames = {
        'FLLoad': 'LFLoad',
        'RFLoad': 'FRLoad',
        'LRLoad': 'LRLoad',
        'RRLoad': 'RRLoad',
        'RifeHeightR': 'RideHeightR',
        'RideHeightR': 'RideHeightR',
        'RideHeightF': 'RideHeightF',
        'GPS Speed': 'GPS_Speed',
        'GPS LatAcc': 'GPS_LatAcc',
        'GPS LonAcc': 'GPS_LonAcc',
        'GPS Gyro': 'GPS_Gyro',
        'AirDensity': 'Air_Density',
        'Air Temp': 'AirTemp',
        'Track Temp': 'TrackTemp'
    }
    df = df.rename(columns=column_renames)

    # Convertir a numérico
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- 1. CÁLCULO DINÁMICO DE SAMPLE RATE Y DT ---
    if 'Time' in df.columns and len(df['Time']) > 1:
        dt_series = df['Time'].diff().dropna()
        dt = dt_series.median()
        if dt > 0:
            sample_rate = round(1.0 / dt, 2)
        else:
            sample_rate = 20.0
            dt = 1.0 / sample_rate
    else:
        sample_rate = 20.0
        dt = 1.0 / sample_rate

    print(f"  [+] Frecuencia de muestreo detectada: {sample_rate} Hz (dt = {dt:.4f} s)")

    # --- 2. ASIGNACIÓN DE VUELTAS ---
    beacon_markers = sorted(beacon_markers)
    if beacon_markers and 'Time' in df.columns and not df['Time'].empty:
        lap_bins = [0.0] + beacon_markers
        if df['Time'].max() > lap_bins[-1]:
            lap_bins.append(df['Time'].max() + 1.0)

        lap_labels = list(range(1, len(lap_bins)))
        df['LapNumber'] = pd.cut(df['Time'], bins=lap_bins, labels=lap_labels, include_lowest=True).astype(int)
        print(f"  [+] Vueltas detectadas: {len(lap_labels)}")
    else:
        df['LapNumber'] = 1
        print("  [!] Sin marcas de baliza. Asignando Vuelta 1 a todo el dataset.")

    # --- 3. CÁLCULO DE DISTANCIA POR VUELTA ---
    if 'GPS_Speed' in df.columns:
        speed_ms = df['GPS_Speed'].fillna(0.0) / 3.6
        df['Distance_Step'] = speed_ms * dt
        df['LapDistance'] = df.groupby('LapNumber')['Distance_Step'].cumsum()
        df.drop(columns=['Distance_Step'], inplace=True)

    # --- VALIDACIÓN DE COLUMNAS CRÍTICAS ---
    required_load_cols = ['LFLoad', 'FRLoad', 'LRLoad', 'RRLoad']
    missing_cols = [col for col in required_load_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Columnas de carga de rueda faltantes: {missing_cols}. "
            f"Columnas disponibles en el DataFrame: {df.columns.tolist()}"
        )

    intermediate_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(intermediate_csv_path, index=False)
    print(f"  [->] Guardado intermedio: {intermediate_csv_path}")

    return sample_rate, df

def process_aero_features(df_intermediate: pd.DataFrame, sample_rate_hz: float, 
                          car_mass_kg=545.0, frontal_area_m2=1.5,
                          cg_height_m=0.30, wheelbase_m=2.70):
    """
    ETAPA 2:
    - Calcula Downforce Total compensando transferencia de carga longitudinal.
    - Filtra rectas estáticas y aplica filtro pasa-bajas Butterworth.
    """
    print("--- ETAPA 2: Filtrado Aerodinámico y Separación de Rectas ---")
    df = df_intermediate.copy()

    load_cols = ['LFLoad', 'FRLoad', 'LRLoad', 'RRLoad']
    if all(col in df.columns for col in load_cols):
        f_total_N = (df['LFLoad'] + df['FRLoad'] + df['LRLoad'] + df['RRLoad']) * 9.81
    else:
        raise KeyError(f"Columnas de carga de rueda faltantes en DataFrame.")

    weight_N = car_mass_kg * 9.81
    
    # [CRÍTICO] Calcular aceleración longitudinal en m/s²
    lon_acc_ms2 = df['InlineAcc'] * 9.81  # Convertir de g a m/s²
    
    # [CRÍTICO] Restar transferencia de carga longitudinal
    # Fórmula: delta_Fz = (m * ax * h_cg) / wheelbase
    load_transfer_N = (car_mass_kg * lon_acc_ms2 * cg_height_m) / wheelbase_m
    
    # Downforce real = Carga total - Peso - Transferencia de carga inercial
    df['Downforce_Raw'] = f_total_N - weight_N - load_transfer_N

    if 'Air_Density' in df.columns and df['Air_Density'].mean() > 0.5:
        air_density = df['Air_Density']
    else:
        air_density = 1.225

    df['Speed_ms'] = df['GPS_Speed'] / 3.6
    q_dyn = 0.5 * air_density * frontal_area_m2 * (df['Speed_ms'] ** 2)
    df['Cl_Raw'] = np.where(df['Speed_ms'] > 15.0, df['Downforce_Raw'] / q_dyn, np.nan)

    df['RH_Front_Avg'] = df.get('RideHeightF', pd.Series(0, index=df.index))
    df['RH_Rear_Avg'] = df.get('RideHeightR', pd.Series(0, index=df.index))
    df['RH_Pitch_Angle'] = df['RH_Rear_Avg'] - df['RH_Front_Avg']

    gyro_col = df.get('YawRate', df.get('GPS_Gyro', pd.Series(0, index=df.index)))
    yaw_rate_rad = np.abs(np.radians(gyro_col))
    df['Turn_Radius'] = np.where(yaw_rate_rad > 0.001, df['Speed_ms'] / yaw_rate_rad, 9999.0)

    steer_col = df.get('SteerAngle', pd.Series(0, index=df.index))
    lat_acc_col = df.get('LateralAcc', df.get('GPS_LatAcc', pd.Series(0, index=df.index)))
    lon_acc_col = df.get('InlineAcc', df.get('GPS_LonAcc', pd.Series(0, index=df.index)))
    throttle_col = df.get('Throttle', pd.Series(100, index=df.index))
    brake_col = df.get('Brake', pd.Series(0, index=df.index))

    # [CRÍTICO] Filtros MUCHO más estrictos para rectas puras
    c_steer = np.abs(steer_col) < 0.4        # Era 3.0
    c_radius = df['Turn_Radius'] > 100.0    # Era 500.0
    c_acc_lat = np.abs(lat_acc_col) < 0.45   # Era 0.15
    c_acc_long = np.abs(lon_acc_col) < 0.25  # Era 0.10 (MUY estricto)
    c_throttle = throttle_col > 70.0         # Era 90.0
    c_brake = brake_col < 0.5                # Era 2.0
    c_speed = df['GPS_Speed'] > 80.0        # Era 80.0

    valid_mask = c_steer & c_radius & c_acc_lat & c_acc_long & c_throttle & c_brake & c_speed

    cutoff = 2.0
    nyquist = 0.5 * sample_rate_hz
    normal_cutoff = min(cutoff / nyquist, 0.99)
    b, a = signal.butter(4, normal_cutoff, btype='low')

    df['Cl_Filtered'] = signal.filtfilt(b, a, np.nan_to_num(df['Cl_Raw']))
    df['Downforce_Filtered'] = signal.filtfilt(b, a, np.nan_to_num(df['Downforce_Raw']))

    clean_dataset = df[valid_mask].copy()
    
    # [NUEVO] Filtrado físico estricto
    clean_dataset = clean_dataset[
        (clean_dataset['Cl_Filtered'] > 0.3) & 
        (clean_dataset['Cl_Filtered'] < 1.95) &
        (clean_dataset['RH_Front_Avg'] > 5) & 
        (clean_dataset['RH_Front_Avg'] < 45) &
        (clean_dataset['RH_Rear_Avg'] > 10) & 
        (clean_dataset['RH_Rear_Avg'] < 50)
    ]
    
    print(f"  [i] Muestras después de filtro físico: {len(clean_dataset)}")

    output_cols = [
        'LapNumber', 'LapDistance', 'Time', 'GPS_Speed',
        'RH_Front_Avg', 'RH_Rear_Avg', 'RH_Pitch_Angle',
        'Downforce_Filtered', 'Cl_Filtered'
    ]
    actual_cols = [c for c in output_cols if c in clean_dataset.columns]

    return clean_dataset[actual_cols].replace([np.inf, -np.inf], np.nan).dropna()

def main():
    parser = argparse.ArgumentParser(description="Procesamiento de Telemetría por Pista y Sesión")
    parser.add_argument("--track", type=str, default="all", help="ID o nombre de la pista (ej. '01', '02', 'all')")
    parser.add_argument("--session", type=str, default="all", help="Número de sesión (ej. 'S01', 'S02', 'all')")
    args = parser.parse_args()
    # [CORRECCIÓN 1] Solo buscar archivos *_raw.csv
    raw_files = list(config.RAW_DATA_DIR.rglob("*_raw.csv"))
    print(f"[DEBUG] Archivos *_raw.csv encontrados: {len(raw_files)}")
    for f in raw_files:
        print(f"  - {f.name}")

    if not raw_files:
        print(f"[!] No se encontraron archivos *_raw.csv en {config.RAW_DATA_DIR}")
        return

    processed_count = 0
    for raw_file in raw_files:
        filename = raw_file.name

        # [CORRECCIÓN 2] Filtrado estricto por pista
        # El nombre debe empezar con "01_" (ej: 01_Monza_S01_raw.csv)
        if args.track != "all":
            track_prefix = f"{args.track}_"
            if not filename.startswith(track_prefix):
                print(f"[DEBUG] Filtrado (track): {filename} no empieza con {track_prefix}")
                continue

        # [CORRECCIÓN 3] Filtrado por sesión
        if args.session != "all":
            session_pattern = f"_{args.session}_"
            session_pattern_alt = f"_{args.session}."
            if session_pattern not in filename and session_pattern_alt not in filename:
                print(f"[DEBUG] Filtrado (session): {filename} no contiene {session_pattern}")
                continue

        print(f"[DEBUG] Procesando: {filename}")

        folder_name = raw_file.parent.name if raw_file.parent != config.RAW_DATA_DIR else ""
        target_dir = config.PROCESSED_DATA_DIR / folder_name

        base_name = raw_file.stem.replace("_raw", "")
        parsed_path = target_dir / f"{base_name}_parsed.csv"
        cleaned_path = target_dir / f"{base_name}_cleaned.csv"

        try:
            sample_rate, df_inter = create_intermediate_csv(raw_file, parsed_path)
            df_clean = process_aero_features(df_inter, sample_rate_hz=sample_rate)
            df_clean.to_csv(cleaned_path, index=False)
            print(f"  [->] Guardado final limpio: {cleaned_path}")
            processed_count += 1
        except Exception as e:
            print(f"  [X] Error procesando {filename}: {e}")

    print(f"\n[Éxito] Finalizado. Archivos procesados: {processed_count}")
if __name__ == "__main__":
    main()