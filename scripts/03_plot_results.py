"""
03_plot_results.py - Visualización de Telemetría Aerodinámica por Vuelta
Uso:
    python scripts/03_plot_results.py --track 02 --session S01 --lap 8
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import config


def plot_lap_telemetry(track_id: str, session_id: str, lap_number: int):
    processed_dir = config.PROCESSED_DATA_DIR
    target_files = []

    for file_path in processed_dir.rglob("*_cleaned.csv"):
        filename = file_path.name
        if track_id in filename and session_id in filename:
            target_files.append(file_path)

    if not target_files:
        raise FileNotFoundError(f"No se encontró archivo limpio para Track='{track_id}' y Session='{session_id}' en {processed_dir}")

    clean_file = target_files[0]
    df = pd.read_csv(clean_file)

    lap_df = df[df['LapNumber'] == lap_number].sort_values('LapDistance')

    if lap_df.empty:
        available_laps = sorted(df['LapNumber'].unique())
        raise ValueError(f"La vuelta {lap_number} no existe en el archivo. Vueltas disponibles: {available_laps}")

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Eje Izquierdo: C_L
    ax1.plot(lap_df['LapDistance'], lap_df['Cl_Filtered'], color='navy', linewidth=2, label='$C_L$ Filtered')
    ax1.set_xlabel('Distancia de Vuelta (m)', fontsize=12)
    ax1.set_ylabel('Coeficiente de Carga ($C_L$)', color='navy', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Eje Derecho: Ride Heights
    ax2 = ax1.twinx()
    ax2.plot(lap_df['LapDistance'], lap_df['RH_Front_Avg'], color='crimson', linestyle='--', label='RH Front (mm)')
    ax2.plot(lap_df['LapDistance'], lap_df['RH_Rear_Avg'], color='darkorange', linestyle='--', label='RH Rear (mm)')
    ax2.set_ylabel('Altura de Suspensión (mm)', color='crimson', fontsize=12)

    plt.title(f'Mapa Aerodinámico Vuelta {lap_number} - {clean_file.stem}', fontsize=14)
    fig.tight_layout()

    out_file = config.FIGURES_DIR / f"{track_id}_{session_id}_lap_{lap_number}_aeromap.png"
    plt.savefig(out_file, dpi=300)
    print(f"[->] Gráfica guardada en: {out_file}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graficar Telemetría Aerodinámica por Vuelta")
    parser.add_argument("--track", type=str, default="02", help="ID de la pista (ej. '02')")
    parser.add_argument("--session", type=str, default="S01", help="Identificador de sesión (ej. 'S01')")
    parser.add_argument("--lap", type=int, default=8, help="Número de vuelta a graficar (ej. 8)")
    args = parser.parse_args()

    plot_lap_telemetry(args.track, args.session, args.lap)