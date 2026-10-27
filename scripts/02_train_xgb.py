"""
02_train_xgb.py - Entrenamiento del Modelo XGBoost por Pista / Sesión
Uso:
    python scripts/02_train_xgb.py --track 02 --session S01
    python scripts/02_train_xgb.py --track 02 --session all
    python scripts/02_train_xgb.py --track all --session all
"""
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split

import config


class AeroSurrogateClean:
    def __init__(self, include_wings: bool = False, use_gpu: bool = True):
        self.include_wings = include_wings
        # Features: Solo alturas por defecto
        self.feature_names = ['RH_Front_Avg', 'RH_Rear_Avg']
        if self.include_wings:
            self.feature_names.extend(['fw_angle_deg', 'rw_angle_deg'])

        gpu_kwargs = {
            'tree_method': 'hist',
            'device': 'cpu'
        } if use_gpu else {}

        self.model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.5,
            min_child_weight=2,
            random_state=42,
            **gpu_kwargs
        )

    def fit_from_clean(self, df_clean: pd.DataFrame, use_binning: bool = True):
        df_ready = df_clean.copy()
    
        actual_feature_names = [f for f in self.feature_names if f in df_ready.columns]
        cols_to_use = actual_feature_names + ['Cl_Filtered']
        
        print(f"[i] Features a usar: {actual_feature_names}")
        print(f"[i] Dataset crudo: {len(df_ready)} muestras")
        
        # Debug: Ver valores únicos antes de binning
        print(f"\n{'='*60}")
        print(f"[DEBUG] ANÁLISIS DE DATOS CRUDOS")
        print(f"{'='*60}")
        print(f"RH_Front_Avg - únicos: {df_ready['RH_Front_Avg'].nunique()}")
        print(f"  Rango: [{df_ready['RH_Front_Avg'].min():.4f}, {df_ready['RH_Front_Avg'].max():.4f}]")
        print(f"  Primeros 10 valores: {sorted(df_ready['RH_Front_Avg'].unique())[:10]}")
        
        print(f"RH_Rear_Avg - únicos: {df_ready['RH_Rear_Avg'].nunique()}")
        print(f"  Rango: [{df_ready['RH_Rear_Avg'].min():.4f}, {df_ready['RH_Rear_Avg'].max():.4f}]")
        
        df_ready = df_ready[cols_to_use].replace([np.inf, -np.inf], np.nan).dropna()

        # Filtrar valores físicos
        df_ready = df_ready[
            (df_ready['Cl_Filtered'] > 0.0) & 
            (df_ready['Cl_Filtered'] < 2.0) &
            (df_ready['RH_Front_Avg'] > 5) & 
            (df_ready['RH_Front_Avg'] < 60)
        ]
        
        print(f"\n[i] Tras filtros físicos: {len(df_ready)} muestras")

        if use_binning:
            print(f"\n{'='*60}")
            print(f"[DEBUG] PROCESO DE BINNING")
            print(f"{'='*60}")
            
            # Binning con redondeo explícito
            df_ready['RH_F_bin'] = (df_ready['RH_Front_Avg'] / 1.0).round(0) * 1.0
            df_ready['RH_R_bin'] = (df_ready['RH_Rear_Avg'] / 1.0).round(0) * 1.0
            
            # Debug: Ver bins únicos
            print(f"RH_F_bin - únicos: {df_ready['RH_F_bin'].nunique()}")
            print(f"  Valores únicos: {sorted(df_ready['RH_F_bin'].unique())}")
            print(f"RH_R_bin - únicos: {df_ready['RH_R_bin'].nunique()}")
            print(f"  Valores únicos: {sorted(df_ready['RH_R_bin'].unique())}")

            group_cols = ['RH_F_bin', 'RH_R_bin']
            
            if 'fw_angle_deg' in df_ready.columns and 'rw_angle_deg' in df_ready.columns:
                group_cols.extend(['fw_angle_deg', 'rw_angle_deg'])
                print(f"[i] Incluyendo ángulos de alas en el groupby")
            else:
                print(f"[!] Ángulos de alas no encontrados, usando solo alturas")

            # Debug: Combinaciones únicas antes de groupby
            n_unique_combinations = df_ready[group_cols].drop_duplicates().shape[0]
            print(f"\n[DEBUG] Combinaciones únicas de {group_cols}: {n_unique_combinations}")
            
            # Debug: Distribución de muestras por bin
            bin_counts = df_ready.groupby(group_cols).size()
            print(f"\n[DEBUG] DISTRIBUCIÓN DE MUESTRAS POR BIN:")
            print(f"  Bins con 1 muestra: {(bin_counts == 1).sum()}")
            print(f"  Bins con 2-5 muestras: {((bin_counts >= 2) & (bin_counts <= 5)).sum()}")
            print(f"  Bins con 6-10 muestras: {((bin_counts >= 6) & (bin_counts <= 10)).sum()}")
            print(f"  Bins con >10 muestras: {(bin_counts > 10).sum()}")
            print(f"  Total de bins: {len(bin_counts)}")
            
            # Debug: Mostrar bins con más muestras
            print(f"\n[DEBUG] TOP 10 BINS CON MÁS MUESTRAS:")
            top_bins = bin_counts.nlargest(10)
            for bin_key, count in top_bins.items():
                print(f"  {bin_key}: {count} muestras")
            
            # Debug: Mostrar bins con 1 muestra (primeros 10)
            print(f"\n[DEBUG] PRIMEROS 10 BINS CON 1 SOLA MUESTRA:")
            single_sample_bins = bin_counts[bin_counts == 1].head(10)
            for bin_key, count in single_sample_bins.items():
                print(f"  {bin_key}: {count} muestra")

            # AGRUPAMIENTO
            df_grouped = df_ready.groupby(group_cols, as_index=False).agg({
                'Cl_Filtered': ['mean', 'count']
            })
            
            # Aplanar columnas multi-index
            df_grouped.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                for col in df_grouped.columns]
            
            # Renombrar
            rename_map = {
                'RH_F_bin': 'RH_Front_Avg',
                'RH_R_bin': 'RH_Rear_Avg',
                'Cl_Filtered_mean': 'Cl_Filtered',
                'Cl_Filtered_count': 'bin_count'
            }
            df_grouped = df_grouped.rename(columns=rename_map)
            
            print(f"\n[DEBUG] TRAS GROUPBY: {len(df_grouped)} bins")
            
            # Filtrar bins con pocas muestras
            min_samples = 3
            before = len(df_grouped)
            df_grouped = df_grouped[df_grouped['bin_count'] >= min_samples]
            after = len(df_grouped)
            print(f"[DEBUG] Tras filtrar bins con <{min_samples} muestras: {before} → {after}")
            
            # Seleccionar columnas finales
            output_cols = [c for c in cols_to_use if c in df_grouped.columns]
            df_ready = df_grouped[output_cols].copy()

        print(f"\n{'='*60}")
        print(f"[i] Dataset final: {len(df_ready)} muestras")
        print(f"[i] Rango Cl: {df_ready['Cl_Filtered'].min():.3f} a {df_ready['Cl_Filtered'].max():.3f}")
        print(f"[i] Rango RH_Front: {df_ready['RH_Front_Avg'].min():.1f} a {df_ready['RH_Front_Avg'].max():.1f}")
        print(f"{'='*60}\n")

        X = df_ready[actual_feature_names]
        y = df_ready['Cl_Filtered']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        r2_score = self.model.score(X_test, y_test)
        print(f"[+] Modelo entrenado con {len(df_ready)} celdas aerodinámicas. R² en Test: {r2_score:.4f}")
        
        print("[i] Importancia de features:")
        for name, imp in zip(actual_feature_names, self.model.feature_importances_):
            print(f"   {name}: {imp:.4f}")
        
        return X_test, y_test

    def inspect_aero_stall(self, X_test: pd.DataFrame, output_path: Path = None):
        """Método que faltaba - genera las gráficas PDP e importancia"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Gráfica PDP
        PartialDependenceDisplay.from_estimator(
            self.model,
            X_test,
            features=['RH_Front_Avg', ('RH_Front_Avg', 'RH_Rear_Avg')],
            ax=axes[0]
        )
        axes[0].set_title("PDP: $C_L$ Puro vs Altura Delantera y Trasera")

        # Importancia de features
        importances = self.model.feature_importances_
        actual_feature_names = [f for f in self.feature_names if f in X_test.columns]
        axes[1].barh(actual_feature_names, importances, color='steelblue')
        axes[1].set_title("Importancia Relativa de Variables")
        axes[1].invert_yaxis()

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300)
            print(f"[->] Gráfica PDP guardada en: {output_path}")
        else:
            plt.show()

            
def load_dataset(track_filter: str, session_filter: str) -> pd.DataFrame:
    processed_dir = config.PROCESSED_DATA_DIR
    metadata_df = None

    if config.METADATA_PATH.exists():
        metadata_df = pd.read_csv(config.METADATA_PATH)
        metadata_df.columns = metadata_df.columns.str.strip()
        # Normalizar ID de pista a 2 dígitos (ejemplo: '01')
        if 'track_id' in metadata_df.columns:
            metadata_df['track_id'] = metadata_df['track_id'].astype(str).str.zfill(2)
        print(f"[+] Archivo de metadatos cargado correctamente desde {config.METADATA_PATH}")

    files_to_load = []
    for file_path in processed_dir.rglob("*_cleaned.csv"):
        filename = file_path.name

        if track_filter != "all" and track_filter not in filename and track_filter not in str(file_path.parent):
            continue
        if session_filter != "all" and f"_{session_filter}_" not in filename and f"_{session_filter}." not in filename:
            continue

        files_to_load.append(file_path)

    if not files_to_load:
        raise FileNotFoundError(f"No se encontraron archivos procesados para Track='{track_filter}' y Session='{session_filter}'")

    data_frames = []
    for file_path in files_to_load:
        df = pd.read_csv(file_path)

        # Extraer Track_ID y Session_ID del nombre (ej: 01_Monza_S01_cleaned.csv)
        parts = file_path.name.split('_')
        if len(parts) >= 3 and metadata_df is not None:
            file_track_id = parts[0]   # '01'
            file_session_id = parts[2] # 'S01'

            match = metadata_df[
                (metadata_df['track_id'] == file_track_id) & 
                (metadata_df['session_id'] == file_session_id)
            ]

            if not match.empty:
                df['fw_angle_deg'] = match['fw_angle_deg'].values[0]
                df['rw_angle_deg'] = match['rw_angle_deg'].values[0]
                print(f"  [+] Alas vinculadas a {file_path.name}: FW={match['fw_angle_deg'].values[0]}°, RW={match['rw_angle_deg'].values[0]}°")
            else:
                print(f"  [!] Sin coincidencia en metadatos para Track={file_track_id}, Session={file_session_id}")

        data_frames.append(df)

    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento del Modelo XGBoost por Pista/Sesión")
    parser.add_argument("--track", type=str, default="02", help="ID o nombre de la pista (ej. '01', '02', 'all')")
    parser.add_argument("--session", type=str, default="all", help="Número de sesión (ej. 'S01', 'S02', 'all')")
    args = parser.parse_args()

    print(f"\n=== Ejecutando XGBoost | Track: '{args.track}' | Session: '{args.session}' ===")

    df_clean = load_dataset(args.track, args.session)

    has_wings = 'fw_angle_deg' in df_clean.columns and 'rw_angle_deg' in df_clean.columns
    surrogate = AeroSurrogateClean(include_wings=False)

    X_test, y_test = surrogate.fit_from_clean(df_clean)

    # Guardar modelo
    model_filename = f"{args.track}_{args.session}_xgb.json"
    model_save_path = config.MODELS_DIR / model_filename
    surrogate.model.save_model(model_save_path)
    print(f"[->] Modelo guardado en: {model_save_path}")

    # Guardar gráfica PDP
    figure_filename = f"{args.track}_{args.session}_pdp.png"
    figure_save_path = config.FIGURES_DIR / figure_filename
    surrogate.inspect_aero_stall(X_test, output_path=figure_save_path)


if __name__ == "__main__":
    main()