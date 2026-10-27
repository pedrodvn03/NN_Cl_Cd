"""
03_train_speed.py - Entrenamiento XGBoost incluyendo Velocidad como feature
Uso:
    python scripts/03_train_speed.py --track 01 --session all
    python scripts/03_train_speed.py --track 01 --session S01
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


class AeroSurrogateWithSpeed:
    def __init__(self, include_wings: bool = False, use_gpu: bool = True):
        self.include_wings = include_wings
        # Features: Alturas + Velocidad
        self.feature_names = ['RH_Front_Avg', 'RH_Rear_Avg', 'Speed_ms']
        if self.include_wings:
            self.feature_names.extend(['fw_angle_deg', 'rw_angle_deg'])

        gpu_kwargs = {
            'tree_method': 'hist',
            'device': 'cpu'
        } if use_gpu else {}

        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            **gpu_kwargs
        )

    def fit_from_clean(self, df_clean: pd.DataFrame, use_binning: bool = True):
        df_ready = df_clean.copy()

        # Calcular Speed_ms si no existe
        if 'Speed_ms' not in df_ready.columns and 'GPS_Speed' in df_ready.columns:
            df_ready['Speed_ms'] = df_ready['GPS_Speed'] / 3.6

        # Verificar qué columnas existen realmente
        actual_feature_names = [f for f in self.feature_names if f in df_ready.columns]
        cols_to_use = actual_feature_names + ['Cl_Filtered']

        print(f"[i] Features a usar: {actual_feature_names}")

        df_ready = df_ready[cols_to_use].replace([np.inf, -np.inf], np.nan).dropna()

        # Filtrado físico
        df_ready = df_ready[
            (df_ready['Cl_Filtered'] > 0.2) &
            (df_ready['Cl_Filtered'] < 1.5) &
            (df_ready['RH_Front_Avg'] > 20) &
            (df_ready['RH_Front_Avg'] < 45) &
            (df_ready['RH_Rear_Avg'] > 20) &
            (df_ready['RH_Rear_Avg'] < 55)
        ]

        if use_binning:
            df_ready['RH_F_bin'] = (df_ready['RH_Front_Avg'] / 1.0).round() * 1.0
            df_ready['RH_R_bin'] = (df_ready['RH_Rear_Avg'] / 1.0).round() * 1.0

            # Binning de velocidad: 5 m/s (~18 km/h)
            if 'Speed_ms' in df_ready.columns:
                df_ready['Speed_bin'] = (df_ready['Speed_ms'] / 5.0).round() * 5.0

            group_cols = ['RH_F_bin', 'RH_R_bin']
            if 'Speed_ms' in df_ready.columns:
                group_cols.append('Speed_bin')
            if self.include_wings:
                for wing_col in ['fw_angle_deg', 'rw_angle_deg']:
                    if wing_col in df_ready.columns and df_ready[wing_col].nunique() > 1:
                        group_cols.append(wing_col)

            # Incluir alas en el agg directamente
            agg_dict = {'Cl_Filtered': ['mean', 'count']}
            for wing_col in ['fw_angle_deg', 'rw_angle_deg']:
                if wing_col in group_cols:
                    agg_dict[wing_col] = 'first'

            df_grouped = df_ready.groupby(group_cols, as_index=False).agg(agg_dict)

            # Aplanar nombres de columnas multi-índice
            df_grouped.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                  for col in df_grouped.columns.values]

            # Renombrar Cl_Filtered_mean a Cl_Filtered
            if 'Cl_Filtered_mean' in df_grouped.columns:
                df_grouped = df_grouped.rename(columns={'Cl_Filtered_mean': 'Cl_Filtered'})

            # Renombrar columnas de alas (fw_angle_deg_first -> fw_angle_deg)
            for wing_col in ['fw_angle_deg', 'rw_angle_deg']:
                if f'{wing_col}_first' in df_grouped.columns:
                    df_grouped = df_grouped.rename(columns={f'{wing_col}_first': wing_col})

            # Solo celdas con al menos 5 muestras
            df_grouped = df_grouped[df_grouped['Cl_Filtered_count'] >= 5]

            # Restaurar nombres originales
            df_grouped['RH_Front_Avg'] = df_grouped['RH_F_bin']
            df_grouped['RH_Rear_Avg'] = df_grouped['RH_R_bin']
            if 'Speed_ms' in df_ready.columns:
                df_grouped['Speed_ms'] = df_grouped['Speed_bin']

            # Construir lista final de columnas
            final_cols = ['RH_Front_Avg', 'RH_Rear_Avg', 'Cl_Filtered']
            if 'Speed_ms' in df_ready.columns:
                final_cols.append('Speed_ms')
            for wing_col in ['fw_angle_deg', 'rw_angle_deg']:
                if wing_col in group_cols:
                    final_cols.append(wing_col)

            df_ready = df_grouped[[c for c in final_cols if c in df_grouped.columns]]

        print(f"[i] Dataset final: {len(df_ready)} celdas")
        print(f"[i] Rango Cl: {df_ready['Cl_Filtered'].min():.3f} a {df_ready['Cl_Filtered'].max():.3f}")
        print(f"[i] Rango RH_Front: {df_ready['RH_Front_Avg'].min():.1f} a {df_ready['RH_Front_Avg'].max():.1f}")

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
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # PDP con velocidad incluida
        features_to_plot = ['RH_Front_Avg', 'Speed_ms']
        PartialDependenceDisplay.from_estimator(
            self.model,
            X_test,
            features=features_to_plot,
            ax=axes[0]
        )
        axes[0].set_title("PDP: $C_L$ vs Altura Delantera y Velocidad")

        # Importancia
        actual_feature_names = [f for f in self.feature_names if f in X_test.columns]
        importances = self.model.feature_importances_
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

        parts = file_path.name.split('_')
        if len(parts) >= 3 and metadata_df is not None:
            file_track_id = parts[0]
            file_session_id = parts[2]

            match = metadata_df[
                (metadata_df['track_id'] == file_track_id) &
                (metadata_df['session_id'] == file_session_id)
            ]

            if not match.empty:
                df['fw_angle_deg'] = match['fw_angle_deg'].values[0]
                df['rw_angle_deg'] = match['rw_angle_deg'].values[0]
                print(f"  [+] Alas vinculadas a {file_path.name}: FW={match['fw_angle_deg'].values[0]}°, RW={match['rw_angle_deg'].values[0]}°")

        data_frames.append(df)

    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento XGBoost con Velocidad")
    parser.add_argument("--track", type=str, default="01", help="ID de pista")
    parser.add_argument("--session", type=str, default="all", help="Sesión")
    args = parser.parse_args()

    print(f"\n=== Ejecutando XGBoost + Speed | Track: '{args.track}' | Session: '{args.session}' ===")

    df_clean = load_dataset(args.track, args.session)

    has_wings = 'fw_angle_deg' in df_clean.columns and 'rw_angle_deg' in df_clean.columns
    surrogate = AeroSurrogateWithSpeed(include_wings=has_wings)

    X_test, y_test = surrogate.fit_from_clean(df_clean)

    model_filename = f"{args.track}_{args.session}_xgb_speed.json"
    model_save_path = config.MODELS_DIR / model_filename
    surrogate.model.save_model(model_save_path)
    print(f"[->] Modelo guardado en: {model_save_path}")

    figure_filename = f"{args.track}_{args.session}_pdp_speed.png"
    figure_save_path = config.FIGURES_DIR / figure_filename
    surrogate.inspect_aero_stall(X_test, output_path=figure_save_path)


if __name__ == "__main__":
    main()