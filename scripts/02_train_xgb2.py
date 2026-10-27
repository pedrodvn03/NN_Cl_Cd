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
        # Features: Solo alturas por defecto (ideal para AeroMap estático)
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

    def fit_from_clean(self, df_clean: pd.DataFrame, use_binning: bool = False):
        """
        Entrena el modelo XGBoost. 
        use_binning=False entrena con todos los datos puros para maximizar el R2.
        """
        df_ready = df_clean.copy()
        
        actual_feature_names = [f for f in self.feature_names if f in df_ready.columns]
        cols_to_use = actual_feature_names + ['Cl_Filtered']
        
        print(f"[i] Features a usar: {actual_feature_names}")
        
        df_ready = df_ready[cols_to_use].replace([np.inf, -np.inf], np.nan).dropna()

        # Filtrar valores físicos correspondientes a los filtros de 01_process_data.py
        df_ready = df_ready[
            (df_ready['Cl_Filtered'] > 0.0) & 
            (df_ready['Cl_Filtered'] < 2.0) &
            (df_ready['RH_Front_Avg'] > 5) & 
            (df_ready['RH_Front_Avg'] < 60)
        ]

        if use_binning:
            df_ready['RH_F_bin'] = (df_ready['RH_Front_Avg'] / 1.0).round() * 1.0
            df_ready['RH_R_bin'] = (df_ready['RH_Rear_Avg'] / 1.0).round() * 1.0

            group_cols = ['RH_F_bin', 'RH_R_bin']
            
            if 'fw_angle_deg' in df_ready.columns and 'rw_angle_deg' in df_ready.columns:
                group_cols.extend(['fw_angle_deg', 'rw_angle_deg'])
                print(f"[i] Incluyendo ángulos de alas en el groupby")
            else:
                print(f"[!] Ángulos de alas no encontrados, usando solo alturas")

            df_grouped = df_ready.groupby(group_cols, as_index=False).agg({
                'Cl_Filtered': 'mean'
            })
            
            df_grouped['RH_Front_Avg'] = df_grouped['RH_F_bin']
            df_grouped['RH_Rear_Avg'] = df_grouped['RH_R_bin']
            
            df_ready = df_grouped[[c for c in cols_to_use if c in df_grouped.columns]]

        print(f"[i] Dataset final: {len(df_ready)} muestras")
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

    def generate_static_aeromap(self, rh_f_range=(5, 45), rh_r_range=(10, 50), step=1.0, output_csv: Path = None):
        """Genera un AeroMap estático de Cl sobre una cuadrícula perfecta de alturas."""
        # Crear cuadrícula (bins perfectos)
        rh_f_vals = np.arange(rh_f_range[0], rh_f_range[1] + step, step)
        rh_r_vals = np.arange(rh_r_range[0], rh_r_range[1] + step, step)
        
        grid_f, grid_r = np.meshgrid(rh_f_vals, rh_r_vals)
        df_map = pd.DataFrame({
            'RH_Front_Avg': grid_f.ravel(),
            'RH_Rear_Avg': grid_r.ravel()
        })
        
        # Predecir el Cl para cada punto del mapa usando el modelo entrenado
        df_map['Cl_Predicted'] = self.model.predict(df_map[['RH_Front_Avg', 'RH_Rear_Avg']])
        
        if output_csv:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df_map.to_csv(output_csv, index=False)
            print(f"[->] AeroMap estático guardado en: {output_csv}")
            
        return df_map


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

    # Excluimos explícitamente las alas porque hay poca variabilidad de datos
    surrogate = AeroSurrogateClean(include_wings=False)

    # Entrenar sin binning (use_binning=False por defecto ahora)
    X_test, y_test = surrogate.fit_from_clean(df_clean, use_binning=False)

    # Guardar modelo
    model_filename = f"{args.track}_{args.session}_xgb.json"
    model_save_path = config.MODELS_DIR / model_filename
    surrogate.model.save_model(model_save_path)
    print(f"[->] Modelo guardado en: {model_save_path}")

    # Guardar gráfica PDP
    figure_filename = f"{args.track}_{args.session}_pdp.png"
    figure_save_path = config.FIGURES_DIR / figure_filename
    surrogate.inspect_aero_stall(X_test, output_path=figure_save_path)
    
    # Generar y guardar el AeroMap Estático limpio
    aeromap_filename = f"{args.track}_{args.session}_aeromap.csv"
    # Lo guardamos en una carpeta 'tables' junto a donde guardas tus figuras/modelos
    aeromap_save_path = config.MODELS_DIR.parent / "tables" / aeromap_filename
    surrogate.generate_static_aeromap(output_csv=aeromap_save_path)


if __name__ == "__main__":
    main()