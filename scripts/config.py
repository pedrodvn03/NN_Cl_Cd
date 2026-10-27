"""
config.py - Módulo de rutas y configuración centralizada
"""
from pathlib import Path

# Ruta raíz del proyecto (NN_Cl_Cd/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Directorios de datos
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_PATH = DATA_DIR / "sessions_metadata.csv"

# Directorios de salida
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"

# Crear estructura de carpetas automáticamente si no existe
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)