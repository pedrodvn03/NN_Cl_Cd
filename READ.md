# NN_C1_Cd - Modelo Surrogate Aerodinámico

Predicción de Cd (coeficiente de arrastre) usando XGBoost con datos de F1.

## Estructura
- `data/raw/`: datos crudos (no versionados)
- `data/processed/`: datos limpios (no versionados)
- `scripts/`: scripts ejecutables
- `outputs/models/`: modelos entrenados (no versionados)

## Uso
```bash
python scripts/01_process_data.py
python scripts/02_train_xgb.py