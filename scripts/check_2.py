import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carga tu archivo cleaned
df = pd.read_csv('data/processed/01_Monza/01_Monza_S01_cleaned.csv')

# Verificar correlación entre Cl y Velocidad
correlation = df['Cl_Filtered'].corr(df['GPS_Speed'])
print(f"Correlación entre Cl_Filtered y GPS_Speed: {correlation:.4f}")

# Gráfica de dispersión
plt.figure(figsize=(10, 5))
plt.scatter(df['GPS_Speed'], df['Cl_Filtered'], alpha=0.3, s=10)
plt.xlabel('GPS Speed (km/h)')
plt.ylabel('Cl_Filtered')
plt.title('Dependencia de Cl con la Velocidad (Si hay tendencia, hay problema en Etapa 1)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Estadísticas básicas
print(f"\nRango de Cl: {df['Cl_Filtered'].min():.3f} a {df['Cl_Filtered'].max():.3f}")
print(f"Media de Cl: {df['Cl_Filtered'].mean():.3f}")
print(f"Número de muestras: {len(df)}")