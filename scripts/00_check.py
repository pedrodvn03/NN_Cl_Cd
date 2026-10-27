import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/processed/01_Monza/01_Monza_S01_cleaned.csv')

# Filtros MUY estrictos para ver la física real
df_clean = df[
    (df['Cl_Filtered'] > 0.3) & (df['Cl_Filtered'] < 1.4) &
    (df['RH_Front_Avg'] > 22) & (df['RH_Front_Avg'] < 40) &  # Eliminamos alturas irreales (>40mm)
    (df['RH_Rear_Avg'] > 25) & (df['RH_Rear_Avg'] < 50) &
    (df['GPS_Speed'] > 180)  # Solo velocidad muy alta
]

print(f"Muestras ultra-filtradas: {len(df_clean)}")
print(f"Std de Cl: {df_clean['Cl_Filtered'].std():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].scatter(df_clean['RH_Front_Avg'], df_clean['Cl_Filtered'], alpha=0.3, s=10)
axes[0].set_xlabel('RH_Front_Avg (mm)')
axes[0].set_ylabel('Cl_Filtered')
axes[0].set_title('Cl vs RH_Front (Filtrado Estricto)')
axes[0].grid(True)

axes[1].scatter(df_clean['RH_Rear_Avg'], df_clean['Cl_Filtered'], alpha=0.3, s=10, color='orange')
axes[1].set_xlabel('RH_Rear_Avg (mm)')
axes[1].set_ylabel('Cl_Filtered')
axes[1].set_title('Cl vs RH_Rear (Filtrado Estricto)')
axes[1].grid(True)

plt.tight_layout()
plt.show()