"""
Schnelles Regressions-Scatterplot für M2
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Lade Daten
data_path = Path('data/processed/conflict_market_features.csv')
critical_cols = ['volatility_change', 'total_deaths', 'pre_volatility', 
                 'lethality_ratio', 'region', 'severity_class', 'year']
df = pd.read_csv(data_path, usecols=critical_cols)
df_clean = df.dropna(subset=critical_cols)

print("Erstelle Scatterplots...")

# Erstelle klassisches Regressions-Scatterplot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('M2 Lineare Regression: Klassische Visualisierung', 
             fontsize=14, fontweight='bold')

# Sample für Übersichtlichkeit
np.random.seed(42)
sample_idx = np.random.choice(len(df_clean), size=min(5000, len(df_clean)), replace=False)
df_sample = df_clean.iloc[sample_idx]

# ---- PLOT A: Pre_volatility (X) vs volatility_change (Y) ----
ax = axes[0]
ax.scatter(df_sample['pre_volatility'], df_sample['volatility_change'], 
          alpha=0.3, s=20, color='#3498db', edgecolor='none', label='Daten')

# Regressionslinie (lineare Approximation)
z = np.polyfit(df_clean['pre_volatility'], df_clean['volatility_change'], 1)
p = np.poly1d(z)
pre_vol_range = np.linspace(df_clean['pre_volatility'].min(), 
                             df_clean['pre_volatility'].max(), 100)
ax.plot(pre_vol_range, p(pre_vol_range), 'r-', linewidth=3, label='Regressionslinie')

ax.set_xlabel('Pre-Volatilität (X)', fontsize=12, fontweight='bold')
ax.set_ylabel('Volatility Change (Y)', fontsize=12, fontweight='bold')
ax.set_title('Mean Reversion: Höhere Baseline-Vol → Kleinere Änderung', 
            fontsize=11, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

# Korrelation
corr1 = df_clean['pre_volatility'].corr(df_clean['volatility_change'])
ax.text(0.05, 0.95, f'Korrelation: {corr1:.4f}\nR²: {corr1**2:.6f}', 
       transform=ax.transAxes, fontsize=10, verticalalignment='top',
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# ---- PLOT B: Total_deaths (X) vs volatility_change (Y) ----
ax = axes[1]
ax.scatter(df_sample['total_deaths'], df_sample['volatility_change'], 
          alpha=0.3, s=20, color='#2ecc71', edgecolor='none', label='Daten')

# Regressionslinie
z2 = np.polyfit(df_clean['total_deaths'], df_clean['volatility_change'], 1)
p2 = np.poly1d(z2)
deaths_range = np.linspace(0, df_clean['total_deaths'].quantile(0.95), 100)
ax.plot(deaths_range, p2(deaths_range), 'r-', linewidth=3, label='Regressionslinie')

ax.set_xlabel('Total Deaths (X)', fontsize=12, fontweight='bold')
ax.set_ylabel('Volatility Change (Y)', fontsize=12, fontweight='bold')
ax.set_title('Effekt von Todesfällen auf Volatilitätsveränderung', 
            fontsize=11, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

# Korrelation
corr2 = df_clean['total_deaths'].corr(df_clean['volatility_change'])
ax.text(0.05, 0.95, f'Korrelation: {corr2:.4f}\nR²: {corr2**2:.6f}', 
       transform=ax.transAxes, fontsize=10, verticalalignment='top',
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('results/m2_regression_scatterplot.png', dpi=300, bbox_inches='tight')
print("✓ Gespeichert: results/m2_regression_scatterplot.png")
plt.close()

print("\nScatterplot erfolgreich erstellt!")
