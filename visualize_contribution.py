"""
Visualisierung: Woher kommt das R²?
"""
import matplotlib.pyplot as plt
import numpy as np

# Daten
components = ['Konflikte\n(total_deaths,\nlethality)', 
              'Pre-Volatilität\n(Mean Reversion)', 
              'Year Dummies\n(Makroschocks)',
              'Region +\nSeverity']
r2_values = [0.0045, 11.35, 7.58, 0.15]
colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Woher kommt das R² = 18.93%?', fontsize=14, fontweight='bold')

# ---- PLOT 1: Stacked Bar ----
ax = axes[0]
cumsum = np.cumsum([0] + r2_values)
for i, (component, value, color) in enumerate(zip(components, r2_values, colors)):
    ax.barh(0, value, left=cumsum[i], height=0.5, color=color, edgecolor='black', linewidth=1.5)
    # Label
    ax.text(cumsum[i] + value/2, 0, f'{value:.2f}%\n({value/18.93*100:.1f}% of R²)',
           ha='center', va='center', fontweight='bold', fontsize=9, color='white')

ax.set_xlim(0, 20)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel('Erklärte Varianz (%)', fontsize=11, fontweight='bold')
ax.set_yticks([])
ax.set_title('Stacked Decomposition', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# ---- PLOT 2: Pie Chart ----
ax = axes[1]
wedges, texts, autotexts = ax.pie(r2_values, labels=components, autopct='%1.2f%%',
                                    colors=colors, startangle=90, textprops={'fontsize': 10})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)

ax.set_title('Relative Beiträge zum R²', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('results/r2_decomposition.png', dpi=300, bbox_inches='tight')
print("✓ Gespeichert: results/r2_decomposition.png")
plt.close()

# ---- Zusätzlich: Zeitliche Serie ----
fig, ax = plt.subplots(figsize=(12, 6))

models = ['M_CONFLICT\n(nur Konflikte)', 'M_CONTROL\n(+pre_vol)', 'M_TIME\n(+year)', 'M2\n(+region+sev)']
r2_models = [0.0045, 11.35, 18.93, 18.94]

bars = ax.bar(models, r2_models, color=['#e74c3c', '#3498db', '#f39c12', '#2ecc71'], 
             edgecolor='black', linewidth=2, alpha=0.8)

# Labels auf den Balken
for bar, val in zip(bars, r2_models):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{val:.2f}%',
           ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('R² (%)', fontsize=12, fontweight='bold')
ax.set_title('Modell-Vergleich: Der Schritt-für-Schritt Aufbau von M2', fontsize=13, fontweight='bold')
ax.set_ylim(0, 22)
ax.grid(True, alpha=0.3, axis='y')

# Annotationen
ax.annotate('', xy=(0, 0.5), xytext=(1, 0.5),
           arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
ax.text(0.5, 1, '+11.35%\n(Mean Reversion)', ha='center', fontweight='bold', fontsize=9,
       bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))

ax.annotate('', xy=(1, 11.5), xytext=(2, 11.5),
           arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
ax.text(1.5, 12, '+7.58%\n(Makroschocks)', ha='center', fontweight='bold', fontsize=9,
       bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))

plt.tight_layout()
plt.savefig('results/model_progression.png', dpi=300, bbox_inches='tight')
print("✓ Gespeichert: results/model_progression.png")
plt.close()

print("\nVisualisierungen erstellt!")
