import pandas as pd
import numpy as np

# Lade Daten und Cluster-Zuordnungen
data = pd.read_csv('data/processed/conflict_market_features_sample10k.csv')
assigns = pd.read_csv('results/kmeans_cluster_assignments.csv')

# Finde Ruanda-Konflikt
outlier_idx = assigns[assigns['cluster'] == 1].index[0]
outlier_row = data.iloc[outlier_idx]

# Cluster 0 (Normal-Konflikte)
cluster_0_data = data[assigns['cluster'] == 0]

print("="*70)
print("MARKTAUSWIRKUNGEN: RUANDA vs NORMALE KONFLIKTE")
print("="*70)
print()

print("RUANDA-KONFLIKT (ID 445207):")
print(f"  Datum: 1994-04-18 bis 1994-05-15")
print(f"  Todesfaelle: {int(outlier_row['best']):,}")
print(f"  Severity Class: {int(outlier_row['severity_class'])} (max)")
print()
print("  MARKTREAKTIONEN:")
print(f"    Pre-Event Return (10 Tage vor): {outlier_row['pre_return']:+.6f} ({100*outlier_row['pre_return']:+.2f}%)")
print(f"    Event-Tag Return: {outlier_row['event_return']:+.6f} ({100*outlier_row['event_return']:+.2f}%)")
print(f"    Post-Event Return (5 Tage nach): {outlier_row['post_return']:+.6f} ({100*outlier_row['post_return']:+.2f}%)")
print(f"    MARKET REACTION (Post - Pre): {outlier_row['market_reaction']:+.6f} ({100*outlier_row['market_reaction']:+.2f}%)")
print(f"    Volatilitaets-Aenderung: {outlier_row['volatility_change']:+.6f}")
print(f"    Indizes: {int(outlier_row['n_indices_tracked'])} Indizes")
print()

print("="*70)
print("VERGLEICH MIT NORMALEN KONFLIKTEN (Cluster 0):")
print("="*70)
print()

# Market Reaction Statistiken
market_react_normal = cluster_0_data['market_reaction'].dropna()
market_react_ruanda = outlier_row['market_reaction']

print("MARKET REACTION (Post - Pre Return):")
print(f"  Ruanda: {market_react_ruanda:+.6f}")
print()
print("  Cluster 0 Statistics:")
print(f"    Durchschnitt: {market_react_normal.mean():+.6f}")
print(f"    Median: {market_react_normal.median():+.6f}")
print(f"    Std Dev: {market_react_normal.std():.6f}")
print(f"    Min: {market_react_normal.min():+.6f}")
print(f"    Max: {market_react_normal.max():+.6f}")
print()

# Vergleich
ruanda_z_score = (market_react_ruanda - market_react_normal.mean()) / market_react_normal.std()
print(f"  Ruanda Z-Score: {ruanda_z_score:.2f}")
if abs(ruanda_z_score) < 1:
    label = "NORMAL (keine statistischen Ausreißer)"
elif abs(ruanda_z_score) < 2:
    label = "LEICHT ERHÖHT"
else:
    label = "EXTREME AUSWIRKUNG"
print(f"  Klassifikation: {label}")
print()

# Volatilitaets-Analyse
vol_change_normal = cluster_0_data['volatility_change'].dropna()
vol_change_ruanda = outlier_row['volatility_change']

print("VOLATILITAETS-AENDERUNG:")
print(f"  Ruanda: {vol_change_ruanda:+.6f}")
print()
print("  Cluster 0 Statistics:")
print(f"    Durchschnitt: {vol_change_normal.mean():+.6f}")
print(f"    Median: {vol_change_normal.median():+.6f}")
print(f"    Std Dev: {vol_change_normal.std():.6f}")
print(f"    Min: {vol_change_normal.min():+.6f}")
print(f"    Max: {vol_change_normal.max():+.6f}")
print()

vol_z_score = (vol_change_ruanda - vol_change_normal.mean()) / vol_change_normal.std()
print(f"  Ruanda Z-Score: {vol_z_score:.2f}")
print()

# Kategorisierung nach Todesfall-Schweregrad
print("="*70)
print("TODESFALL-SCHWEREGRAD vs MARKTREAKTION:")
print("="*70)
print()

# Gruppiere nach Todesfallbereichen
death_ranges = [
    (0, 0, "Keine Todesfaelle"),
    (1, 10, "1-10 Todesfaelle"),
    (11, 100, "11-100 Todesfaelle"),
    (101, 1000, "101-1.000 Todesfaelle"),
    (1001, 10000, "1.001-10.000 Todesfaelle"),
    (10001, 1000000, ">10.000 Todesfaelle")
]

for min_d, max_d, label in death_ranges:
    mask = (cluster_0_data['best'] >= min_d) & (cluster_0_data['best'] <= max_d)
    subset = cluster_0_data[mask]['market_reaction'].dropna()
    
    if len(subset) > 0:
        avg_react = subset.mean()
        count = len(subset)
        print(f"{label}:")
        print(f"  N = {count}")
        print(f"  Avg Market Reaction: {avg_react:+.6f}")
        print()

print("Ruanda (40.000 Todesfaelle):")
print(f"  Market Reaction: {market_react_ruanda:+.6f}")
print()

# Korrelation prüfen
print("="*70)
print("KORRELATION: SCHWEREGRAD vs MARKTREAKTION")
print("="*70)
print()

correlation = cluster_0_data['best'].corr(cluster_0_data['market_reaction'])
print(f"Pearson Korrelation (Todesfaelle vs Market Reaction): {correlation:.4f}")

if abs(correlation) < 0.1:
    corr_label = "SEHR SCHWACH (fast keine Beziehung)"
elif abs(correlation) < 0.3:
    corr_label = "SCHWACH"
elif abs(correlation) < 0.5:
    corr_label = "MODERAT"
else:
    corr_label = "STARK"

print(f"Interpretation: {corr_label}")
print()

print("="*70)
print("FAZIT")
print("="*70)
print()

if abs(ruanda_z_score) < 1 and correlation < 0.1:
    print("❌ NEIN: Der Ruanda-Konflikt hatte KEINE extreme Marktauswirkung!")
    print()
    print("Erklaerung:")
    print("• Market Reaction (-0.34%) ist normal im Vergleich zu anderen Konflikten")
    print("• Z-Score ist niedrig = keine statistischen Ausreißer Reaktion")
    print("• SCHWACHE Korrelation: Todesfall-Zahlen beeinflussen Maerkte wenig")
    print("• Moegliche Gruende:")
    print("  - Konflikte sind 'bekannt' / eingepreist bevor sie eskalieren")
    print("  - Emotionale/Ethische Reaktionen schlagen sich nicht in Kursen nieder")
    print("  - Maerkte reagieren auf UEBERRASCHUNGEN, nicht auf Magnitude")
else:
    print("✓ JA: Der Ruanda-Konflikt hatte extreme Marktauswirkung!")
