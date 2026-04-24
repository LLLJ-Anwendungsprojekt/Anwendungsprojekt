import pandas as pd

# Lade alle Daten
data = pd.read_csv('data/processed/conflict_market_features_sample10k.csv')
assigns = pd.read_csv('results/kmeans_cluster_assignments.csv')

# Finde den Ausreißer
outlier_idx = assigns[assigns['cluster'] == 1].index[0]
outlier_row = data.iloc[outlier_idx]

print('='*60)
print('AUSREISSER DETAILS (ID 445207)')
print('='*60)
print()
print('KONFLIKT-EIGENSCHAFTEN:')
print(f'  Konflikt-ID: {int(outlier_row["id"])}')
print(f'  Konflikttyp: {int(outlier_row["type_of_violence"])}')
print(f'  Region: {outlier_row["region"]}')
print(f'  Datum: {outlier_row["date_start"]} bis {outlier_row["date_end"]}')
print(f'  Dauer: {int(outlier_row["conflict_duration_days"])} Tage')
print()
print('OPFERZAHLEN (!!BESORGNISERREGEND!!):')
print(f'  Best Estimate (Todesfaelle): {int(outlier_row["best"])} ⚠️')
print(f'  Total Deaths: {int(outlier_row["total_deaths"])} ⚠️')
print(f'  Deaths A (Partei 1): {int(outlier_row["deaths_a"])}')
print(f'  Deaths B (Partei 2): {int(outlier_row["deaths_b"])}')
print(f'  Zivilisten: {int(outlier_row["deaths_civilians"])}')
print(f'  Severity Class: {int(outlier_row["severity_class"])} (hoechste)')
print()
print('MARKTREAKTIONEN:')
print(f'  Pre-Event Return: {outlier_row["pre_return"]:.6f}')
print(f'  Event-Tag Return: {outlier_row["event_return"]:.6f}')
print(f'  Post-Event Return: {outlier_row["post_return"]:.6f}')
print(f'  Market Reaction (Post - Pre): {outlier_row["market_reaction"]:.6f}')
print(f'  Pre-Volatilitaet: {outlier_row["pre_volatility"]:.6f}')
print(f'  Post-Volatilitaet: {outlier_row["post_volatility"]:.6f}')
print(f'  Volatilitaets-Aenderung: {outlier_row["volatility_change"]:.6f}')
print(f'  Indizes tracked: {int(outlier_row["n_indices_tracked"])}')
print()
print('KLASSIFIKATION:')
print(f'  Market Direction 5d: {int(outlier_row["market_direction_5d"])} (0=negativ)')
print()
print('='*60)
print('WHY IS THIS AN OUTLIER?')
print('='*60)
print('Der Ausreißer hat 40.000 Todesfaelle - das ist EXTREM!')
print('Zum Vergleich - Average in Cluster 0:')

cluster_0 = data[assigns['cluster'] == 0]
print(f'  Average Deaths in Cluster 0: {cluster_0["best"].mean():.0f}')
print(f'  Max Deaths in Cluster 0: {cluster_0["best"].max():.0f}')
print(f'  Min Deaths in Cluster 0: {cluster_0["best"].min():.0f}')
print()
print('Der Konflikt in ID 445207 ist eine MASSAKER-KLASSE Veranstaltung!')
