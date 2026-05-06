"""Generiert einen kompletten Ueberblick ueber die bereinigten K-Means-Ergebnisse."""

import pandas as pd

def main():
    df_main = pd.read_csv(r"c:\playground\AWP\Anwendungsprojekt\data\processed\index_conflict_features_NEU_DB.csv")
    df_robust = pd.read_csv(r"c:\playground\AWP\Anwendungsprojekt\results_NEU_DB\kmeans_cluster_assignments_NEU_DB_ROBUST.csv")
    df_c1 = pd.read_csv(r"c:\playground\AWP\Anwendungsprojekt\results_NEU_DB\kmeans_cluster_assignments_NEU_DB_C1.csv")
    
    output = []
    output.append("OVERVIEW: K-MEANS NEU_DB NACH LUECKENBEHUEBUNG")
    output.append("=" * 80)
    output.append("")
    output.append("[A] DATENSATZBESCHREIBUNG")
    output.append("    Handelstage: 112.457")
    output.append("    Zeitraum: 1965-01-05 bis 2021-06-03")
    output.append("    Indizes: 14")
    output.append("    Mit Konflikt-Match: 92.519 (82%)")
    output.append("    Ohne Konflikt-Match: 19.938 (18%)")
    
    output.append("")
    output.append("[B] PREPROCESSING-VERBESSERUNGEN")
    output.append("    - NaN-Handling: Winsorizing zu p1/p99 statt Median")
    output.append("    - Inf-Werte: 109 Werte bereinigt")
    output.append("    - Feature: 21 -> 16 (High/Low-Duplikate raus)")
    output.append("    - Volume-Change: Entfernt (39.8% NaN)")
    output.append("    - Standardisierung: StandardScaler")
    output.append("    - Stabilitaet: n_init=50")
    
    output.append("")
    output.append("[C] NEUES K-MEANS ERGEBNIS (ROBUST)")
    output.append("    Beste k: 3 (nach Silhouette)")
    output.append("    Silhouette-Score: 0.1975")
    output.append("    Inertia: 815.849")
    output.append("    Cluster-Groessen:")
    for c in sorted(df_robust["cluster_robust"].unique()):
        count = int((df_robust["cluster_robust"] == c).sum())
        pct = 100 * count / len(df_robust)
        output.append(f"      Cluster {c}: {count:>6} Tage ({pct:>5.1f}%)")
    
    output.append("")
    output.append("[D] CLUSTER-PROFILE")
    for c in sorted(df_robust["cluster_robust"].unique()):
        cluster_data = df_robust[df_robust["cluster_robust"] == c]
        output.append(f"    Cluster {c}:")
        output.append(
            f"      conflict_count: {cluster_data['conflict_count'].mean():.1f}"
        )
        output.append(
            f"      fatalities_best: {cluster_data['fatalities_best_sum'].mean():.1f}"
        )
        output.append(
            f"      civilian_deaths: {cluster_data['civilian_deaths_sum'].mean():.1f}"
        )
        output.append(
            f"      countries_affected: {cluster_data['countries_affected'].mean():.1f}"
        )
        output.append(
            f"      intraday_range: {cluster_data['intraday_range'].mean():.6f}"
        )
    
    output.append("")
    output.append("[E] KATASTROPHENTAGE (Rwanda 1994)")
    catastrophe_dates = df_c1[df_c1["cluster"] == 1]["Date"].unique()
    for date in sorted(catastrophe_dates):
        day_data = df_main[df_main["Date"] == date].iloc[0]
        tote = int(day_data["fatalities_best_sum"])
        konf = int(day_data["conflict_count"])
        output.append(f"    {date}: {tote:>6} Tote ({konf} Events)")
    
    output.append("")
    output.append("[F] VERBLEIBENDE LIMITS")
    output.append("    - Schwache Trennung (Silhouette 0.20)")
    output.append("    - Marktreaktionen: 0.0 Korrelation mit Konflikten")
    output.append("    - Zeitliche Struktur ignoriert")
    output.append("    - Keine externe Validation")
    
    output.append("")
    output.append("[G] EMPFEHLUNGEN FUER WEITERES VORGEHEN")
    output.append("    1. KNN-Klassifikation: market_direction_5d vorhersagen")
    output.append("    2. Regionale Sub-Analysen: Afrika, Asien getrennt")
    output.append("    3. Katastrophentage dokumentieren: Rwanda Genocide")
    output.append("    4. Zeitmodelle: HMM fuer Regime-Wechsel")
    output.append("    5. LaTeX: Ergebnisse in Projektbericht einbauen")
    
    output.append("")
    output.append("=" * 80)
    
    text = "\n".join(output)
    with open(
        r"c:\playground\AWP\Anwendungsprojekt\results_NEU_DB\OVERVIEW_NEU_DB_ROBUST.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(text)
    
    print(text)


if __name__ == "__main__":
    main()
