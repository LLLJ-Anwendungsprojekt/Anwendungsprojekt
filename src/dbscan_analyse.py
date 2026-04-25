"""
DBSCAN-Analyse: Geopolitische Konflikte & Aktienindizes

Forschungsfrage: Wie beeinflussen geopolitische Konflikte mit ähnlichen Mustern 
die Reaktion von Aktienindizes?

Workflow:
1. Daten laden & explorieren
2. Feature Engineering (Log-Transform, Ordinal-Encoding)
3. Standardisierung (essentiell für DBSCAN)
4. k-distance plot zur Epsilon-Bestimmung
5. DBSCAN-Clustering
6. Cluster-Analyse & Marktreaktion
7. Statistische Validierung
8. Visualisierungen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from scipy.stats import kruskal
import warnings
import os

try:
    _kneed_module = __import__("kneed", fromlist=["KneeLocator"])
    KneeLocator = getattr(_kneed_module, "KneeLocator", None)
    KNEELOCATOR_AVAILABLE = KneeLocator is not None
except ImportError:
    KneeLocator = None
    KNEELOCATOR_AVAILABLE = False

warnings.filterwarnings('ignore')

# ============================================================================
# KONFIGURATION
# ============================================================================

DATA_PATH = "data/processed/conflict_market_features.csv"
RESULTS_DIR = "results"
PLOT_DIR = "results/dbscan_plots"

# DBSCAN Parameter
INITIAL_EPS = 0.5
MIN_SAMPLES = 5
AUTO_EPS_PERCENTILE = 95
SENSITIVITY_EPS_VALUES = [0.4, 0.5, 0.6]

# Feature Engineering
RANDOM_STATE = 42
VIOLENCE_FEATURE_WEIGHT = 0.35

# Performance-Schutz: DBSCAN skaliert schlecht bei sehr großen Datenmengen
MAX_ROWS_FOR_DBSCAN = 20000

# Visualisierungs-Schutz: nur die größten Cluster in Detailplots zeigen
TOP_N_CLUSTERS_FOR_DETAIL_PLOTS = 15
TOP_N_STRATA_PER_LEVEL = 12
MIN_ROWS_PER_STRATUM = 200

# Empfohlenes Kernset (methodisch robust gegen Fluch der Dimensionalitaet)
BASE_FEATURES = [
    "latitude",
    "longitude",
    "log_deaths",
    "lethality_ratio",
    "conflict_duration_days",
]

TEMPORAL_EXTRA_FEATURES = [
    "post_return",
    "lag_pre_to_event",
    "lag_event_to_post",
    "lag_pre_to_post",
    "date_start_year",
    "date_start_month_sin",
    "date_start_month_cos",
]


# ============================================================================
# HILFSFUNKTIONEN
# ============================================================================

def create_directories():
    """Erstelle Output-Verzeichnisse"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"[OK] Verzeichnisse erstellt: {RESULTS_DIR}, {PLOT_DIR}")


def load_data(path):
    """Lade Konflikt-Marktdaten"""
    df = pd.read_csv(path)
    
    # Nur präzise Datumsangaben behalten: date_prec 1-3
    if "date_prec" in df.columns:
        rows_before = len(df)
        df = df[df["date_prec"].isin([1, 2, 3])].copy()
        rows_after = len(df)
        print(f"\n📅 date_prec-Filter angewendet (1-3): {rows_after}/{rows_before} Zeilen behalten")
    else:
        print("\n⚠️ Spalte 'date_prec' nicht gefunden - kein date_prec-Filter angewendet")

    print(f"\n📊 Daten geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")
    print(f"Spalten: {df.columns.tolist()}")
    return df


def feature_engineering(df):
    """
    Feature Engineering für DBSCAN
    
    1. Log-Transformation: Todeszahlen sind rechtsschief
    2. Ordinal-Encoding: type_of_violence als robuste Zusatzvariable
    3. Zeitdimension: date_start + Return-Lag-Features
    4. Methodischer Vergleich: ohne vs. mit type_of_violence + Zeitmodell
    """
    df_fe = df.copy()
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING")
    print("="*70)
    
    # 1. Log-Transformation (Todeszahlen)
    print("\n1️⃣ Log-Transformation: log_deaths = log(total_deaths + 1)")
    df_fe["log_deaths"] = np.log1p(df_fe["total_deaths"])
    
    print(f"   - total_deaths: min={df_fe['total_deaths'].min()}, max={df_fe['total_deaths'].max()}")
    print(f"   - log_deaths: min={df_fe['log_deaths'].min():.2f}, max={df_fe['log_deaths'].max():.2f}")
    print(f"   - Grund: Todeszahlen sind extrem rechtsschief verteilt!")
    
    # 2. Ordinal-Encoding fuer type_of_violence
    print("\n2️⃣ Ordinal-Encoding: type_of_violence")
    violence_values = sorted(df_fe["type_of_violence"].dropna().unique().tolist())
    violence_mapping = {value: idx for idx, value in enumerate(violence_values)}
    df_fe["violence_ordinal"] = df_fe["type_of_violence"].map(violence_mapping)
    print(f"   - Mapping: {violence_mapping}")

    # 3. Zeitfeatures und Lag-Features
    print("\n3️⃣ Zeitdimension: date_start + Lag-Features")
    df_fe["date_start_dt"] = pd.to_datetime(df_fe.get("date_start"), errors="coerce")
    df_fe["date_start_year"] = df_fe["date_start_dt"].dt.year
    df_fe["date_start_month"] = df_fe["date_start_dt"].dt.month
    month_angle = 2 * np.pi * (df_fe["date_start_month"] - 1) / 12.0
    df_fe["date_start_month_sin"] = np.sin(month_angle)
    df_fe["date_start_month_cos"] = np.cos(month_angle)

    # Ereignisfenster als pseudo-lags (t-1 -> t -> t+1)
    df_fe["lag_pre_to_event"] = df_fe["event_return"] - df_fe["pre_return"]
    df_fe["lag_event_to_post"] = df_fe["post_return"] - df_fe["event_return"]
    df_fe["lag_pre_to_post"] = df_fe["post_return"] - df_fe["pre_return"]
    print("   - Erstellt: date_start_year, date_start_month_sin/cos, lag_pre_to_event, lag_event_to_post, lag_pre_to_post")

    # 4. Kern-Feature-Set + optionale Featuresets
    model_features = BASE_FEATURES + ["violence_ordinal"] + TEMPORAL_EXTRA_FEATURES
    print(f"\n4️⃣ Feature-Liste fuer Modellvergleich ({len(model_features)} Features):")
    for i, feat in enumerate(model_features, 1):
        print(f"   {i:2d}. {feat}")
    print("   Hinweis: Primaermodell nutzt BASE_FEATURES; Violence- und Zeitmodell werden separat verglichen.")

    # Pruefe auf fehlende Werte
    print(f"\n⚠️ Fehlende Werte pruefen...")
    X = df_fe[model_features].copy()
    missing = X.isnull().sum()
    if missing.sum() > 0:
        print(f"   Warnung: {missing[missing > 0].to_dict()}")
        X = X.fillna(X.mean())
        print(f"   -> Mit Mittelwert gefuellt")
    else:
        print(f"   ✓ Keine fehlenden Werte")

    # Rueckschreiben der imputierten Spalten
    for col in model_features:
        df_fe[col] = X[col]

    return df_fe, violence_mapping


def standardize_features(X, feature_names, feature_weights=None):
    """
    Standardisierung: ESSENTIELL für DBSCAN
    
    StandardScaler normalisiert alle Features auf:
    - Mittelwert = 0
    - Standardabweichung = 1
    
    WARUM?
    DBSCAN berechnet Distanzen. Ohne Standardisierung würden Features
    mit großen Zahlenbereichen (z.B. Todeszahlen 0-10000) die Distanz
    dominieren, während kleine Features ignoriert würden.
    """
    print("\n" + "="*70)
    print("STANDARDISIERUNG (StandardScaler)")
    print("="*70)
    print("\n✓ Standardisierung: (X - mean) / std fuer alle Features")
    print("  -> Alle Features haben Mittelwert=0, Std=1")
    print("  -> DBSCAN-Distanzen werden dadurch vergleichbar")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if feature_weights:
        print("\n⚖️ Feature-Gewichte nach Standardisierung:")
        for feature_name, weight in feature_weights.items():
            if feature_name in feature_names:
                idx = feature_names.index(feature_name)
                X_scaled[:, idx] = X_scaled[:, idx] * weight
                print(f"   - {feature_name}: Gewicht {weight}")
    
    example_feature = feature_names[0]
    example_idx = 0
    print(f"\nVor Standardisierung (Beispiel - {example_feature}):")
    print(f"  Min={X[example_feature].min():.2f}, Max={X[example_feature].max():.2f}, Mean={X[example_feature].mean():.2f}")

    print(f"\nNach Standardisierung (Beispiel - {example_feature}):")
    print(f"  Min={X_scaled[:, example_idx].min():.2f}, Max={X_scaled[:, example_idx].max():.2f}, Mean={X_scaled[:, example_idx].mean():.2f}")
    
    return X_scaled, scaler


def plot_kdistance(X_scaled, k=5, save_path=None, metric="euclidean", model_name="Model"):
    """
    k-distance Plot zur Epsilon-Bestimmung
    
    Zeigt die Distanz zum k-ten nächsten Nachbarn für jeden Punkt.
    Der "Knick" in der Kurve = optimales Epsilon
    """
    print("\n" + "="*70)
    print(f"K-DISTANCE PLOT (Epsilon-Bestimmung, {model_name})")
    print("="*70)
    
    print(f"\nBerechne {k}-distance Graph...")
    algorithm = "ball_tree" if metric == "haversine" else "auto"
    neighbors = NearestNeighbors(n_neighbors=k, metric=metric, algorithm=algorithm)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, _ = neighbors_fit.kneighbors(X_scaled)
    
    # Sortiere distances vom k-ten Nachbar
    distances_sorted = np.sort(distances[:, k-1])
    
    # Plotte
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(distances_sorted, linewidth=1.5)
    ax.set_xlabel("Datenpunkte (sortiert nach Distanz)", fontsize=12)
    if metric == "haversine":
        ax.set_ylabel(f"Distanz zum {k}. naechsten Nachbarn (Radiant)", fontsize=12)
    else:
        ax.set_ylabel(f"Distanz zum {k}. naechsten Nachbarn (standardisierter Merkmalsraum)", fontsize=12)
    ax.set_title(f"k-distance Plot (k={k}, {model_name})", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gespeichert: {save_path}")
    
    # plt.show() verzögert auf Headless-System - überspringen
    plt.close()
    
    print(f"\n📍 k-distance Statistik:")
    print(f"   - Min: {distances_sorted.min():.4f}")
    print(f"   - 25%: {np.percentile(distances_sorted, 25):.4f}")
    print(f"   - Median: {np.median(distances_sorted):.4f}")
    print(f"   - 75%: {np.percentile(distances_sorted, 75):.4f}")
    print(f"   - 95%: {np.percentile(distances_sorted, 95):.4f}")
    print(f"   - Max: {distances_sorted.max():.4f}")
    
    print(f"\n💡 Interpretations-Hinweis:")
    print(f"   Automatische eps-Schaetzung nutzt das {AUTO_EPS_PERCENTILE}. Perzentil")
    print(f"   der k-distance-Verteilung als robuste Naeherung fuer den Knick.")
    
    return distances_sorted


def estimate_eps(distances_sorted, percentile=AUTO_EPS_PERCENTILE, model_name="Model"):
    """Schaetze epsilon per Perzentil und optional per KneeLocator."""
    eps_percentile = float(np.percentile(distances_sorted, percentile))
    eps_knee = None
    method_used = "percentile"

    if KNEELOCATOR_AVAILABLE and len(distances_sorted) >= 10:
        x = np.arange(len(distances_sorted))
        try:
            knee = KneeLocator(x, distances_sorted, curve="convex", direction="increasing")
            if knee.knee is not None:
                knee_idx = int(knee.knee)
                if 0 <= knee_idx < len(distances_sorted):
                    eps_knee = float(distances_sorted[knee_idx])
        except Exception as exc:
            print(f"⚠️ KneeLocator Fehler ({model_name}): {exc}")

    eps_used = eps_knee if eps_knee is not None and eps_knee > 0 else eps_percentile
    if eps_knee is not None and eps_knee > 0:
        method_used = "knee"

    print("\n" + "="*70)
    print(f"AUTOMATISCHE EPSILON-SCHAETZUNG ({model_name})")
    print("="*70)
    print(f"Perzentil-Regel: eps = {percentile}. Perzentil -> {eps_percentile:.6f}")
    if KNEELOCATOR_AVAILABLE:
        if eps_knee is None:
            print("KneeLocator: kein stabiler Knick gefunden")
        else:
            print(f"KneeLocator: eps_knee = {eps_knee:.6f}")
    else:
        print("KneeLocator: Paket 'kneed' nicht installiert, Vergleich uebersprungen")
    print(f"Verwendetes eps: {eps_used:.6f} ({method_used})")

    return {
        "eps_used": eps_used,
        "eps_percentile": eps_percentile,
        "eps_knee": eps_knee,
        "method_used": method_used,
    }


def apply_dbscan(X_scaled, eps=0.5, min_samples=5, metric="euclidean", model_name="Model"):
    """Wende DBSCAN-Clustering an"""
    print("\n" + "="*70)
    print(f"DBSCAN-CLUSTERING [{model_name}] (eps={eps}, min_samples={min_samples}, metric={metric})")
    print("="*70)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = dbscan.fit_predict(X_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"\n✓ DBSCAN angewendet")
    print(f"   - Anzahl Cluster: {n_clusters}")
    print(f"   - Noise-Punkte (-1): {n_noise} ({100*n_noise/len(labels):.1f}%)")
    print(f"   - Punkte in Clustern: {len(labels) - n_noise}")
    
    # Cluster-Statistik
    print(f"\n📊 Cluster-Größen:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        if label == -1:
            print(f"   Noise (-1): {count} Punkte")
        else:
            print(f"   Cluster {label}: {count} Punkte")
    
    return labels


def analyze_clusters(df, labels, save_name="cluster_analysis.csv"):
    """Analysiere Cluster und Marktreaktion"""
    print("\n" + "="*70)
    print("CLUSTER-ANALYSE & MARKTREAKTION")
    print("="*70)
    
    df_analysis = df.copy()
    df_analysis["cluster"] = labels
    
    # Aggregiere nach Cluster
    analysis = df_analysis.groupby("cluster").agg({
        "market_reaction": ["mean", "std", "count"],
        "volatility_change": ["mean", "std"],
        "event_return": ["mean", "std"],
        "total_deaths": ["mean", "median"],
        "conflict_duration_days": ["mean"],
        "lethality_ratio": ["mean"],
        "latitude": ["mean"],
        "longitude": ["mean"],
    }).round(4)
    
    print("\n✓ Aggregierte Statistik pro Cluster:")
    print(analysis)
    
    # Speichere
    analysis_path = f"{RESULTS_DIR}/{save_name}"
    analysis.to_csv(analysis_path)
    print(f"\n✓ Gespeichert: {analysis_path}")
    
    return df_analysis, analysis


def statistical_validation(df_analysis):
    """Statistische Validierung: Sind Unterschiede signifikant?"""
    print("\n" + "="*70)
    print("STATISTISCHE VALIDIERUNG (Kruskal-Wallis Test)")
    print("="*70)
    
    # Nur Cluster (keine Noise)
    clusters_only = df_analysis[df_analysis["cluster"] != -1]
    unique_clusters = [c for c in clusters_only["cluster"].unique() if c != -1]
    
    if len(unique_clusters) < 2:
        print("⚠️  Weniger als 2 Cluster - Kruskal-Wallis Test nicht sinnvoll")
        return None, None
    
    # Kruskal-Wallis Test
    data_groups = [
        clusters_only[clusters_only["cluster"] == c]["market_reaction"].values
        for c in unique_clusters
    ]
    
    h_stat, p_value = kruskal(*data_groups)
    
    print(f"\nNull-Hypothese: Marktreaktion ist GLEICH über alle Cluster")
    print(f"Alternative: Marktreaktion ist UNTERSCHIEDLICH zwischen Clustern")
    print(f"\nErgebnis:")
    print(f"   H-Statistik: {h_stat:.4f}")
    print(f"   p-Wert: {p_value:.6f}")
    
    if p_value < 0.05:
        print(f"\n✅ SIGNIFIKANT (p < 0.05)")
        print(f"   → Die Unterschiede in der Marktreaktion sind NICHT zufällig!")
        print(f"   → Es gibt echte Muster!")
    else:
        print(f"\n❌ NICHT signifikant (p >= 0.05)")
        print(f"   → Die Unterschiede könnten Zufall sein")
    
    return h_stat, p_value


def calculate_silhouette(X_scaled, labels):
    """Berechne Silhouette Score"""
    print("\n" + "="*70)
    print("QUALITÄTSMETRIKEN (Silhouette Score)")
    print("="*70)
    
    # Nur Cluster (nicht Noise)
    cluster_mask = labels != -1
    if cluster_mask.sum() < 2:
        print("⚠️  Weniger als 2 Cluster - Silhouette Score nicht berechenbar")
        return None

    unique_cluster_labels = np.unique(labels[cluster_mask])
    if len(unique_cluster_labels) < 2:
        print("⚠️  Nur ein Cluster vorhanden - Silhouette Score nicht berechenbar")
        return None

    # Silhouette ist nur gueltig, wenn 2 <= Anzahl Label <= n_samples - 1
    n_samples = int(cluster_mask.sum())
    n_labels = int(len(unique_cluster_labels))
    if n_labels >= n_samples:
        print("⚠️  Ungueltige Label-Konstellation fuer Silhouette (n_labels >= n_samples)")
        return None
    
    try:
        score = silhouette_score(X_scaled[cluster_mask], labels[cluster_mask])
    except ValueError as exc:
        print(f"⚠️  Silhouette Score konnte nicht berechnet werden: {exc}")
        return None
    
    print(f"\nSilhouette Score: {score:.4f}")
    print("Hinweis: Bei dichtebasierten Verfahren (DBSCAN) ist die Aussagekraft")
    print("des Silhouette Scores wegen Noise-Punkten eingeschraenkt.")
    print(f"Interpretation:")
    if score > 0.5:
        print(f"   ✅ Gute Cluster ({score:.3f} > 0.5)")
    elif score > 0.3:
        print(f"   ⚠️  Mittelmäßige Cluster ({score:.3f})")
    else:
        print(f"   ❌ Schwache Cluster ({score:.3f} < 0.3)")
    
    return score


def run_sensitivity_analysis(X_scaled, eps_values, min_samples, metric, model_name):
    """Teste Robustheit der Clusterung fuer mehrere eps-Werte."""
    print("\n" + "="*70)
    print(f"SENSITIVITAETSANALYSE [{model_name}]")
    print("="*70)

    rows = []
    for eps in eps_values:
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        noise_share = 100 * n_noise / len(labels)
        rows.append(
            {
                "model": model_name,
                "eps": eps,
                "min_samples": min_samples,
                "metric": metric,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_share_pct": round(noise_share, 2),
            }
        )
        print(
            f"eps={eps:.3f} -> Cluster={n_clusters}, Noise={n_noise} "
            f"({noise_share:.1f}%)"
        )

    sensitivity_df = pd.DataFrame(rows)
    out_path = f"{RESULTS_DIR}/dbscan_sensitivity_{model_name.lower().replace(' ', '_')}.csv"
    sensitivity_df.to_csv(out_path, index=False)
    print(f"✓ Sensitivitaetsanalyse gespeichert: {out_path}")
    return sensitivity_df


def run_regional_robustness_analysis(df_analysis, model_name):
    """Pruefe Robustheit der Clusterstruktur nach region und country_id."""
    print("\n" + "="*70)
    print(f"REGIONALE ROBUSTHEITSANALYSE [{model_name}]")
    print("="*70)

    rows = []
    strata_columns = ["region", "country_id"]

    for col in strata_columns:
        if col not in df_analysis.columns:
            print(f"⚠️ Spalte '{col}' fehlt - ueberspringe")
            continue

        top_values = (
            df_analysis[col]
            .value_counts(dropna=True)
            .head(TOP_N_STRATA_PER_LEVEL)
            .index
            .tolist()
        )

        for value in top_values:
            subset = df_analysis[df_analysis[col] == value]
            if len(subset) < MIN_ROWS_PER_STRATUM:
                continue

            labels = subset["cluster"].to_numpy()
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_share = float(100 * (labels == -1).sum() / len(labels))

            clusters_only = subset[subset["cluster"] != -1]
            unique_clusters = clusters_only["cluster"].unique().tolist()
            p_value = None
            if len(unique_clusters) >= 2:
                data_groups = [
                    clusters_only.loc[clusters_only["cluster"] == c, "market_reaction"].values
                    for c in unique_clusters
                ]
                try:
                    _, p_value = kruskal(*data_groups)
                except Exception:
                    p_value = None

            rows.append(
                {
                    "model": model_name,
                    "strata_level": col,
                    "strata_value": value,
                    "n_rows": len(subset),
                    "n_clusters": n_clusters,
                    "noise_share_pct": round(noise_share, 2),
                    "p_value_kruskal": p_value,
                }
            )

    regional_df = pd.DataFrame(rows)
    out_path = f"{RESULTS_DIR}/regional_robustness_{model_name.lower().replace(' ', '_')}.csv"
    regional_df.to_csv(out_path, index=False)
    print(f"✓ Regionale Robustheitsanalyse gespeichert: {out_path}")
    return regional_df


def visualize_results(df_analysis, X_scaled, labels, save_dir=PLOT_DIR):
    """Erstelle Visualisierungen"""
    print("\n" + "="*70)
    print("VISUALISIERUNGEN")
    print("="*70)
    
    # 1. Geografische Cluster-Karte
    print("\n1️⃣ Geografische Cluster-Karte...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Farben für Cluster
    scatter = ax.scatter(
        df_analysis["longitude"],
        df_analysis["latitude"],
        c=df_analysis["cluster"],
        cmap="tab20",
        s=30,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Noise in Rot
    noise_mask = df_analysis["cluster"] == -1
    ax.scatter(
        df_analysis.loc[noise_mask, "longitude"],
        df_analysis.loc[noise_mask, "latitude"],
        marker='X',
        s=100,
        c='red',
        label='Noise',
        edgecolors='darkred',
        linewidth=1
    )
    
    ax.set_xlabel("Längengrad (longitude, Grad)", fontsize=12)
    ax.set_ylabel("Breitengrad (latitude, Grad)", fontsize=12)
    ax.set_title("Geopolitische Konflikt-Cluster", fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label="Cluster-ID (DBSCAN)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    path1 = f"{save_dir}/01_geographic_clusters.png"
    plt.savefig(path1, dpi=300, bbox_inches='tight')
    print(f"   ✓ Gespeichert: {path1}")
    plt.close()
    
    # 2. Marktreaktion pro Cluster (Boxplot)
    print("2️⃣ Marktreaktion pro Cluster...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Nur größte Cluster für lesbare Detailplots
    df_plot = df_analysis[df_analysis["cluster"] != -1]
    top_clusters = (
        df_plot["cluster"]
        .value_counts()
        .head(TOP_N_CLUSTERS_FOR_DETAIL_PLOTS)
        .index
        .tolist()
    )
    df_plot_top = df_plot[df_plot["cluster"].isin(top_clusters)].copy()

    print(
        f"   ℹ️ Detailplots zeigen Top-{TOP_N_CLUSTERS_FOR_DETAIL_PLOTS} Cluster "
        f"(ohne Noise) für bessere Lesbarkeit"
    )

    sns.boxplot(
        x="cluster",
        y="market_reaction",
        data=df_plot_top,
        order=top_clusters,
        ax=ax,
        palette="Set2"
    )
    ax.set_xlabel(f"Cluster (ID, Top-{TOP_N_CLUSTERS_FOR_DETAIL_PLOTS} nach Größe)", fontsize=12)
    ax.set_ylabel("Marktreaktion (market_reaction)", fontsize=12)
    ax.set_title("Marktreaktion pro Konflikt-Cluster (Top-Cluster)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    path2 = f"{save_dir}/02_market_reaction_boxplot.png"
    plt.savefig(path2, dpi=300, bbox_inches='tight')
    print(f"   ✓ Gespeichert: {path2}")
    plt.close()
    
    # 3. Cluster-Größen (Bar Plot)
    print("3️⃣ Cluster-Größen...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Zeige Noise plus größte Cluster für bessere Lesbarkeit
    cluster_counts_full = df_analysis["cluster"].value_counts()
    top_clusters_size_plot = cluster_counts_full[cluster_counts_full.index != -1].head(TOP_N_CLUSTERS_FOR_DETAIL_PLOTS).index.tolist()
    plot_clusters = ([-1] if -1 in cluster_counts_full.index else []) + top_clusters_size_plot
    cluster_counts = cluster_counts_full.reindex(plot_clusters).dropna()

    # Erstelle Farb-Array (rot für Noise, blau für Cluster)
    colors_map = {-1: 'red'}
    colors = [colors_map.get(idx, 'steelblue') for idx in cluster_counts.index]
    
    cluster_counts.plot(kind='bar', ax=ax, color=colors, rot=45)
    ax.set_xlabel(f"Cluster (ID, -1 = Noise, sonst Top-{TOP_N_CLUSTERS_FOR_DETAIL_PLOTS})", fontsize=12)
    ax.set_ylabel("Anzahl Konflikte (count)", fontsize=12)
    ax.set_title("Cluster-Größen (Noise + Top-Cluster)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    path3 = f"{save_dir}/03_cluster_sizes.png"
    plt.savefig(path3, dpi=300, bbox_inches='tight')
    print(f"   ✓ Gespeichert: {path3}")
    plt.close()
    
    # 4. Volatilität pro Cluster
    print("4️⃣ Volatilität pro Cluster...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.boxplot(
        x="cluster",
        y="volatility_change",
        data=df_plot_top,
        order=top_clusters,
        ax=ax,
        palette="Set3"
    )
    ax.set_xlabel(f"Cluster (ID, Top-{TOP_N_CLUSTERS_FOR_DETAIL_PLOTS} nach Größe)", fontsize=12)
    ax.set_ylabel("Volatilitätsänderung (volatility_change)", fontsize=12)
    ax.set_title("Volatilitätsveränderung pro Cluster (Top-Cluster)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    path4 = f"{save_dir}/04_volatility_boxplot.png"
    plt.savefig(path4, dpi=300, bbox_inches='tight')
    print(f"   ✓ Gespeichert: {path4}")
    plt.close()
    
    # 5. Totales Deaths pro Cluster
    print("5️⃣ Todeszahlen pro Cluster...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.violinplot(
        x="cluster",
        y="log_deaths",
        data=df_plot_top,
        order=top_clusters,
        ax=ax,
        palette="muted"
    )
    ax.set_xlabel(f"Cluster (ID, Top-{TOP_N_CLUSTERS_FOR_DETAIL_PLOTS} nach Größe)", fontsize=12)
    ax.set_ylabel("Konfliktintensität (log(total_deaths + 1))", fontsize=12)
    ax.set_title("Konflikt-Intensität (log-transformiert) pro Cluster (Top-Cluster)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    path5 = f"{save_dir}/05_deaths_violin.png"
    plt.savefig(path5, dpi=300, bbox_inches='tight')
    print(f"   ✓ Gespeichert: {path5}")
    plt.close()
    
    print(f"\n✓ Alle Plots gespeichert in: {save_dir}/")


def generate_report(
    primary_df_analysis,
    primary_analysis_stats,
    primary_silhouette,
    primary_h_stat,
    primary_p_value,
    primary_eps_info,
    min_samples,
    comparison_df,
    sensitivity_primary,
    sensitivity_temporal,
    sensitivity_haversine,
    regional_primary,
    regional_temporal,
):
    """Generiere zusammenfassenden Report"""
    print("\n" + "="*70)
    print("ZUSAMMENFASSUNG & REPORT")
    print("="*70)
    
    # Hilfsvariablen fuer Report-Formatierung
    silhouette_text = f"{primary_silhouette:.4f}" if primary_silhouette is not None else "N/A"
    h_stat_text = f"{primary_h_stat:.4f}" if primary_h_stat is not None else "N/A"
    p_value_text = f"{primary_p_value:.6f}" if primary_p_value is not None else "N/A"
    if primary_silhouette is None:
        silhouette_interp = "N/A"
    elif primary_silhouette > 0.5:
        silhouette_interp = "Gute Cluster"
    elif primary_silhouette > 0.3:
        silhouette_interp = "Mittelmaessige Cluster"
    else:
        silhouette_interp = "Schwache Cluster"

    if primary_p_value is None:
        sig_text = "N/A"
        market_text = "Keine Aussage moeglich"
    elif primary_p_value < 0.05:
        sig_text = "JA (p < 0.05)"
        market_text = "Die Marktreaktion ist abhaengig vom Konflikt-Cluster!"
    else:
        sig_text = "NEIN (p >= 0.05)"
        market_text = "Die Marktreaktion ist unabhaengig vom Konflikt-Cluster!"

    n_clusters = primary_df_analysis[primary_df_analysis["cluster"] != -1]["cluster"].nunique()
    
    report = f"""
================================================================================
DBSCAN-ANALYSE: Geopolitische Konflikte & Aktienindizes
================================================================================

FORSCHUNGSFRAGE:
Wie beeinflussen geopolitische Konflikte mit ähnlichen Mustern die Reaktion 
von Aktienindizes?

ERGEBNISSE:
================================================================================

1. DBSCAN-PARAMETER:
    - Epsilon (eps, verwendet): {primary_eps_info['eps_used']:.6f}
    - Epsilon (Perzentil): {primary_eps_info['eps_percentile']:.6f}
    - Epsilon (KneeLocator): {primary_eps_info['eps_knee'] if primary_eps_info['eps_knee'] is not None else 'N/A'}
    - Epsilon-Methode: {primary_eps_info['method_used']}
   - Minimale Nachbarn (min_samples): {min_samples}

2. CLUSTER-ERGEBNISSE:
   - Anzahl Cluster: {n_clusters}
    - Punkte in Clustern: {(primary_df_analysis['cluster'] != -1).sum()}
    - Noise-Punkte: {(primary_df_analysis['cluster'] == -1).sum()}
    - Anteil Noise: {100*(primary_df_analysis['cluster'] == -1).sum()/len(primary_df_analysis):.1f}%

3. QUALITÄT (Silhouette Score):
   - Score: {silhouette_text}
   - Interpretation: {silhouette_interp}
    - Methodischer Hinweis: Der Silhouette Score wurde ergaenzend berechnet,
      seine Aussagekraft ist bei dichtebasierten Verfahren (DBSCAN) eingeschraenkt.

4. STATISTISCHE SIGNIFIKANZ (Kruskal-Wallis):
   - H-Statistik: {h_stat_text}
   - p-Wert: {p_value_text}
   - Signifikant? {sig_text}

   → {market_text}

5. MARKTREAKTION PRO CLUSTER:
   
{primary_analysis_stats.to_string()}

6. MODELLVERGLEICH (OHNE vs. MIT type_of_violence):

{comparison_df.to_string(index=False)}

7. SENSITIVITAETSANALYSE (eps = 0.4 / 0.5 / 0.6):

Primaermodell (ohne violence):
{sensitivity_primary.to_string(index=False)}

Zeitmodell (date_start + post_return + Lag-Features):
{sensitivity_temporal.to_string(index=False)}

Haversine-Baseline (nur Koordinaten in Radiant):
{sensitivity_haversine.to_string(index=False)}

8. REGIONALE ROBUSTHEIT (STRATA: region / country_id):

Primaermodell:
{regional_primary.to_string(index=False)}

Zeitmodell:
{regional_temporal.to_string(index=False)}

INTERPRETATION:
================================================================================
- Welche Cluster haben die stärkste Marktreaktion?
- Gibt es geografische Muster?
- Sind hochgradig tödliche Konflikte / hohe Volatilität korreliert?
- Reagiert der Markt auf die Dauer oder die Intensität des Konflikts?

NÄCHSTE SCHRITTE:
- Weitere Feature-Engineering (z.B. Zeitverzögerungen)?
- Andere Clustering-Methoden vergleichen (K-Means, Hierarchical)?
- Zeitreihen-Analyse pro Cluster?
- Causal Inference zur Markt-Reaktion?

DATEIEN:
================================================================================
- {RESULTS_DIR}/cluster_analysis_no_violence.csv
- {RESULTS_DIR}/cluster_analysis_with_violence.csv
- {RESULTS_DIR}/cluster_analysis_temporal_model.csv
- {RESULTS_DIR}/cluster_comparison_summary.csv
- {RESULTS_DIR}/dbscan_sensitivity_no_violence.csv
- {RESULTS_DIR}/dbscan_sensitivity_temporal_model.csv
- {RESULTS_DIR}/dbscan_sensitivity_haversine_geo_only.csv
- {RESULTS_DIR}/regional_robustness_no_violence.csv
- {RESULTS_DIR}/regional_robustness_temporal_model.csv
- {PLOT_DIR}/01_geographic_clusters.png
- {PLOT_DIR}/02_market_reaction_boxplot.png
- {PLOT_DIR}/03_cluster_sizes.png
- {PLOT_DIR}/04_volatility_boxplot.png
- {PLOT_DIR}/05_deaths_violin.png

================================================================================
"""
    
    print(report)
    
    # Speichere Report
    report_path = f"{RESULTS_DIR}/DBSCAN_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✓ Report gespeichert: {report_path}")
    
    print("\n" + "="*70)
    print("✅ DBSCAN-ANALYSE ABGESCHLOSSEN!")
    print("="*70)
    print(f"Alle Ausgaben befinden sich in: {RESULTS_DIR}/")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Hauptfunktion - vollständiger DBSCAN-Workflow"""
    
    create_directories()
    
    # 1. Daten laden
    df = load_data(DATA_PATH)
    
    # 2. Feature Engineering
    df_fe, _ = feature_engineering(df)

    X_no_violence = df_fe[BASE_FEATURES].copy()
    X_with_violence = df_fe[BASE_FEATURES + ["violence_ordinal"]].copy()
    temporal_features = BASE_FEATURES + TEMPORAL_EXTRA_FEATURES
    X_temporal = df_fe[temporal_features].copy()
    coords_rad = np.radians(df_fe[["latitude", "longitude"]].to_numpy())

    # 2b. Optionales Sampling für DBSCAN (vermeidet MemoryError bei großen Datenmengen)
    if MAX_ROWS_FOR_DBSCAN and len(df_fe) > MAX_ROWS_FOR_DBSCAN:
        print("\n" + "="*70)
        print("SAMPLING FÜR DBSCAN")
        print("="*70)
        print(
            f"\n⚠️ Datensatz ist sehr groß ({len(df_fe)} Zeilen). "
            f"Nutze Stichprobe mit {MAX_ROWS_FOR_DBSCAN} Zeilen für stabile Laufzeit."
        )
        sample_idx = df_fe.sample(n=MAX_ROWS_FOR_DBSCAN, random_state=RANDOM_STATE).index
        sample_mask = df_fe.index.isin(sample_idx)
        df_fe = df_fe.loc[sample_idx].reset_index(drop=True)
        X_no_violence = X_no_violence.loc[sample_idx].reset_index(drop=True)
        X_with_violence = X_with_violence.loc[sample_idx].reset_index(drop=True)
        X_temporal = X_temporal.loc[sample_idx].reset_index(drop=True)
        coords_rad = coords_rad[sample_mask]
        print(f"✓ Sampling abgeschlossen: {len(df_fe)} Zeilen")

    # 3. Standardisierung fuer Primaermodell (ohne violence)
    X_no_scaled, _ = standardize_features(
        X_no_violence,
        feature_names=BASE_FEATURES,
    )

    # 3b. Standardisierung fuer Vergleichsmodell (mit violence + reduziertes Gewicht)
    with_features = BASE_FEATURES + ["violence_ordinal"]
    X_with_scaled, _ = standardize_features(
        X_with_violence,
        feature_names=with_features,
        feature_weights={"violence_ordinal": VIOLENCE_FEATURE_WEIGHT},
    )

    # 3c. Standardisierung fuer Zeitmodell
    X_temporal_scaled, _ = standardize_features(
        X_temporal,
        feature_names=temporal_features,
    )

    # 4. k-distance + automatische eps-Schaetzung (Primaermodell)
    distances_no = plot_kdistance(
        X_no_scaled,
        k=MIN_SAMPLES,
        save_path=f"{PLOT_DIR}/00_kdistance_plot_no_violence.png",
        metric="euclidean",
        model_name="No Violence",
    )
    eps_no_info = estimate_eps(distances_no, model_name="No Violence")

    # 4b. k-distance + automatische eps-Schaetzung (Vergleichsmodell)
    distances_with = plot_kdistance(
        X_with_scaled,
        k=MIN_SAMPLES,
        save_path=f"{PLOT_DIR}/00_kdistance_plot_with_violence.png",
        metric="euclidean",
        model_name="With Violence",
    )
    eps_with_info = estimate_eps(distances_with, model_name="With Violence")

    # 4c. k-distance + automatische eps-Schaetzung (Zeitmodell)
    distances_temporal = plot_kdistance(
        X_temporal_scaled,
        k=MIN_SAMPLES,
        save_path=f"{PLOT_DIR}/00_kdistance_plot_temporal_model.png",
        metric="euclidean",
        model_name="Temporal Model",
    )
    eps_temporal_info = estimate_eps(distances_temporal, model_name="Temporal Model")

    # 4d. Haversine-Baseline nur auf Koordinaten in Radiant
    distances_haversine = plot_kdistance(
        coords_rad,
        k=MIN_SAMPLES,
        save_path=f"{PLOT_DIR}/00_kdistance_plot_haversine_geo_only.png",
        metric="haversine",
        model_name="Haversine Geo Only",
    )
    eps_haversine_info = estimate_eps(distances_haversine, model_name="Haversine Geo Only")

    # 5. DBSCAN anwenden
    labels_no = apply_dbscan(
        X_no_scaled,
        eps=eps_no_info["eps_used"] if eps_no_info["eps_used"] > 0 else INITIAL_EPS,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        model_name="No Violence",
    )
    labels_with = apply_dbscan(
        X_with_scaled,
        eps=eps_with_info["eps_used"] if eps_with_info["eps_used"] > 0 else INITIAL_EPS,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        model_name="With Violence",
    )
    labels_temporal = apply_dbscan(
        X_temporal_scaled,
        eps=eps_temporal_info["eps_used"] if eps_temporal_info["eps_used"] > 0 else INITIAL_EPS,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        model_name="Temporal Model",
    )
    labels_haversine = apply_dbscan(
        coords_rad,
        eps=eps_haversine_info["eps_used"] if eps_haversine_info["eps_used"] > 0 else INITIAL_EPS,
        min_samples=MIN_SAMPLES,
        metric="haversine",
        model_name="Haversine Geo Only",
    )

    # 6. Cluster-Analyse
    df_analysis_no, analysis_stats_no = analyze_clusters(
        df_fe,
        labels_no,
        save_name="cluster_analysis_no_violence.csv",
    )
    df_analysis_with, analysis_stats_with = analyze_clusters(
        df_fe,
        labels_with,
        save_name="cluster_analysis_with_violence.csv",
    )
    df_analysis_temporal, analysis_stats_temporal = analyze_clusters(
        df_fe,
        labels_temporal,
        save_name="cluster_analysis_temporal_model.csv",
    )

    # 7. Statistische Validierung
    h_stat_no, p_value_no = statistical_validation(df_analysis_no)
    h_stat_with, p_value_with = statistical_validation(df_analysis_with)
    h_stat_temporal, p_value_temporal = statistical_validation(df_analysis_temporal)

    # 8. Qualitaetsmetrik
    silhouette_no = calculate_silhouette(X_no_scaled, labels_no)
    silhouette_with = calculate_silhouette(X_with_scaled, labels_with)
    silhouette_temporal = calculate_silhouette(X_temporal_scaled, labels_temporal)

    # 8b. Modellvergleich speichern
    comparison_df = pd.DataFrame(
        [
            {
                "model": "no_violence",
                "eps_auto": round(eps_no_info["eps_used"], 6),
                "eps_percentile": round(eps_no_info["eps_percentile"], 6),
                "eps_knee": None if eps_no_info["eps_knee"] is None else round(eps_no_info["eps_knee"], 6),
                "eps_method": eps_no_info["method_used"],
                "n_clusters": int(df_analysis_no[df_analysis_no["cluster"] != -1]["cluster"].nunique()),
                "noise_share_pct": round(100 * (df_analysis_no["cluster"] == -1).sum() / len(df_analysis_no), 2),
                "silhouette": None if silhouette_no is None else round(float(silhouette_no), 4),
                "p_value_kruskal": None if p_value_no is None else round(float(p_value_no), 6),
            },
            {
                "model": "with_violence_ordinal_weighted",
                "eps_auto": round(eps_with_info["eps_used"], 6),
                "eps_percentile": round(eps_with_info["eps_percentile"], 6),
                "eps_knee": None if eps_with_info["eps_knee"] is None else round(eps_with_info["eps_knee"], 6),
                "eps_method": eps_with_info["method_used"],
                "n_clusters": int(df_analysis_with[df_analysis_with["cluster"] != -1]["cluster"].nunique()),
                "noise_share_pct": round(100 * (df_analysis_with["cluster"] == -1).sum() / len(df_analysis_with), 2),
                "silhouette": None if silhouette_with is None else round(float(silhouette_with), 4),
                "p_value_kruskal": None if p_value_with is None else round(float(p_value_with), 6),
            },
            {
                "model": "temporal_model",
                "eps_auto": round(eps_temporal_info["eps_used"], 6),
                "eps_percentile": round(eps_temporal_info["eps_percentile"], 6),
                "eps_knee": None if eps_temporal_info["eps_knee"] is None else round(eps_temporal_info["eps_knee"], 6),
                "eps_method": eps_temporal_info["method_used"],
                "n_clusters": int(df_analysis_temporal[df_analysis_temporal["cluster"] != -1]["cluster"].nunique()),
                "noise_share_pct": round(100 * (df_analysis_temporal["cluster"] == -1).sum() / len(df_analysis_temporal), 2),
                "silhouette": None if silhouette_temporal is None else round(float(silhouette_temporal), 4),
                "p_value_kruskal": None if p_value_temporal is None else round(float(p_value_temporal), 6),
            },
            {
                "model": "haversine_geo_only",
                "eps_auto": round(eps_haversine_info["eps_used"], 6),
                "eps_percentile": round(eps_haversine_info["eps_percentile"], 6),
                "eps_knee": None if eps_haversine_info["eps_knee"] is None else round(eps_haversine_info["eps_knee"], 6),
                "eps_method": eps_haversine_info["method_used"],
                "n_clusters": int(len(set(labels_haversine)) - (1 if -1 in labels_haversine else 0)),
                "noise_share_pct": round(100 * (labels_haversine == -1).sum() / len(labels_haversine), 2),
                "silhouette": None,
                "p_value_kruskal": None,
            },
        ]
    )
    comparison_path = f"{RESULTS_DIR}/cluster_comparison_summary.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"✓ Modellvergleich gespeichert: {comparison_path}")

    # 8c. Sensitivitaetsanalyse (explizit geforderte eps-Werte)
    sensitivity_no = run_sensitivity_analysis(
        X_no_scaled,
        SENSITIVITY_EPS_VALUES,
        MIN_SAMPLES,
        metric="euclidean",
        model_name="No Violence",
    )
    sensitivity_temporal = run_sensitivity_analysis(
        X_temporal_scaled,
        SENSITIVITY_EPS_VALUES,
        MIN_SAMPLES,
        metric="euclidean",
        model_name="Temporal Model",
    )
    sensitivity_haversine = run_sensitivity_analysis(
        coords_rad,
        SENSITIVITY_EPS_VALUES,
        MIN_SAMPLES,
        metric="haversine",
        model_name="Haversine Geo Only",
    )

    # 8d. Regionale Robustheitsanalyse
    regional_no = run_regional_robustness_analysis(df_analysis_no, model_name="No Violence")
    regional_temporal = run_regional_robustness_analysis(df_analysis_temporal, model_name="Temporal Model")

    # 9. Visualisierungen fuer Primaermodell
    visualize_results(df_analysis_no, X_no_scaled, labels_no)

    # 10. Report
    generate_report(
        primary_df_analysis=df_analysis_no,
        primary_analysis_stats=analysis_stats_no,
        primary_silhouette=silhouette_no,
        primary_h_stat=h_stat_no,
        primary_p_value=p_value_no,
        primary_eps_info=eps_no_info,
        min_samples=MIN_SAMPLES,
        comparison_df=comparison_df,
        sensitivity_primary=sensitivity_no,
        sensitivity_temporal=sensitivity_temporal,
        sensitivity_haversine=sensitivity_haversine,
        regional_primary=regional_no,
        regional_temporal=regional_temporal,
    )
    
if __name__ == "__main__":
    main()
