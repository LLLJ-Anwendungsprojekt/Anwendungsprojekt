"""Bereinigtes Preprocessing fuer NEU_DB-Clustering.

Behebt:
- NaN/Inf-Handling (Winsorizing statt Imputation)
- Redundante Feature-Duplikate (nur best_fatalities)
- Extreme Outlier-Isolation
"""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_clustering_data_robust(df: pd.DataFrame) -> tuple:
    """Robuste Feature-Vorbereitung mit Winsorizing statt Imputation."""
    
    features_to_use = [
        "daily_return",
        "intraday_range",
        "conflict_count",
        "countries_affected",
        "regions_affected",
        "violence_type_count",
        "type_of_violence_1_count",
        "type_of_violence_2_count",
        "type_of_violence_3_count",
        "fatalities_best_sum",
        "civilian_deaths_sum",
        "unknown_deaths_sum",
        "avg_sources",
        "avg_conflict_duration_days",
        "share_high_clarity",
        "share_high_where_prec",
    ]
    
    df_feat = df[features_to_use].copy()
    
    # Ersetze Inf durch NaN
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    
    # Winsorize (clamp zu Percentilen statt median-fill)
    for col in df_feat.columns:
        p1, p99 = df_feat[col].quantile([0.01, 0.99])
        df_feat[col] = df_feat[col].clip(lower=p1, upper=p99)
    
    # Verbleibende NaN mit Spalten-Median füllen
    df_feat = df_feat.fillna(df_feat.median())
    
    # Standardisierung
    scaler = StandardScaler()
    X = scaler.fit_transform(df_feat)
    
    return X, features_to_use, df_feat


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "processed" / "index_conflict_features_NEU_DB.csv"
    results_dir = project_root / "results_NEU_DB"
    
    print("[1/5] Lade und bereinige Daten...")
    df_main = pd.read_csv(data_path)
    X, features, df_feat = prepare_clustering_data_robust(df_main)
    
    print(f"[2/5] K-Means mit Elbow-Methode (k=2..8) auf {X.shape[0]} Samples, {X.shape[1]} Features...")
    silhouettes = []
    inertias = []
    best_k = 2
    best_sil = -1
    
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=50)
        labels = km.fit_predict(X)
        from sklearn.metrics import silhouette_score
        sil = silhouette_score(X[:min(10000, len(X)), :], labels[:min(10000, len(X))], random_state=42)
        inertias.append(km.inertia_)
        silhouettes.append(sil)
        if sil > best_sil:
            best_sil = sil
            best_k = k
        print(f"  k={k}: silhouette={sil:.4f}, inertia={km.inertia_:.0f}")
    
    print(f"[3/5] Trainiere finales Modell mit k={best_k}...")
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=50)
    labels = km_final.fit_predict(X)
    
    df_main["cluster_robust"] = labels
    
    print(f"[4/5] Speichere Ergebnisse...")
    results_dir.mkdir(exist_ok=True, parents=True)
    df_main.to_csv(
        results_dir / "kmeans_cluster_assignments_NEU_DB_ROBUST.csv",
        index=False
    )
    
    profile = df_main.groupby("cluster_robust")[features].mean()
    profile.to_csv(results_dir / "kmeans_cluster_profile_NEU_DB_ROBUST.csv")
    
    print(f"\n[5/5] Cluster-Größen:")
    print(df_main["cluster_robust"].value_counts().sort_index())
    
    print(f"\nCluster-Profile (Mittelwerte):")
    print(profile.iloc[:, :8].round(3))


if __name__ == "__main__":
    main()
