"""
Grid-Search-Tuning für K-Means auf den Monats-Features.

Input:  data/processed/stocks_gpr_features.csv
Output: results/k_means/grid_search_results.csv, best_kmeans_assignments.csv,
        best_kmeans_pca.pdf
"""

from pathlib import Path
import itertools
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA


sns.set_style("whitegrid")

OUT_DIR = Path("results/k_means")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path("data/processed/stocks_gpr_features.csv")

DEFAULT_INCLUDE = [
    "Close_last",
    "Stock_monthly_pct",
    "GPRD_mean",
    "GPRD_ACT_mean",
    "GPRD_THREAT_mean",
    "GPRD_monthly_pct",
    "GPRD_ACT_monthly_pct",
    "GPRD_THREAT_monthly_pct",
    "gpr_lag1",
    "stock_lag1",
    "gpr_lag2",
    "stock_lag2",
    "gpr_lag3",
    "stock_lag3",
    "stock_vol6",
    "gprd_zscore",
    "gpr_spike",
]


def load_and_select(path, include=DEFAULT_INCLUDE):
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include=[np.number])
    cols = [c for c in include if c in numeric.columns]
    X = numeric[cols].copy()
    return df, X, cols


def evaluate_config(X, scaler, k, n_init, seed):
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    model = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    labels = model.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    return sil, ch, model, labels, scaler, imp


def run_grid():
    df, X, cols = load_and_select(DATA_PATH)

    scalers = {
        "standard": StandardScaler(),
        "robust": RobustScaler()
    }

    k_values = list(range(2, 7))
    n_inits = [10, 20, 50]
    seeds = [0, 42, 7]

    records = []
    start = time.time()
    for scaler_name, scaler in scalers.items():
        for n_init in n_inits:
            for k in k_values:
                sils = []
                chs = []
                for seed in seeds:
                    try:
                        sil, ch, model, labels, sc, imp = evaluate_config(X, scaler, k, n_init, seed)
                    except Exception as e:
                        warnings.warn(f"Konfiguration fehlgeschlagen: {e}")
                        sil, ch = np.nan, np.nan
                    sils.append(sil)
                    chs.append(ch)

                records.append(
                    {
                        "scaler": scaler_name,
                        "n_init": n_init,
                        "k": k,
                        "sil_mean": np.nanmean(sils),
                        "sil_std": np.nanstd(sils),
                        "ch_mean": np.nanmean(chs),
                        "ch_std": np.nanstd(chs),
                    }
                )

    grid_df = pd.DataFrame.from_records(records)
    grid_csv = OUT_DIR / "grid_search_results.csv"
    grid_df.to_csv(grid_csv, index=False)
    print(f"Grid-Ergebnisse gespeichert: {grid_csv}")

    # Bestes per Silhouetten-Mittel
    best_row = grid_df.sort_values(["sil_mean", "ch_mean"], ascending=False).iloc[0]
    print("Beste Konfiguration:", best_row.to_dict())

    # Bestes Modell refitten (Seed 42, n_init aus Config)
    best_scaler = StandardScaler() if best_row.scaler == "standard" else RobustScaler()
    best_k = int(best_row.k)
    best_n_init = int(best_row.n_init)
    _, Xcols = X, cols
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    X_scaled = best_scaler.fit_transform(X_imp)
    best_model = KMeans(n_clusters=best_k, n_init=best_n_init, random_state=42)
    labels = best_model.fit_predict(X_scaled)

    out_assign = OUT_DIR / "best_kmeans_assignments.csv"
    out_df = df.copy()
    out_df["cluster"] = labels
    out_df.to_csv(out_assign, index=False)
    print("Beste Zuordnungen gespeichert:", out_assign)

    # PCA-Plot
    pca = PCA(n_components=2, random_state=42)
    x2 = pca.fit_transform(X_scaled)
    plot_df = pd.DataFrame({"pca1": x2[:, 0], "pca2": x2[:, 1], "cluster": labels})
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="pca1", y="pca2", hue="cluster", palette="tab10", s=20)
    plt.title("Best KMeans PCA")
    plt.tight_layout()
    ppath = OUT_DIR / "best_kmeans_pca.pdf"
    plt.savefig(ppath, dpi=180)
    plt.close()
    print("PCA-Plot gespeichert:", ppath)


if __name__ == "__main__":
    run_grid()
