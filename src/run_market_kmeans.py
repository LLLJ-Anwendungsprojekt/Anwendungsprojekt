"""Run K-Means clustering using only market-related features.

Outputs in `results/kmeans_market/`:
- cluster assignments CSV
- summary TXT
- PCA plot
- cluster_means.csv and boxplots
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler


OUT = Path("results/kmeans_market")
OUT.mkdir(parents=True, exist_ok=True)

MARKET_FEATURES = [
    "Close_last",
    "Stock_monthly_pct",
    "stock_lag1",
    "stock_lag2",
    "stock_lag3",
    "stock_vol6",
]


def main():
    df = pd.read_csv("data/processed/stocks_gpr_features.csv")
    available = [c for c in MARKET_FEATURES if c in df.columns]
    X = df[available].select_dtypes(include=[np.number]).copy()

    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    scaler = RobustScaler()
    X_s = scaler.fit_transform(X_imp)

    # find best k by silhouette
    best_k = None
    best_sil = -1
    results = []
    for k in range(2, 7):
        model = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = model.fit_predict(X_s)
        sil = silhouette_score(X_s, labels)
        results.append((k, sil, model.inertia_))
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels
            best_model = model

    # save summary
    with open(OUT / "summary.txt", "w", encoding="utf8") as f:
        f.write(f"Best k: {best_k}\n")
        f.write(f"Best silhouette: {best_sil:.4f}\n")
        f.write("k results (k,sil,inertia):\n")
        for r in results:
            f.write(f"{r}\n")

    df_out = df.copy()
    df_out["cluster"] = best_labels
    df_out.to_csv(OUT / "kmeans_market_assignments.csv", index=False)

    # PCA plot
    pca = PCA(n_components=2, random_state=42)
    x2 = pca.fit_transform(X_s)
    plot_df = pd.DataFrame({"pca1": x2[:, 0], "pca2": x2[:, 1], "cluster": best_labels})
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="pca1", y="pca2", hue="cluster", palette="tab10", s=20)
    plt.title("Market-only KMeans PCA (RobustScaler)")
    plt.tight_layout()
    plt.savefig(OUT / "kmeans_market_pca.pdf", dpi=180)
    plt.close()

    # profiles
    features = [c for c in X.columns]
    profiles = df_out.groupby("cluster")[features].mean()
    profiles.to_csv(OUT / "cluster_means.csv")

    # boxplots top features (all market features)
    cols = 3
    rows = (len(features) + cols - 1) // cols
    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, feat in enumerate(features, 1):
        plt.subplot(rows, cols, i)
        sns.boxplot(x="cluster", y=feat, data=df_out)
        plt.title(feat)
    plt.tight_layout()
    plt.savefig(OUT / "boxplots_market_grid.pdf", dpi=180)
    plt.close()

    # individual
    for feat in features:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="cluster", y=feat, data=df_out)
        plt.title(feat)
        plt.tight_layout()
        plt.savefig(OUT / f"boxplot_{feat}.pdf", dpi=180)
        plt.close()

    print("Market-only clustering saved to", OUT)


if __name__ == "__main__":
    main()
