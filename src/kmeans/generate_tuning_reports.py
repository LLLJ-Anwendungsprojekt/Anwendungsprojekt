"""Generate cluster profiles and boxplots for the tuning best result.

Reads `results/kmeans_tuning/best_kmeans_assignments.csv` and writes:
- `results/kmeans_tuning/cluster_means.csv`
- `results/kmeans_tuning/cluster_mean_diffs_z.csv`
- `results/kmeans_tuning/cluster_profiles.txt`
- `results/kmeans_tuning/boxplots_grid.pdf`
- `results/kmeans_tuning/boxplot_<feature>.pdf` (individual)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


OUT_DIR = Path("results/k_means")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    data_path = Path("data/processed/stocks_gpr_features.csv")
    assign_path = OUT_DIR / "best_kmeans_assignments.csv"
    if not assign_path.exists():
        raise FileNotFoundError(assign_path)

    df = pd.read_csv(data_path)
    assign = pd.read_csv(assign_path)

    # If assignments already include original columns, use them; else merge by order
    if set(["Index", "YearMonth"]).issubset(assign.columns):
        merged = assign.copy()
    else:
        merged = df.copy()
        merged["cluster"] = assign["cluster"].values

    features = [c for c in merged.select_dtypes(include=[np.number]).columns if c != "cluster"]

    global_mean = merged[features].mean()
    global_std = merged[features].std().replace(0, np.nan)

    profiles = merged.groupby("cluster")[features].mean()
    sizes = merged.groupby("cluster").size()
    diffs_z = (profiles - global_mean) / global_std

    (OUT_DIR / "cluster_means.csv").write_text(profiles.to_csv(index=True))
    (OUT_DIR / "cluster_mean_diffs_z.csv").write_text(diffs_z.to_csv(index=True))

    with open(OUT_DIR / "cluster_profiles.txt", "w", encoding="utf8") as f:
        f.write("Cluster summary (n per cluster):\n")
        for c, n in sizes.items():
            f.write(f"- Cluster {c}: n={n}\n")

        f.write("\nTop features per cluster (z-score vs global mean):\n")
        for c in sizes.index:
            f.write(f"\nCluster {c} (n={sizes.loc[c]}):\n")
            top = diffs_z.loc[c].abs().sort_values(ascending=False).head(8)
            for feat in top.index:
                val = diffs_z.loc[c, feat]
                f.write(f"  {feat}: {val:+.3f}\n")

    # Boxplots: pick top 6 features by max abs z-diff
    abs_max = diffs_z.abs().max(axis=0)
    top_feats = abs_max.sort_values(ascending=False).head(6).index.tolist()

    # Grid
    cols = 3
    rows = (len(top_feats) + cols - 1) // cols
    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, feat in enumerate(top_feats, 1):
        plt.subplot(rows, cols, i)
        sns.boxplot(x="cluster", y=feat, data=merged)
        plt.title(feat)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "boxplots_grid.pdf", dpi=180)
    plt.close()

    # Individual
    for feat in top_feats:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="cluster", y=feat, data=merged)
        plt.title(feat)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"boxplot_{feat}.pdf", dpi=180)
        plt.close()

    print("Wrote profiles and boxplots to", OUT_DIR)


if __name__ == "__main__":
    main()
