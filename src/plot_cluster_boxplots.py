"""Generate boxplots for selected features per cluster.

By default selects the top 6 features from
`results/kmeans_test/cluster_mean_diffs_z.csv` (highest absolute z-difference)
and creates a grid figure plus individual PNGs in `results/kmeans_test/`.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


OUT_DIR = Path("results/kmeans_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_merged():
    df = pd.read_csv("data/processed/stocks_gpr_features.csv")
    assign = pd.read_csv(OUT_DIR / "kmeans_cluster_assignments.csv")
    if set(["Index", "YearMonth"]).issubset(assign.columns):
        merged = assign.copy()
    else:
        merged = df.copy()
        merged["cluster"] = assign["cluster"].values
    return merged


def select_top_features(n=6):
    diffs_path = OUT_DIR / "cluster_mean_diffs_z.csv"
    if not diffs_path.exists():
        return None
    diffs = pd.read_csv(diffs_path, index_col=0)
    # absolute max across clusters
    abs_max = diffs.abs().max(axis=0)
    top = abs_max.sort_values(ascending=False).head(n).index.tolist()
    return top


def plot_grid(merged, features, out_path):
    k = merged["cluster"].nunique()
    n = len(features)
    cols = 3
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, feat in enumerate(features, 1):
        plt.subplot(rows, cols, i)
        sns.boxplot(x="cluster", y=feat, data=merged, palette="tab10")
        plt.title(feat)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_individual(merged, features, out_dir: Path):
    for feat in features:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="cluster", y=feat, data=merged, palette="tab10")
        plt.title(feat)
        p = out_dir / f"boxplot_{feat}.pdf"
        plt.tight_layout()
        plt.savefig(p, dpi=180)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="*", default=None, help="Features to plot")
    parser.add_argument("--top", type=int, default=6, help="Top N features to auto-select")
    args = parser.parse_args()

    merged = load_merged()
    if args.features:
        features = args.features
    else:
        features = select_top_features(n=args.top)
        if not features:
            # fallback numeric features except cluster
            features = [c for c in merged.select_dtypes(include=[np.number]).columns if c != "cluster"][:6]

    print("Selected features:", features)
    grid_path = OUT_DIR / "boxplots_grid.pdf"
    plot_grid(merged, features, grid_path)
    plot_individual(merged, features, OUT_DIR)
    print("Saved boxplots to", grid_path)


if __name__ == "__main__":
    main()
