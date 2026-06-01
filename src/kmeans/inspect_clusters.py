"""Inspect K-Means cluster assignments and produce concise profiles.

Writes:
- results/kmeans_test/cluster_means.csv
- results/kmeans_test/cluster_mean_diffs_z.csv
- results/kmeans_test/cluster_profiles.txt

Also prints a short summary to stdout.
"""

from pathlib import Path
import pandas as pd
import numpy as np


OUT_DIR = Path("results/kmeans_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/processed/stocks_gpr_features.csv")
assign = pd.read_csv(OUT_DIR / "kmeans_cluster_assignments.csv")

# Merge: prefer explicit keys if present
if set(["Index", "YearMonth"]).issubset(assign.columns):
    merged = assign.copy()
else:
    # fallback: assume same row ordering
    merged = df.copy()
    merged["cluster"] = assign["cluster"].values

features = [c for c in merged.select_dtypes(include=[np.number]).columns if c not in ("cluster",)]

global_mean = merged[features].mean()
global_std = merged[features].std().replace(0, np.nan)

profiles = merged.groupby("cluster")[features].mean()
sizes = merged.groupby("cluster").size()

diffs_z = (profiles - global_mean) / global_std

# Save outputs
profiles.to_csv(OUT_DIR / "cluster_means.csv")
diffs_z.to_csv(OUT_DIR / "cluster_mean_diffs_z.csv")

with open(OUT_DIR / "cluster_profiles.txt", "w", encoding="utf8") as f:
    f.write(f"Cluster summary (n per cluster):\n")
    for c, n in sizes.items():
        f.write(f"- Cluster {c}: n={n}\n")

    f.write("\nTop features per cluster (z-score vs global mean):\n")
    for c in sizes.index:
        f.write(f"\nCluster {c} (n={sizes.loc[c]}):\n")
        top = diffs_z.loc[c].abs().sort_values(ascending=False).head(5)
        for feat in top.index:
            val = diffs_z.loc[c, feat]
            f.write(f"  {feat}: {val:+.3f}\n")

print("Wrote cluster summaries to:")
print(OUT_DIR / "cluster_profiles.txt")
print(OUT_DIR / "cluster_means.csv")
print(OUT_DIR / "cluster_mean_diffs_z.csv")
