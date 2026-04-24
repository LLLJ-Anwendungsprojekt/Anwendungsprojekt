import argparse
import os
from typing import List

import numpy as np
import pandas as pd


def pooled_std(series_a: pd.Series, series_b: pd.Series) -> float:
    a = series_a.dropna()
    b = series_b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    denom = (len(a) + len(b) - 2)
    if denom <= 0:
        return np.nan
    pooled_var = ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / denom
    return float(np.sqrt(pooled_var)) if pooled_var >= 0 else np.nan


def eta_squared(feature: pd.Series, labels: pd.Series) -> float:
    valid = pd.DataFrame({"x": feature, "cluster": labels}).dropna()
    if valid.empty:
        return np.nan

    grand_mean = valid["x"].mean()
    ss_total = ((valid["x"] - grand_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0

    grouped = valid.groupby("cluster")["x"]
    ss_between = sum(len(group) * (group.mean() - grand_mean) ** 2 for _, group in grouped)
    return float(ss_between / ss_total)


def build_profiles(df: pd.DataFrame, cluster_col: str, numeric_cols: List[str]) -> pd.DataFrame:
    profile_rows = []
    grouped = df.groupby(cluster_col)

    for cluster_id, gdf in grouped:
        for feature in numeric_cols:
            s = gdf[feature]
            profile_rows.append(
                {
                    "cluster": cluster_id,
                    "feature": feature,
                    "count": int(s.notna().sum()),
                    "mean": float(s.mean()) if s.notna().any() else np.nan,
                    "median": float(s.median()) if s.notna().any() else np.nan,
                    "std": float(s.std(ddof=1)) if s.notna().sum() > 1 else np.nan,
                    "min": float(s.min()) if s.notna().any() else np.nan,
                    "max": float(s.max()) if s.notna().any() else np.nan,
                }
            )

    return pd.DataFrame(profile_rows)


def build_driver_table(df: pd.DataFrame, cluster_col: str, numeric_cols: List[str]) -> pd.DataFrame:
    cluster_ids = sorted(df[cluster_col].dropna().unique())
    rows = []

    for feature in numeric_cols:
        row = {
            "feature": feature,
            "eta_squared": eta_squared(df[feature], df[cluster_col]),
        }

        means = df.groupby(cluster_col)[feature].mean()
        medians = df.groupby(cluster_col)[feature].median()

        for cid in cluster_ids:
            row[f"mean_cluster_{cid}"] = float(means.get(cid, np.nan))
            row[f"median_cluster_{cid}"] = float(medians.get(cid, np.nan))

        if len(cluster_ids) == 2:
            c0, c1 = cluster_ids
            s0 = df[df[cluster_col] == c0][feature]
            s1 = df[df[cluster_col] == c1][feature]
            ps = pooled_std(s0, s1)
            mean_diff = float(s1.mean() - s0.mean())
            row["mean_diff_c1_minus_c0"] = mean_diff
            row["cohen_d_abs"] = float(abs(mean_diff / ps)) if ps and ps != 0 else np.nan

        rows.append(row)

    driver_df = pd.DataFrame(rows)

    if "cohen_d_abs" in driver_df.columns:
        return driver_df.sort_values(["cohen_d_abs", "eta_squared"], ascending=False)
    return driver_df.sort_values("eta_squared", ascending=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster-Profiling und Treiberanalyse")
    parser.add_argument(
        "--input-path",
        default="results/kmeans_cluster_assignments.csv",
        help="CSV mit K-Means Clusterzuweisungen",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Ausgabeordner fuer Tabellen",
    )
    parser.add_argument(
        "--cluster-col",
        default="cluster",
        help="Name der Cluster-Spalte",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Anzahl Top-Treiber fuer Konsolen-Output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_path)

    if args.cluster_col not in df.columns:
        raise ValueError(f"Cluster-Spalte nicht gefunden: {args.cluster_col}")

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != args.cluster_col]
    if not numeric_cols:
        raise ValueError("Keine numerischen Features fuer Profiling gefunden.")

    os.makedirs(args.output_dir, exist_ok=True)

    profile_df = build_profiles(df, args.cluster_col, numeric_cols)
    profile_path = os.path.join(args.output_dir, "cluster_profile_summary.csv")
    profile_df.to_csv(profile_path, index=False)

    driver_df = build_driver_table(df, args.cluster_col, numeric_cols)
    drivers_path = os.path.join(args.output_dir, "cluster_top_drivers.csv")
    driver_df.to_csv(drivers_path, index=False)

    cluster_sizes = df[args.cluster_col].value_counts().sort_index()
    print("Clustergroessen:")
    for cid, size in cluster_sizes.items():
        print(f"  Cluster {cid}: {size}")

    print("\nTop Treiber:")
    display_cols = ["feature", "eta_squared"]
    if "cohen_d_abs" in driver_df.columns:
        display_cols.append("cohen_d_abs")
    print(driver_df[display_cols].head(args.top_n).to_string(index=False))

    print(f"\nGespeichert: {profile_path}")
    print(f"Gespeichert: {drivers_path}")


if __name__ == "__main__":
    main()
