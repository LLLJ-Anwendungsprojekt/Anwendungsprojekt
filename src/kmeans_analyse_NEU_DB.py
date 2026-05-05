"""Startskript fuer K-Means auf der NEU_DB-Datenbasis."""

from pathlib import Path

from kmeans_analyse import KMeansAnalyzer, setup_logging


FEATURE_COLUMNS_NEU_DB = [
    "daily_return",
    "intraday_range",
    "volume_change",
    "has_conflict",
    "conflict_count",
    "countries_affected",
    "regions_affected",
    "violence_type_count",
    "type_of_violence_1_count",
    "type_of_violence_2_count",
    "type_of_violence_3_count",
    "fatalities_best_sum",
    "fatalities_high_sum",
    "fatalities_low_sum",
    "civilian_deaths_sum",
    "unknown_deaths_sum",
    "total_deaths_sum",
    "avg_sources",
    "avg_conflict_duration_days",
    "share_high_clarity",
    "share_high_where_prec",
]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "processed" / "index_conflict_features_NEU_DB.csv"
    output_dir = project_root / "results_NEU_DB"
    log_path = output_dir / "kmeans_analysis_NEU_DB.log"

    setup_logging(str(log_path))

    analyzer = KMeansAnalyzer(random_state=42)
    results = analyzer.run_pipeline(
        data_path=str(data_path),
        output_dir=str(output_dir),
        include_columns=FEATURE_COLUMNS_NEU_DB,
        output_prefix="_NEU_DB",
        exclude_columns=["country_id", "latitude", "longitude"],
        k_min=2,
        k_max=10,
    )

    print("\nK-Means NEU_DB abgeschlossen")
    print(f"Bestes k: {results.best_k}")
    print(f"Silhouette: {results.silhouette:.4f}")
    print(f"Inertia: {results.inertia:.2f}")
    print(f"Samples: {results.n_samples}")
    print(f"Features: {results.n_features}")


if __name__ == "__main__":
    main()