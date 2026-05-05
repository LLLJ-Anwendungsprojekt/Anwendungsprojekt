"""K-Means-Subclustering auf Cluster 1 der NEU_DB-Datenbasis.

Cluster 1 enthaelt ausschliesslich Handelstage mit mind. einem tagesgenauen
Konfliktereignis (date_prec == 1). Innerhalb dieser Gruppe werden
Schweregrads- und Volatilitaetsmuster weiter differenziert.
"""

from pathlib import Path

from kmeans_analyse import KMeansAnalyzer, setup_logging


FEATURE_COLUMNS_C1 = [
    "daily_return",
    "intraday_range",
    "volume_change",
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
    assignments_path = (
        project_root
        / "results_NEU_DB"
        / "kmeans_cluster_assignments_NEU_DB.csv"
    )
    output_dir = project_root / "results_NEU_DB"
    log_path = output_dir / "kmeans_analysis_NEU_DB_C1.log"

    setup_logging(str(log_path))

    import pandas as pd

    df_full = pd.read_csv(assignments_path)
    df_c1 = df_full[df_full["cluster"] == 1].copy().reset_index(drop=True)
    import logging
    logging.getLogger(__name__).info(
        "Cluster-1-Teilmenge: %d Handelstage (von %d gesamt)",
        len(df_c1),
        len(df_full),
    )

    # Zwischenspeicher damit KMeansAnalyzer.load_data() genutzt werden kann
    import tempfile, os
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as tmp:
        df_c1.to_csv(tmp, index=False)
        tmp_path = tmp.name

    try:
        analyzer = KMeansAnalyzer(random_state=42)
        results = analyzer.run_pipeline(
            data_path=tmp_path,
            output_dir=str(output_dir),
            include_columns=FEATURE_COLUMNS_C1,
            output_prefix="_NEU_DB_C1",
            k_min=2,
            k_max=8,
        )
    finally:
        os.unlink(tmp_path)

    print("\nK-Means NEU_DB_C1 abgeschlossen")
    print(f"Bestes k: {results.best_k}")
    print(f"Silhouette: {results.silhouette:.4f}")
    print(f"Inertia: {results.inertia:.2f}")
    print(f"Samples: {results.n_samples}")
    print(f"Features: {results.n_features}")


if __name__ == "__main__":
    main()
