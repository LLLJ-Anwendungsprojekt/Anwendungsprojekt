"""Erstellt eine indexzentrierte Datenbasis aus indexData.csv und GEDEvent_v25_1.csv.

Die Basiseinheit ist ein Index-Handelstag. Konfliktdaten werden ueber das Datum
exakt auf denselben Kalendertag gematcht und pro Tag aggregiert. Es werden nur
GED-Ereignisse mit date_prec == 1 verwendet.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
GED_PATH = ROOT / "GEDEvent_v25_1.csv"
IDX_PATH = ROOT / "indexData.csv"
OUT_PATH = ROOT / "Anwendungsprojekt" / "data" / "processed" / "index_conflict_features.csv"

GED_KEEP = [
    "id",
    "date_start",
    "date_end",
    "date_prec",
    "type_of_violence",
    "number_of_sources",
    "country_id",
    "region",
    "event_clarity",
    "where_prec",
    "deaths_a",
    "deaths_b",
    "deaths_civilians",
    "deaths_unknown",
    "best",
    "high",
    "low",
]


def load_index_data(path: Path) -> pd.DataFrame:
    log.info("Lade Indexdaten aus %s", path)
    df = pd.read_csv(path, parse_dates=["Date"], low_memory=False)
    df = df.sort_values(["Index", "Date"]).copy()
    df["daily_return"] = df.groupby("Index")["Close"].pct_change(fill_method=None)
    df["intraday_range"] = (df["High"] - df["Low"]) / df["Low"].replace(0, np.nan)
    df["volume_change"] = df.groupby("Index")["Volume"].pct_change(fill_method=None)
    df["date"] = df["Date"].dt.normalize()
    return df


def load_ged_data(path: Path) -> pd.DataFrame:
    log.info("Lade GED-Daten aus %s", path)
    df = pd.read_csv(path, usecols=GED_KEEP, encoding="latin1", low_memory=False)
    df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
    df["date_end"] = pd.to_datetime(df["date_end"], errors="coerce")

    numeric_columns = [
        "date_prec",
        "type_of_violence",
        "number_of_sources",
        "country_id",
        "event_clarity",
        "where_prec",
        "deaths_a",
        "deaths_b",
        "deaths_civilians",
        "deaths_unknown",
        "best",
        "high",
        "low",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["date_start"])
    df = df[df["date_prec"] == 1].copy()
    df.loc[df["number_of_sources"] < 0, "number_of_sources"] = np.nan
    df["date"] = df["date_start"].dt.normalize()
    df["conflict_duration_days"] = (
        (df["date_end"] - df["date_start"]).dt.days.fillna(0).clip(lower=0).astype(int)
    )
    df["total_deaths"] = df[
        ["deaths_a", "deaths_b", "deaths_civilians", "deaths_unknown"]
    ].fillna(0).sum(axis=1)
    return df


def aggregate_conflicts_by_date(ged_df: pd.DataFrame) -> pd.DataFrame:
    if ged_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "conflict_count",
                "fatalities_best_sum",
                "fatalities_high_sum",
                "fatalities_low_sum",
                "civilian_deaths_sum",
                "unknown_deaths_sum",
                "total_deaths_sum",
                "avg_sources",
                "avg_conflict_duration_days",
                "countries_affected",
                "regions_affected",
                "share_high_clarity",
                "share_high_where_prec",
                "violence_type_count",
            ]
        )

    grouped = ged_df.groupby("date", dropna=False)
    summary = grouped.agg(
        conflict_count=("id", "count"),
        fatalities_best_sum=("best", "sum"),
        fatalities_high_sum=("high", "sum"),
        fatalities_low_sum=("low", "sum"),
        civilian_deaths_sum=("deaths_civilians", "sum"),
        unknown_deaths_sum=("deaths_unknown", "sum"),
        total_deaths_sum=("total_deaths", "sum"),
        avg_sources=("number_of_sources", "mean"),
        avg_conflict_duration_days=("conflict_duration_days", "mean"),
        countries_affected=("country_id", pd.Series.nunique),
        regions_affected=("region", pd.Series.nunique),
        violence_type_count=("type_of_violence", pd.Series.nunique),
    ).reset_index()

    precision_shares = grouped.agg(
        share_high_clarity=("event_clarity", lambda values: float((values == 1).mean())),
        share_high_where_prec=("where_prec", lambda values: float((values == 1).mean())),
    ).reset_index()

    violence_counts = (
        pd.crosstab(ged_df["date"], ged_df["type_of_violence"])
        .rename(columns=lambda value: f"type_of_violence_{int(value)}_count")
        .reset_index()
    )

    result = summary.merge(precision_shares, on="date", how="left")
    result = result.merge(violence_counts, on="date", how="left")
    count_cols = [col for col in result.columns if col.endswith("_count")]
    result[count_cols] = result[count_cols].fillna(0).astype(int)
    return result


def build_index_conflict_dataset(
    ged_path: Path,
    idx_path: Path,
    out_path: Path,
) -> pd.DataFrame:
    idx_df = load_index_data(idx_path)
    ged_df = load_ged_data(ged_path)

    log.info("Index-Zeilen: %d", len(idx_df))
    log.info("GED-Ereignisse mit date_prec = 1: %d", len(ged_df))

    conflict_daily = aggregate_conflicts_by_date(ged_df)
    merged = idx_df.merge(conflict_daily, on="date", how="left")

    zero_fill_columns = [
        "conflict_count",
        "fatalities_best_sum",
        "fatalities_high_sum",
        "fatalities_low_sum",
        "civilian_deaths_sum",
        "unknown_deaths_sum",
        "total_deaths_sum",
        "countries_affected",
        "regions_affected",
        "violence_type_count",
    ]
    zero_fill_columns.extend([col for col in merged.columns if col.endswith("_count")])
    zero_fill_columns = list(dict.fromkeys(zero_fill_columns))

    ratio_fill_columns = [
        "avg_sources",
        "avg_conflict_duration_days",
        "share_high_clarity",
        "share_high_where_prec",
    ]

    existing_zero_fill = [col for col in zero_fill_columns if col in merged.columns]
    existing_ratio_fill = [col for col in ratio_fill_columns if col in merged.columns]

    merged[existing_zero_fill] = merged[existing_zero_fill].fillna(0)
    int_columns = [col for col in existing_zero_fill if merged[col].dtype.kind in "fi"]
    merged[int_columns] = merged[int_columns].astype(int)
    merged[existing_ratio_fill] = merged[existing_ratio_fill].fillna(0.0)

    merged["has_conflict"] = (merged["conflict_count"] > 0).astype(int)

    ordered_columns = [
        "Index",
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
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
    merged = merged[[col for col in ordered_columns if col in merged.columns]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    log.info("Neue Datenbasis gespeichert: %s", out_path)
    log.info("Zeilen: %d | Spalten: %d", len(merged), merged.shape[1])
    log.info("Indexe: %d", merged["Index"].nunique())
    log.info("Tage mit Konfliktmatch: %d", int(merged["has_conflict"].sum()))
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Erstellt eine indexbasierte Datenbasis mit tagesgenau gematchten Konflikten"
    )
    parser.add_argument("--ged-path", default=str(GED_PATH), help="Pfad zu GEDEvent_v25_1.csv")
    parser.add_argument("--idx-path", default=str(IDX_PATH), help="Pfad zu indexData.csv")
    parser.add_argument("--out-path", default=str(OUT_PATH), help="Pfad fuer die Ausgabe-CSV")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_index_conflict_dataset(
        ged_path=Path(args.ged_path),
        idx_path=Path(args.idx_path),
        out_path=Path(args.out_path),
    )