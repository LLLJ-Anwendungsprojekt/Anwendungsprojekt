"""
Erstellt eine analysebereite Datenbasis aus GEDEvent_v25_1.csv und indexData.csv.

Design-Entscheidungen
---------------------
- Einheit: eine Zeile pro Konfliktereignis (GED-Event).
- Markt-Features werden als aggregierte Reaktion über alle verfügbaren Indizes
  berechnet statt als Rohpreise → verhindert das Preisniveau-Cluster-Problem.
- Zeitfenster:
    - pre [-10..-1] Handelstage vor date_start  → Baseline-Marktlage
    - event [0]    Handelstag des Ereignisses
    - post [+1..+5] Handelstage nach date_start → kurzfristige Reaktion
- Markt-Features:
    - pre_return, event_return, post_return   (mean über Indizes)
    - pre_volatility, post_volatility          (Std der Tagesrenditen)
    - market_reaction  = post_return - pre_return
    - volatility_change = post_volatility - pre_volatility
    - n_indices_tracked                        (Datenverfügbarkeit)
- Konflikt-Features (keine IDs, keine Freitext-Spalten):
    - type_of_violence, deaths_best, deaths_civilians, deaths_unknown
    - number_of_sources, where_prec, event_clarity, date_prec
    - region (Label-Encoded), latitude, longitude
    - conflict_duration_days (date_end - date_start)
    - total_deaths_high, total_deaths_low
- Ziel-Spalten für KNN:
    - market_direction_5d  (1 = post_return >= 0, 0 = post_return < 0)
    - severity_class       (0=low/1=medium/2=high anhand deaths_best Quantilen)
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (default relative to repo root)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
GED_PATH   = ROOT / "GEDEvent_v25_1.csv"
IDX_PATH   = ROOT / "indexData.csv"
OUT_PATH   = ROOT / "Anwendungsprojekt" / "data" / "processed" / "conflict_market_features.csv"

# Features to keep from GED (no free-text, no duplicate-ID columns)
GED_KEEP = [
    "id",
    "year",
    "type_of_violence",
    "number_of_sources",
    "where_prec",
    "event_clarity",
    "date_prec",
    "latitude",
    "longitude",
    "country_id",
    "region",
    "deaths_a",
    "deaths_b",
    "deaths_civilians",
    "deaths_unknown",
    "best",        # best estimate total deaths
    "high",        # upper estimate total deaths
    "low",         # lower estimate total deaths
    "date_start",
    "date_end",
    "active_year",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_ged(path: Path) -> pd.DataFrame:
    log.info("GED laden …")
    df = pd.read_csv(path, usecols=GED_KEEP, encoding="latin1", low_memory=False)
    df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
    df["date_end"]   = pd.to_datetime(df["date_end"],   errors="coerce")
    df = df.dropna(subset=["date_start"])
    df["conflict_duration_days"] = (df["date_end"] - df["date_start"]).dt.days.fillna(0).astype(int)
    log.info("GED: %d Ereignisse", len(df))
    return df


def load_index(path: Path) -> pd.DataFrame:
    log.info("Indexdaten laden …")
    df = pd.read_csv(path, parse_dates=["Date"], low_memory=False)
    df = df.sort_values(["Index", "Date"])
    # Tagesrendite je Index (fill_method=None vermeidet FutureWarning)
    df["daily_return"] = df.groupby("Index")["Close"].pct_change(fill_method=None)
    # Intraday-Range als Volatilitätsmaß
    df["intraday_range"] = (df["High"] - df["Low"]) / df["Low"].replace(0, np.nan)
    df = df.dropna(subset=["daily_return"])
    log.info("Indexdaten: %d Zeilen, %d Indizes", len(df), df["Index"].nunique())
    return df


def market_window_features(
    event_date: pd.Timestamp,
    idx_df: pd.DataFrame,
    pre: int = 10,
    post: int = 5,
) -> dict:
    """Aggregiert Markt-Features in den definierten Zeitfenstern."""
    pre_mask  = (idx_df["Date"] >= event_date - pd.Timedelta(days=pre + 14)) & \
                (idx_df["Date"] <  event_date)
    post_mask = (idx_df["Date"] >  event_date) & \
                (idx_df["Date"] <= event_date + pd.Timedelta(days=post + 14))
    evt_mask  = idx_df["Date"] == event_date

    def agg(mask):
        sub = idx_df.loc[mask]
        if sub.empty:
            return None, None, 0
        # Für 'pre' wollen wir die letzten `pre` Handelstage, für 'post' die ersten `post`
        return (
            float(sub["daily_return"].mean()),
            float(sub["daily_return"].std(ddof=1)) if len(sub) > 1 else 0.0,
            int(sub["daily_return"].count()),
        )

    # Pre-window: letzten `pre` Handelstage
    pre_sub = idx_df.loc[pre_mask].copy()
    pre_sub = pre_sub.sort_values("Date")
    pre_sub = pre_sub.groupby("Index").tail(pre)
    pre_ret  = float(pre_sub["daily_return"].mean()) if not pre_sub.empty else np.nan
    pre_vol  = float(pre_sub["daily_return"].std(ddof=1)) if len(pre_sub) > 1 else np.nan

    # Post-window: erste `post` Handelstage
    post_sub = idx_df.loc[post_mask].copy()
    post_sub = post_sub.sort_values("Date")
    post_sub = post_sub.groupby("Index").head(post)
    post_ret = float(post_sub["daily_return"].mean()) if not post_sub.empty else np.nan
    post_vol = float(post_sub["daily_return"].std(ddof=1)) if len(post_sub) > 1 else np.nan

    # Event-Tag
    evt_sub  = idx_df.loc[evt_mask]
    evt_ret  = float(evt_sub["daily_return"].mean()) if not evt_sub.empty else np.nan

    n_indices = int(idx_df.loc[post_mask, "Index"].nunique())

    return {
        "pre_return":         pre_ret,
        "pre_volatility":     pre_vol,
        "event_return":       evt_ret,
        "post_return":        post_ret,
        "post_volatility":    post_vol,
        "market_reaction":    (post_ret - pre_ret) if np.isfinite(post_ret) and np.isfinite(pre_ret) else np.nan,
        "volatility_change":  (post_vol - pre_vol) if np.isfinite(post_vol) and np.isfinite(pre_vol) else np.nan,
        "n_indices_tracked":  n_indices,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build(ged_path: Path, idx_path: Path, out_path: Path, min_indices: int = 1) -> pd.DataFrame:
    ged = load_ged(ged_path)
    idx = load_index(idx_path)

    # Bestimme den Datumsbereich, den indexData abdeckt
    idx_min = idx["Date"].min()
    idx_max = idx["Date"].max()
    log.info("Indexdaten decken %s bis %s ab", idx_min.date(), idx_max.date())

    # Filtere GED auf den gleichen Zeitraum (mit etwas Puffer)
    ged_filtered = ged[
        (ged["date_start"] >= idx_min + pd.Timedelta(days=15)) &
        (ged["date_start"] <= idx_max)
    ].copy()
    log.info("GED nach Datums-Filter: %d Ereignisse", len(ged_filtered))

    # Markt-Features pro Ereignistag berechnen
    # Optimierung: nur einmalig pro einzigartigem Datum berechnen
    unique_dates = ged_filtered["date_start"].dt.normalize().unique()
    log.info("Unique Ereignistage: %d — berechne Markt-Features …", len(unique_dates))

    market_records = {}
    for i, d in enumerate(unique_dates):
        if (i + 1) % 500 == 0:
            log.info("  … %d / %d Tage verarbeitet", i + 1, len(unique_dates))
        market_records[d] = market_window_features(d, idx)

    # Map zurück auf GED
    ged_filtered["_date_norm"] = ged_filtered["date_start"].dt.normalize()
    market_df = pd.DataFrame.from_dict(market_records, orient="index")
    market_df.index.name = "_date_norm"
    market_df = market_df.reset_index()

    df = ged_filtered.merge(market_df, on="_date_norm", how="left").drop(columns=["_date_norm"])

    # Filtere auf Zeilen mit Marktdaten
    df = df[df["n_indices_tracked"] >= min_indices]
    log.info("Nach Markt-Filter (>= %d Index): %d Ereignisse", min_indices, len(df))

    # Feature Engineering
    df["total_deaths"] = df[["deaths_a", "deaths_b", "deaths_civilians", "deaths_unknown"]].sum(axis=1)
    df["lethality_ratio"] = df["best"] / df["number_of_sources"].replace(0, np.nan)

    # Region Label-Encoding
    le = LabelEncoder()
    df["region_enc"] = le.fit_transform(df["region"].fillna("Unknown"))
    log.info("Regionen: %s", dict(zip(le.classes_, le.transform(le.classes_))))

    # Ziel-Variable für KNN: Marktrichtung in 5 Handelstagen
    df["market_direction_5d"] = (df["post_return"] >= 0).astype(int)

    # Ziel-Variable: Schweregrad des Konflikts (Quantile-basiert)
    deaths = df["best"].fillna(0)
    q33 = deaths.quantile(0.33)
    q67 = deaths.quantile(0.67)
    df["severity_class"] = pd.cut(
        deaths,
        bins=[-1, q33, q67, float("inf")],
        labels=[0, 1, 2],
    ).astype(int)

    log.info("Klassen-Verteilung market_direction_5d:\n%s", df["market_direction_5d"].value_counts().to_string())
    log.info("Klassen-Verteilung severity_class:\n%s", df["severity_class"].value_counts().to_string())

    # Finale Spalten-Sortierung
    id_cols     = ["id", "year", "date_start", "date_end"]
    conflict_f  = [
        "type_of_violence", "number_of_sources", "where_prec", "event_clarity",
        "date_prec", "latitude", "longitude", "region", "region_enc", "country_id",
        "deaths_a", "deaths_b", "deaths_civilians", "deaths_unknown",
        "best", "high", "low", "total_deaths", "lethality_ratio",
        "conflict_duration_days", "active_year",
    ]
    market_f    = [
        "pre_return", "pre_volatility", "event_return",
        "post_return", "post_volatility",
        "market_reaction", "volatility_change", "n_indices_tracked",
    ]
    target_cols = ["market_direction_5d", "severity_class"]

    all_cols = id_cols + conflict_f + market_f + target_cols
    df = df[[c for c in all_cols if c in df.columns]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Datenbasis gespeichert: %s (%d Zeilen, %d Spalten)", out_path, len(df), df.shape[1])

    # Kurze Übersicht
    print("\n=== Datenbasis-Übersicht ===")
    print(f"Zeilen: {len(df):,}")
    print(f"Spalten: {df.shape[1]}")
    print(f"\nKonflikt-Features ({len(conflict_f)}):\n  {', '.join([c for c in conflict_f if c in df])}")
    print(f"\nMarkt-Features ({len(market_f)}):\n  {', '.join([c for c in market_f if c in df])}")
    print(f"\nZiel-Spalten: {target_cols}")
    print(f"\nMissing-Werte:\n{df[market_f].isna().sum().to_string()}")
    print(f"\nSeverity-Klassen:\n{df['severity_class'].value_counts().sort_index().to_string()}")
    print(f"\nMarktrichtung:\n{df['market_direction_5d'].value_counts().sort_index().to_string()}")
    print(f"\nZeitraum: {df['date_start'].min().date()} – {df['date_start'].max().date()}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Erstellt analysebereite Datenbasis aus GED + Indexdaten")
    p.add_argument("--ged-path",     default=str(GED_PATH),  help="Pfad zu GEDEvent_v25_1.csv")
    p.add_argument("--idx-path",     default=str(IDX_PATH),  help="Pfad zu indexData.csv")
    p.add_argument("--out-path",     default=str(OUT_PATH),  help="Ausgabepfad")
    p.add_argument("--min-indices",  default=1, type=int,    help="Mindestanzahl Indizes für Markt-Features")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(
        ged_path=Path(args.ged_path),
        idx_path=Path(args.idx_path),
        out_path=Path(args.out_path),
        min_indices=args.min_indices,
    )
