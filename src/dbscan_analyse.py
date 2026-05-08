"""
DBSCAN-MARKTANOMALIEANALYSE MIT KONTROLLGRUPPE

Ziel:
Es wird geprüft, ob ungewöhnliche Marktreaktionen häufiger an Handelstagen
auftreten, denen geopolitische Konfliktereignisse zugeordnet wurden.

Verwendete Dateien:
1. data/processed/indexData.csv
   Erwartete Spalten:
   Index, Date, Open, High, Low, Close, Adj Close, Volume

2. data/processed/GDEEvent_v25_1.csv
   Erwartete Spalten:
   id, date_start, date_end, best, high, low, latitude, longitude usw.

Methodische Logik:
1. DBSCAN wird ausschließlich auf Marktvariablen aus indexData.csv angewendet.
2. Konfliktinformationen aus GDEEvent_v25_1.csv werden NICHT für DBSCAN verwendet.
3. DBSCAN-Noise-Punkte mit Label -1 werden als Marktanomalien interpretiert.
4. Danach wird geprüft, ob Marktanomalien häufiger an Konflikttagen auftreten.

Wichtig:
Diese Analyse zeigt statistische Zusammenhänge, aber keine Kausalität.
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu

warnings.filterwarnings("ignore")


# =============================================================================
# KONFIGURATION
# =============================================================================

STOCK_DATA_PATH = "data/processed/indexData.csv"
CONFLICT_DATA_PATH = "data/processed/GEDEvent_v25_1.csv"

RESULTS_DIR = "results"
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")

RANDOM_STATE = 42

MIN_DAYS_PER_INDEX = 500

# DBSCAN-Grid
MIN_SAMPLES_VALUES = [10, 20, 30, 50]
EPS_PERCENTILES = [80, 85, 90, 92, 95, 97]

# Zielbereich für Marktanomalien
TARGET_NOISE_SHARE = 5.0
MIN_NOISE_SHARE = 1.0
MAX_NOISE_SHARE = 12.0

# Konfliktfenster in Handelstagen
CONFLICT_WINDOW_DAYS = 3

# High-Intensity-Definition anhand der Konflikttodeszahlen
HIGH_INTENSITY_PERCENTILE = 90

# Volume nur verwenden, wenn es brauchbar ist
MAX_ALLOWED_VOLUME_MISSING_SHARE = 0.40
MIN_VOLUME_NONZERO_SHARE = 0.20

# Maximale Entfernung zwischen Konfliktdatum und nächstem Handelstag
MAX_MAPPING_GAP_DAYS = 7

BASE_MARKET_FEATURES = [
    "return_1d",
    "abs_return_1d",
    "return_z_20",
    "volatility_20",
    "intraday_range",
    "open_close_return",
]

VOLUME_FEATURE = "volume_z_20"


# =============================================================================
# SETUP
# =============================================================================

def create_directories():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)


def clean_column_names(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.replace("ï»¿", "", regex=False)
        .str.strip()
    )
    return df


def find_file(preferred_path, pattern):
    if os.path.exists(preferred_path):
        return preferred_path

    candidates = glob.glob(pattern)

    if not candidates:
        raise FileNotFoundError(
            f"Datei nicht gefunden: {preferred_path}\n"
            f"Auch keine passende Datei mit Muster gefunden: {pattern}"
        )

    print("[WARNUNG] Bevorzugter Pfad nicht gefunden.")
    print("[INFO] Nutze stattdessen:", candidates[0])
    return candidates[0]


def safe_rate(numerator, denominator):
    if denominator == 0:
        return np.nan
    return numerator / denominator


def add_fdr_bh(df, p_col, out_col):
    """
    Benjamini-Hochberg-FDR-Korrektur für mehrere Tests.
    """
    df = df.copy()

    if p_col not in df.columns or df.empty:
        df[out_col] = np.nan
        return df

    p = df[p_col].astype(float)
    valid = p.notna()

    df[out_col] = np.nan

    if valid.sum() == 0:
        return df

    p_valid = p[valid].values
    order = np.argsort(p_valid)
    ranked_p = p_valid[order]
    m = len(ranked_p)

    adjusted = np.empty(m)
    prev = 1.0

    for i in range(m - 1, -1, -1):
        rank = i + 1
        value = ranked_p[i] * m / rank
        prev = min(prev, value)
        adjusted[i] = prev

    adjusted_original_order = np.empty(m)
    adjusted_original_order[order] = adjusted

    df.loc[valid, out_col] = np.minimum(adjusted_original_order, 1.0)

    return df


# =============================================================================
# DATEN LADEN
# =============================================================================

def load_stock_data():
    stock_path = find_file(
        STOCK_DATA_PATH,
        "data/processed/*indexData*.csv"
    )

    df = pd.read_csv(stock_path)
    df = clean_column_names(df)

    print("\n" + "=" * 80)
    print("INDEXDATEN GELADEN")
    print("=" * 80)
    print("Datei:", stock_path)
    print("Zeilen:", len(df))
    print("Spalten:", df.columns.tolist())

    required = ["Index", "Date", "Close"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"In indexData fehlen Pflichtspalten: {missing}\n"
            "Erwartet werden mindestens: Index, Date, Close"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    df = df.dropna(subset=["Index", "Date", "Close"]).copy()

    # Für Renditen wird Adj Close bevorzugt, falls ausreichend vorhanden.
    adj_available_share = df["Adj Close"].notna().mean()

    if adj_available_share >= 0.80:
        df["price_for_return"] = df["Adj Close"]
        price_col_used = "Adj Close"
    else:
        df["price_for_return"] = df["Close"]
        price_col_used = "Close"

    df = df.dropna(subset=["price_for_return"]).copy()
    df = df.sort_values(["Index", "Date"]).reset_index(drop=True)

    counts = df["Index"].value_counts()
    valid_indexes = counts[counts >= MIN_DAYS_PER_INDEX].index.tolist()

    df = df[df["Index"].isin(valid_indexes)].copy()

    print("\nNach Filter auf ausreichend lange Indexhistorien:")
    print("Zeilen:", len(df))
    print("Indizes:", sorted(df["Index"].unique().tolist()))
    print("Preisvariable für Renditen:", price_col_used)

    if len(df) < 1000:
        raise ValueError(
            "Zu wenige Marktdaten nach Filterung. DBSCAN wäre nicht sinnvoll."
        )

    return df


def load_conflict_data():
    conflict_path = find_file(
        CONFLICT_DATA_PATH,
        "data/processed/*DEEvent*.csv"
    )

    df = pd.read_csv(conflict_path)
    df = clean_column_names(df)

    print("\n" + "=" * 80)
    print("GDE-KONFLIKTDATEN GELADEN")
    print("=" * 80)
    print("Datei:", conflict_path)
    print("Zeilen:", len(df))
    print("Spalten:", df.columns.tolist())

    required = ["date_start", "best"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"In GDEEvent fehlen Pflichtspalten: {missing}\n"
            "Für diese Analyse werden mindestens date_start und best benötigt."
        )

    df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")

    if "date_end" in df.columns:
        df["date_end"] = pd.to_datetime(df["date_end"], errors="coerce")
    else:
        df["date_end"] = df["date_start"]

    df["total_deaths"] = pd.to_numeric(df["best"], errors="coerce").fillna(0)
    df["total_deaths"] = df["total_deaths"].clip(lower=0)

    if "high" in df.columns:
        df["deaths_high"] = pd.to_numeric(df["high"], errors="coerce").fillna(df["total_deaths"])
    else:
        df["deaths_high"] = df["total_deaths"]

    if "low" in df.columns:
        df["deaths_low"] = pd.to_numeric(df["low"], errors="coerce").fillna(df["total_deaths"])
    else:
        df["deaths_low"] = df["total_deaths"]

    df["conflict_duration_days"] = (
        df["date_end"] - df["date_start"]
    ).dt.days

    df["conflict_duration_days"] = (
        df["conflict_duration_days"]
        .fillna(0)
        .clip(lower=0)
    )

    # zusätzliche Intensitätsmaße
    df["log_deaths"] = np.log1p(df["total_deaths"])
    df["death_uncertainty"] = (df["deaths_high"] - df["deaths_low"]).clip(lower=0)

    df = df.dropna(subset=["date_start"]).copy()

    print("Gültige Konfliktereignisse:", len(df))
    print("Zeitraum:", df["date_start"].min(), "bis", df["date_start"].max())
    print("Todesvariable: best -> total_deaths")

    if len(df) < 100:
        print("[WARNUNG] Sehr wenige Konfliktereignisse. Tests können instabil sein.")

    return df


# =============================================================================
# DATEN-PRÜFUNG
# =============================================================================

def check_input_usability(stock_df, conflict_df):
    """
    Prüft hart, ob die beiden CSVs für diese DBSCAN-Analyse geeignet sind.
    """
    print("\n" + "=" * 80)
    print("DATENPRÜFUNG: SIND DIE CSVs FÜR DIE ANALYSE GEEIGNET?")
    print("=" * 80)

    rows = []

    # Stock-Daten
    n_stock = len(stock_df)
    n_indexes = stock_df["Index"].nunique()
    date_min = stock_df["Date"].min()
    date_max = stock_df["Date"].max()
    close_missing = stock_df["price_for_return"].isna().mean()
    close_unique = stock_df["price_for_return"].nunique()

    stock_usable = (
        n_stock >= 1000
        and n_indexes >= 1
        and close_missing < 0.05
        and close_unique > 100
    )

    rows.append({
        "dataset": "indexData",
        "usable_for_dbscan_market_anomaly": stock_usable,
        "n_rows": n_stock,
        "n_unique_indexes": n_indexes,
        "date_min": date_min,
        "date_max": date_max,
        "main_check": "Muss vollständige Handelstage und numerische Preise enthalten.",
        "comment": (
            "Geeignet für DBSCAN auf Marktvariablen."
            if stock_usable
            else "Nicht sauber geeignet: zu wenige Daten, fehlende Preise oder keine Preisvariation."
        ),
    })

    # Konflikt-Daten
    n_conflict = len(conflict_df)
    conflict_date_min = conflict_df["date_start"].min()
    conflict_date_max = conflict_df["date_start"].max()
    deaths_nonmissing = conflict_df["total_deaths"].notna().mean()

    conflict_usable = (
        n_conflict >= 100
        and deaths_nonmissing > 0.95
        and conflict_df["date_start"].notna().mean() > 0.95
    )

    rows.append({
        "dataset": "GDEEvent_v25_1",
        "usable_for_dbscan_market_anomaly": False,
        "n_rows": n_conflict,
        "n_unique_indexes": np.nan,
        "date_min": conflict_date_min,
        "date_max": conflict_date_max,
        "main_check": "Wird nicht für DBSCAN genutzt, sondern als Konfliktlabel nach DBSCAN.",
        "comment": (
            "Geeignet als Konfliktlabel und Intensitätsquelle."
            if conflict_usable
            else "Nur eingeschränkt geeignet: zu wenige Ereignisse oder fehlende Datums-/Todeswerte."
        ),
    })

    report = pd.DataFrame(rows)

    print(report.to_string(index=False))

    output_path = os.path.join(RESULTS_DIR, "data_usability_report.csv")
    report.to_csv(output_path, index=False)

    if not stock_usable:
        raise ValueError(
            "indexData.csv ist für DBSCAN auf Marktvariablen nicht sauber geeignet. "
            "Bitte Datenqualität prüfen."
        )

    if not conflict_usable:
        print(
            "[WARNUNG] GDEEvent_v25_1.csv ist nur eingeschränkt als Konfliktlabel geeignet. "
            "Die Analyse läuft weiter, aber Interpretation vorsichtig."
        )

    return report


# =============================================================================
# MARKTFEATURES
# =============================================================================

def create_market_features(stock_df):
    df = stock_df.copy()
    df = df.sort_values(["Index", "Date"]).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("MARKTFEATURES ERSTELLEN")
    print("=" * 80)

    df["return_1d"] = df.groupby("Index")["price_for_return"].pct_change()
    df["abs_return_1d"] = df["return_1d"].abs()

    df["volatility_20"] = (
        df.groupby("Index")["return_1d"]
        .rolling(window=20, min_periods=10)
        .std()
        .reset_index(level=0, drop=True)
    )

    rolling_mean = (
        df.groupby("Index")["return_1d"]
        .rolling(window=20, min_periods=10)
        .mean()
        .reset_index(level=0, drop=True)
    )

    rolling_std = (
        df.groupby("Index")["return_1d"]
        .rolling(window=20, min_periods=10)
        .std()
        .reset_index(level=0, drop=True)
    )

    df["return_z_20"] = (df["return_1d"] - rolling_mean) / rolling_std

    df["intraday_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["open_close_return"] = (df["Close"] - df["Open"]) / df["Open"]

    df["intraday_range"] = df["intraday_range"].replace([np.inf, -np.inf], np.nan)
    df["open_close_return"] = df["open_close_return"].replace([np.inf, -np.inf], np.nan)

    volume_missing_share = df["Volume"].isna().mean()
    volume_nonzero_share = (df["Volume"].fillna(0) > 0).mean()

    use_volume = (
        volume_missing_share <= MAX_ALLOWED_VOLUME_MISSING_SHARE
        and volume_nonzero_share >= MIN_VOLUME_NONZERO_SHARE
        and df["Volume"].nunique(dropna=True) > 20
    )

    if use_volume:
        volume_mean = (
            df.groupby("Index")["Volume"]
            .rolling(window=20, min_periods=10)
            .mean()
            .reset_index(level=0, drop=True)
        )

        volume_std = (
            df.groupby("Index")["Volume"]
            .rolling(window=20, min_periods=10)
            .std()
            .reset_index(level=0, drop=True)
        )

        df["volume_z_20"] = (df["Volume"] - volume_mean) / volume_std
        df["volume_z_20"] = df["volume_z_20"].replace([np.inf, -np.inf], np.nan)
        df["volume_z_20"] = df["volume_z_20"].fillna(0)

        market_features = BASE_MARKET_FEATURES + [VOLUME_FEATURE]
        print("Volume wird verwendet.")
    else:
        df["volume_z_20"] = 0
        market_features = BASE_MARKET_FEATURES
        print("Volume wird NICHT verwendet.")
        print(f"Volume Missing Share: {volume_missing_share:.2%}")
        print(f"Volume Nonzero Share: {volume_nonzero_share:.2%}")

    before = len(df)

    required_market_cols = market_features.copy()

    for col in required_market_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=required_market_cols).copy()

    # Extremwerte winsorisieren, damit DBSCAN nicht nur Datenfehler findet.
    for col in required_market_cols:
        lower = df[col].quantile(0.001)
        upper = df[col].quantile(0.999)
        df[col] = df[col].clip(lower, upper)

    after = len(df)

    print(f"Zeilen nach Feature-Erstellung: {after}/{before}")
    print("DBSCAN-Marktfeatures:", market_features)

    if after < 1000:
        raise ValueError(
            "Nach Feature-Erstellung bleiben zu wenige Handelstage übrig."
        )

    feature_check = df[market_features].agg(["mean", "std", "min", "max"]).T
    feature_check_path = os.path.join(RESULTS_DIR, "market_feature_quality_check.csv")
    feature_check.to_csv(feature_check_path)

    return df, market_features


# =============================================================================
# KONFLIKTE AUF HANDELSTAGE MAPPEN
# =============================================================================

def map_conflicts_to_next_trading_day(market_df, conflict_df):
    """
    Globaler Schock-Ansatz:
    Jedes Konfliktereignis wird jedem Index dem nächsten Handelstag zugeordnet.

    Wichtig:
    Bei GDE-Daten kann es sehr viele Konfliktereignisse geben. Falls dadurch fast
    jeder Handelstag ein Konflikttag wird, sind die einfachen Konflikttag-Tests
    wenig aussagekräftig. Dann sind High-Intensity-Definitionen wichtiger.
    """
    print("\n" + "=" * 80)
    print("KONFLIKTE AUF NÄCHSTEN HANDELSTAG MAPPEN")
    print("=" * 80)

    market_df = market_df.copy()
    conflict_df = conflict_df.copy()

    market_min = market_df["Date"].min()
    market_max = market_df["Date"].max()

    conflict_df = conflict_df[
        (conflict_df["date_start"] >= market_min - pd.Timedelta(days=MAX_MAPPING_GAP_DAYS))
        & (conflict_df["date_start"] <= market_max)
    ].copy()

    print("Konflikte im Marktzeitraum:", len(conflict_df))

    all_indexes = sorted(market_df["Index"].unique().tolist())
    mapped_rows = []

    conflict_cols = [
        "date_start",
        "date_end",
        "total_deaths",
        "deaths_high",
        "deaths_low",
        "log_deaths",
        "death_uncertainty",
        "conflict_duration_days",
    ]

    optional_cols = [
        "region",
        "country",
        "country_id",
        "type_of_violence",
        "latitude",
        "longitude",
        "where_prec",
        "date_prec",
    ]

    for col in optional_cols:
        if col in conflict_df.columns:
            conflict_cols.append(col)

    conflict_small = conflict_df[conflict_cols].copy()
    conflict_small = conflict_small.sort_values("date_start").reset_index(drop=True)

    for index_name in all_indexes:
        idx_dates = (
            market_df.loc[market_df["Index"] == index_name, ["Date"]]
            .drop_duplicates()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        mapped = pd.merge_asof(
            conflict_small,
            idx_dates,
            left_on="date_start",
            right_on="Date",
            direction="forward",
            tolerance=pd.Timedelta(days=MAX_MAPPING_GAP_DAYS)
        )

        mapped["Index"] = index_name
        mapped_rows.append(mapped)

    mapped_conflicts = pd.concat(mapped_rows, ignore_index=True)
    mapped_conflicts = mapped_conflicts.dropna(subset=["Date"]).copy()

    print("Gemappte Konflikt-Index-Beobachtungen:", len(mapped_conflicts))

    agg = mapped_conflicts.groupby(["Index", "Date"]).agg(
        conflict_count=("date_start", "count"),
        total_deaths_sum=("total_deaths", "sum"),
        total_deaths_max=("total_deaths", "max"),
        total_deaths_mean=("total_deaths", "mean"),
        log_deaths_sum=("log_deaths", "sum"),
        death_uncertainty_sum=("death_uncertainty", "sum"),
        conflict_duration_mean=("conflict_duration_days", "mean"),
    ).reset_index()

    panel = market_df.merge(
        agg,
        on=["Index", "Date"],
        how="left"
    )

    fill_cols = [
        "conflict_count",
        "total_deaths_sum",
        "total_deaths_max",
        "total_deaths_mean",
        "log_deaths_sum",
        "death_uncertainty_sum",
        "conflict_duration_mean",
    ]

    for col in fill_cols:
        panel[col] = panel[col].fillna(0)

    panel["is_conflict_day"] = (panel["conflict_count"] > 0).astype(int)

    conflict_share = panel["is_conflict_day"].mean() * 100

    print("\nPanel erstellt:")
    print("Zeilen:", len(panel))
    print("Konflikttage:", int(panel["is_conflict_day"].sum()))
    print("Nicht-Konflikttage:", int((panel["is_conflict_day"] == 0).sum()))
    print("Anteil Konflikttage:", round(conflict_share, 2), "%")

    if conflict_share > 80:
        print(
            "\n[WARNUNG] Mehr als 80% der Handelstage sind Konflikttage.\n"
            "Das ist bei globalen GDE-Ereignissen plausibel, macht aber den einfachen\n"
            "Vergleich Konflikttag vs. Nicht-Konflikttag schwach. Interpretiere dann\n"
            "vor allem High-Intensity-Konflikttage und Konfliktfenster."
        )

    if conflict_share < 1:
        print(
            "\n[WARNUNG] Sehr wenige Konflikttage. Tests können wenig Power haben."
        )

    return panel


def add_high_intensity_and_windows(panel):
    """
    Erstellt stärkere Konfliktdefinitionen:
    - high intensity conflict day
    - conflict window 3d
    - high intensity conflict window 3d
    """
    print("\n" + "=" * 80)
    print("KONFLIKTINTENSITÄT UND KONFLIKTFENSTER")
    print("=" * 80)

    df = panel.copy()
    df = df.sort_values(["Index", "Date"]).reset_index(drop=True)

    conflict_values = df.loc[df["is_conflict_day"] == 1, "total_deaths_sum"]

    if len(conflict_values) > 0:
        threshold = float(np.percentile(conflict_values, HIGH_INTENSITY_PERCENTILE))
    else:
        threshold = np.inf

    df["is_high_intensity_conflict_day"] = (
        (df["is_conflict_day"] == 1)
        & (df["total_deaths_sum"] >= threshold)
    ).astype(int)

    df["is_conflict_window_3d"] = 0
    df["is_high_intensity_conflict_window_3d"] = 0

    for index_name, group in df.groupby("Index"):
        group = group.sort_values("Date")

        conflict_rolling = (
            group["is_conflict_day"]
            .rolling(window=CONFLICT_WINDOW_DAYS, min_periods=1)
            .max()
        )

        high_conflict_rolling = (
            group["is_high_intensity_conflict_day"]
            .rolling(window=CONFLICT_WINDOW_DAYS, min_periods=1)
            .max()
        )

        df.loc[group.index, "is_conflict_window_3d"] = conflict_rolling.astype(int)
        df.loc[group.index, "is_high_intensity_conflict_window_3d"] = high_conflict_rolling.astype(int)

    print(f"High-Intensity-Schwelle total_deaths_sum: {threshold:.2f}")
    print("Anteil Konflikttage:", round(df["is_conflict_day"].mean() * 100, 2), "%")
    print("Anteil High-Intensity-Konflikttage:", round(df["is_high_intensity_conflict_day"].mean() * 100, 2), "%")
    print("Anteil Konfliktfenster 3d:", round(df["is_conflict_window_3d"].mean() * 100, 2), "%")
    print("Anteil High-Intensity-Konfliktfenster 3d:", round(df["is_high_intensity_conflict_window_3d"].mean() * 100, 2), "%")

    output_path = os.path.join(RESULTS_DIR, "clean_market_conflict_panel.csv")
    df.to_csv(output_path, index=False)
    print("[OK] Panel gespeichert:", output_path)

    return df, threshold


# =============================================================================
# DBSCAN
# =============================================================================

def standardize_dbscan_features(panel, market_features):
    X = panel[market_features].copy()

    for col in market_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        X[col] = X[col].fillna(X[col].median())

    if X.shape[0] < 1000:
        raise ValueError("Zu wenige Beobachtungen für DBSCAN.")

    if (X.std() == 0).any():
        zero_var_cols = X.columns[X.std() == 0].tolist()
        raise ValueError(
            f"Folgende Marktfeatures haben keine Varianz und sind für DBSCAN ungeeignet: {zero_var_cols}"
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def run_dbscan_gridsearch(X_scaled):
    print("\n" + "=" * 80)
    print("DBSCAN GRID-SEARCH AUF MARKTVARIABLEN")
    print("=" * 80)

    rows = []

    for min_samples in MIN_SAMPLES_VALUES:
        neighbors = NearestNeighbors(
            n_neighbors=min_samples,
            metric="euclidean"
        )

        neighbors.fit(X_scaled)
        distances, _ = neighbors.kneighbors(X_scaled)
        kth_distances = np.sort(distances[:, min_samples - 1])

        for percentile in EPS_PERCENTILES:
            eps = float(np.percentile(kth_distances, percentile))

            labels = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric="euclidean"
            ).fit_predict(X_scaled)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int((labels == -1).sum())
            noise_share = 100 * n_noise / len(labels)

            cluster_counts = pd.Series(labels).value_counts()
            cluster_counts_no_noise = cluster_counts.drop(index=-1, errors="ignore")

            if len(cluster_counts_no_noise) > 0:
                largest_cluster_share = 100 * cluster_counts_no_noise.max() / len(labels)
            else:
                largest_cluster_share = 0

            rows.append({
                "min_samples": min_samples,
                "eps_percentile": percentile,
                "eps": eps,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_share_pct": noise_share,
                "largest_cluster_share_pct": largest_cluster_share,
            })

            print(
                f"min_samples={min_samples}, "
                f"percentile={percentile}, "
                f"eps={eps:.6f}, "
                f"clusters={n_clusters}, "
                f"noise={noise_share:.2f}%"
            )

    grid_df = pd.DataFrame(rows)

    output_path = os.path.join(RESULTS_DIR, "dbscan_market_gridsearch.csv")
    grid_df.to_csv(output_path, index=False)
    print("[OK] Grid-Search gespeichert:", output_path)

    return grid_df


def choose_best_dbscan_params(grid_df):
    candidates = grid_df[
        (grid_df["noise_share_pct"] >= MIN_NOISE_SHARE)
        & (grid_df["noise_share_pct"] <= MAX_NOISE_SHARE)
        & (grid_df["n_clusters"] >= 1)
    ].copy()

    if candidates.empty:
        print("[WARNUNG] Keine Parameter im Zielbereich gefunden. Nutze Fallback.")
        candidates = grid_df[grid_df["n_clusters"] >= 1].copy()

    candidates["distance_to_target_noise"] = (
        candidates["noise_share_pct"] - TARGET_NOISE_SHARE
    ).abs()

    best = candidates.sort_values(
        by=["distance_to_target_noise", "largest_cluster_share_pct"],
        ascending=[True, True]
    ).iloc[0]

    print("\n" + "=" * 80)
    print("GEWÄHLTE DBSCAN-PARAMETER")
    print("=" * 80)
    print(best.to_string())

    return {
        "eps": float(best["eps"]),
        "min_samples": int(best["min_samples"]),
        "eps_percentile": int(best["eps_percentile"]),
    }


def apply_dbscan_with_params(panel, X_scaled, eps, min_samples):
    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean"
    ).fit_predict(X_scaled)

    result = panel.copy()
    result["dbscan_cluster"] = labels
    result["is_market_anomaly"] = (labels == -1).astype(int)

    return result


def apply_final_dbscan(panel, X_scaled, params):
    print("\n" + "=" * 80)
    print("FINALES DBSCAN-MODELL")
    print("=" * 80)

    result = apply_dbscan_with_params(
        panel=panel,
        X_scaled=X_scaled,
        eps=params["eps"],
        min_samples=params["min_samples"]
    )

    n_noise = int(result["is_market_anomaly"].sum())
    noise_share = 100 * n_noise / len(result)

    print("eps:", params["eps"])
    print("min_samples:", params["min_samples"])
    print("Marktanomalien:", n_noise)
    print("Anomalieanteil:", round(noise_share, 2), "%")

    output_path = os.path.join(RESULTS_DIR, "dbscan_market_anomaly_results.csv")
    result.to_csv(output_path, index=False)
    print("[OK] DBSCAN-Ergebnisse gespeichert:", output_path)

    anomaly_path = os.path.join(RESULTS_DIR, "dbscan_market_anomaly_cases.csv")
    result[result["is_market_anomaly"] == 1].to_csv(anomaly_path, index=False)
    print("[OK] Marktanomalien gespeichert:", anomaly_path)

    return result


# =============================================================================
# STATISTISCHE TESTS
# =============================================================================

def contingency_test(result, conflict_variable):
    table = pd.crosstab(
        result[conflict_variable],
        result["is_market_anomaly"]
    )

    table = table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

    row0_sum = table.loc[0].sum()
    row1_sum = table.loc[1].sum()

    anomaly_rate_non_conflict = safe_rate(table.loc[0, 1], row0_sum)
    anomaly_rate_conflict = safe_rate(table.loc[1, 1], row1_sum)

    valid_for_test = (
        row0_sum > 0
        and row1_sum > 0
        and table[0].sum() > 0
        and table[1].sum() > 0
    )

    if valid_for_test:
        chi2, p_chi, dof, expected = chi2_contingency(table)
        odds_ratio, p_fisher = fisher_exact(table)
    else:
        chi2 = np.nan
        p_chi = np.nan
        odds_ratio = np.nan
        p_fisher = np.nan

    a = table.loc[1, 1] + 0.5
    b = table.loc[1, 0] + 0.5
    c = table.loc[0, 1] + 0.5
    d = table.loc[0, 0] + 0.5

    corrected_or = (a / b) / (c / d)

    summary = {
        "conflict_variable": conflict_variable,
        "n_non_conflict": int(row0_sum),
        "n_conflict": int(row1_sum),
        "anomaly_non_conflict": int(table.loc[0, 1]),
        "anomaly_conflict": int(table.loc[1, 1]),
        "anomaly_rate_non_conflict": anomaly_rate_non_conflict,
        "anomaly_rate_conflict": anomaly_rate_conflict,
        "difference_conflict_minus_non_conflict": anomaly_rate_conflict - anomaly_rate_non_conflict,
        "chi2": chi2,
        "p_value_chi2": p_chi,
        "fisher_odds_ratio": odds_ratio,
        "p_value_fisher": p_fisher,
        "corrected_odds_ratio": corrected_or,
        "test_valid": valid_for_test,
    }

    return summary, table


def run_all_contingency_tests(result):
    print("\n" + "=" * 80)
    print("KONTINGENZTESTS")
    print("=" * 80)

    conflict_variables = [
        "is_conflict_day",
        "is_high_intensity_conflict_day",
        "is_conflict_window_3d",
        "is_high_intensity_conflict_window_3d",
    ]

    summaries = []
    tables = {}

    for var in conflict_variables:
        print("\n" + "-" * 80)
        print("Testvariable:", var)

        summary, table = contingency_test(result, var)
        summaries.append(summary)
        tables[var] = table

        print("Kontingenztabelle:")
        print(table)
        print("Zusammenfassung:")
        print(pd.DataFrame([summary]).to_string(index=False))

        table_path = os.path.join(RESULTS_DIR, f"contingency_table_{var}.csv")
        table.to_csv(table_path)

    summary_df = pd.DataFrame(summaries)
    summary_df = add_fdr_bh(summary_df, "p_value_chi2", "p_value_chi2_fdr_bh")
    summary_df = add_fdr_bh(summary_df, "p_value_fisher", "p_value_fisher_fdr_bh")

    output_path = os.path.join(RESULTS_DIR, "stat_tests_anomaly_vs_conflict_definitions.csv")
    summary_df.to_csv(output_path, index=False)

    print("[OK] Kontingenztests gespeichert:", output_path)

    return summary_df, tables


def logistic_regression_tests(result):
    """
    Logistische Regression mit geclusterten Standardfehlern nach Index.

    Ziel:
    Prüfen, ob Konfliktvariablen mit höherer Anomaliewahrscheinlichkeit
    verbunden sind, kontrolliert für Index, Wochentag und Jahr.
    """
    print("\n" + "=" * 80)
    print("LOGISTISCHE REGRESSIONEN MIT GECLUSTERTEN STANDARDFEHLERN")
    print("=" * 80)

    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("[WARNUNG] statsmodels nicht installiert.")
        print("Installiere es mit: pip install statsmodels")
        return None

    df = result.copy()

    df["weekday"] = df["Date"].dt.dayofweek.astype(str)
    df["year"] = df["Date"].dt.year.astype(str)

    conflict_variables = [
        "is_conflict_day",
        "is_high_intensity_conflict_day",
        "is_conflict_window_3d",
        "is_high_intensity_conflict_window_3d",
    ]

    rows = []

    for var in conflict_variables:
        print("\nRegression für:", var)

        if df[var].nunique() < 2:
            print("[WARNUNG] Variable hat keine Variation. Überspringe:", var)
            continue

        if df["is_market_anomaly"].nunique() < 2:
            print("[WARNUNG] Zielvariable hat keine Variation. Regression nicht möglich.")
            break

        formula = (
            f"is_market_anomaly ~ {var} "
            "+ C(Index) + C(weekday) + C(year)"
        )

        try:
            model = smf.logit(formula=formula, data=df).fit(
                disp=False,
                maxiter=200,
                cov_type="cluster",
                cov_kwds={"groups": df["Index"]}
            )
        except Exception as exc:
            print("[WARNUNG] Logit-Modell fehlgeschlagen:", exc)
            continue

        coef = model.params.get(var, np.nan)
        p_value = model.pvalues.get(var, np.nan)
        odds_ratio = np.exp(coef)

        rows.append({
            "conflict_variable": var,
            "coef_logit": coef,
            "odds_ratio": odds_ratio,
            "p_value": p_value,
            "significant_5pct": p_value < 0.05,
        })

        text_path = os.path.join(RESULTS_DIR, f"logit_full_summary_{var}.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(str(model.summary()))

    summary_df = pd.DataFrame(rows)

    if not summary_df.empty:
        summary_df = add_fdr_bh(summary_df, "p_value", "p_value_fdr_bh")

    output_path = os.path.join(RESULTS_DIR, "logistic_regression_conflict_anomaly_robust.csv")
    summary_df.to_csv(output_path, index=False)

    print("\nLogit-Zusammenfassung:")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("Keine Regression erfolgreich berechnet.")

    return summary_df


def conflict_intensity_tests(result):
    print("\n" + "=" * 80)
    print("INTENSITÄTSTESTS INNERHALB DER KONFLIKTTAGE")
    print("=" * 80)

    df = result[result["is_conflict_day"] == 1].copy()

    test_vars = [
        "conflict_count",
        "total_deaths_sum",
        "total_deaths_max",
        "total_deaths_mean",
        "log_deaths_sum",
        "death_uncertainty_sum",
        "conflict_duration_mean",
    ]

    rows = []

    for var in test_vars:
        anomaly = df.loc[df["is_market_anomaly"] == 1, var].dropna()
        normal = df.loc[df["is_market_anomaly"] == 0, var].dropna()

        if len(anomaly) < 5 or len(normal) < 5:
            continue

        stat, p = mannwhitneyu(
            anomaly,
            normal,
            alternative="two-sided"
        )

        rows.append({
            "variable": var,
            "median_anomaly_days": anomaly.median(),
            "median_normal_days": normal.median(),
            "mean_anomaly_days": anomaly.mean(),
            "mean_normal_days": normal.mean(),
            "mann_whitney_u": stat,
            "p_value": p,
            "significant_5pct": p < 0.05,
        })

    tests = pd.DataFrame(rows)

    if not tests.empty:
        tests = add_fdr_bh(tests, "p_value", "p_value_fdr_bh")

    output_path = os.path.join(RESULTS_DIR, "conflict_intensity_tests_within_conflict_days.csv")
    tests.to_csv(output_path, index=False)

    if not tests.empty:
        print(tests.to_string(index=False))
    else:
        print("Keine Intensitätstests berechenbar.")

    return tests


# =============================================================================
# ROBUSTHEITSANALYSE ÜBER DBSCAN-PARAMETER
# =============================================================================

def robustness_tests_over_dbscan_params(panel, X_scaled, grid_df):
    print("\n" + "=" * 80)
    print("ROBUSTHEITSTESTS ÜBER DBSCAN-PARAMETER")
    print("=" * 80)

    rows = []

    conflict_variables = [
        "is_conflict_day",
        "is_high_intensity_conflict_day",
        "is_conflict_window_3d",
        "is_high_intensity_conflict_window_3d",
    ]

    for _, row in grid_df.iterrows():
        eps = float(row["eps"])
        min_samples = int(row["min_samples"])

        result = apply_dbscan_with_params(
            panel=panel,
            X_scaled=X_scaled,
            eps=eps,
            min_samples=min_samples
        )

        for var in conflict_variables:
            summary, _ = contingency_test(result, var)

            rows.append({
                "min_samples": min_samples,
                "eps_percentile": int(row["eps_percentile"]),
                "eps": eps,
                "n_clusters": int(row["n_clusters"]),
                "noise_share_pct": float(row["noise_share_pct"]),
                "largest_cluster_share_pct": float(row["largest_cluster_share_pct"]),
                "conflict_variable": var,
                "anomaly_rate_non_conflict": summary["anomaly_rate_non_conflict"],
                "anomaly_rate_conflict": summary["anomaly_rate_conflict"],
                "difference_conflict_minus_non_conflict": summary["difference_conflict_minus_non_conflict"],
                "corrected_odds_ratio": summary["corrected_odds_ratio"],
                "p_value_chi2": summary["p_value_chi2"],
                "p_value_fisher": summary["p_value_fisher"],
                "test_valid": summary["test_valid"],
            })

    robustness_df = pd.DataFrame(rows)

    output_path = os.path.join(RESULTS_DIR, "robustness_anomaly_conflict_by_dbscan_params.csv")
    robustness_df.to_csv(output_path, index=False)

    print("[OK] Robustheitstests gespeichert:", output_path)

    return robustness_df


# =============================================================================
# VISUALISIERUNGEN
# =============================================================================

def create_plots(result, grid_df):
    print("\n" + "=" * 80)
    print("PLOTS ERSTELLEN")
    print("=" * 80)

    # 1. Anomalierate nach Konfliktdefinition
    plot_rows = []

    for var in [
        "is_conflict_day",
        "is_high_intensity_conflict_day",
        "is_conflict_window_3d",
        "is_high_intensity_conflict_window_3d",
    ]:
        rates = result.groupby(var)["is_market_anomaly"].mean()

        if 0 in rates.index and 1 in rates.index:
            plot_rows.append({
                "conflict_definition": var,
                "normal": rates.loc[0],
                "conflict": rates.loc[1],
            })

    rates_df = pd.DataFrame(plot_rows)

    if not rates_df.empty:
        x = np.arange(len(rates_df))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x - width / 2, rates_df["normal"], width, label="Definition nicht erfüllt")
        plt.bar(x + width / 2, rates_df["conflict"], width, label="Definition erfüllt")
        plt.xticks(x, rates_df["conflict_definition"], rotation=30, ha="right")
        plt.ylabel("Anteil DBSCAN-Marktanomalien")
        plt.title("Anomalierate nach Konfliktdefinition")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)

        path = os.path.join(PLOT_DIR, "01_anomaly_rate_by_conflict_definition.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    # 2. Return-Verteilung
    plt.figure(figsize=(9, 6))
    result.boxplot(
        column="return_1d",
        by="is_market_anomaly",
        grid=False
    )
    plt.title("Tagesrenditen: normale Tage vs. DBSCAN-Anomalien")
    plt.suptitle("")
    plt.xlabel("0 = normal, 1 = Marktanomalie")
    plt.ylabel("return_1d")
    plt.grid(axis="y", alpha=0.3)

    path = os.path.join(PLOT_DIR, "02_return_boxplot_anomaly.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Volatilität
    plt.figure(figsize=(9, 6))
    result.boxplot(
        column="volatility_20",
        by="is_market_anomaly",
        grid=False
    )
    plt.title("Volatilität: normale Tage vs. DBSCAN-Anomalien")
    plt.suptitle("")
    plt.xlabel("0 = normal, 1 = Marktanomalie")
    plt.ylabel("volatility_20")
    plt.grid(axis="y", alpha=0.3)

    path = os.path.join(PLOT_DIR, "03_volatility_boxplot_anomaly.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Anomalien pro Index
    anomaly_by_index = (
        result.groupby("Index")["is_market_anomaly"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 6))
    anomaly_by_index.plot(kind="bar")
    plt.ylabel("Anteil Marktanomalien")
    plt.title("DBSCAN-Anomalieanteil pro Index")
    plt.grid(axis="y", alpha=0.3)

    path = os.path.join(PLOT_DIR, "04_anomaly_rate_by_index.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    # 5. DBSCAN Grid Noise-Anteil
    plt.figure(figsize=(10, 6))

    for min_samples, group in grid_df.groupby("min_samples"):
        group = group.sort_values("eps_percentile")
        plt.plot(
            group["eps_percentile"],
            group["noise_share_pct"],
            marker="o",
            label=f"min_samples={min_samples}"
        )

    plt.axhline(TARGET_NOISE_SHARE, linestyle="--", label="Ziel-Anomalieanteil")
    plt.xlabel("eps-Perzentil")
    plt.ylabel("Noise-Anteil in %")
    plt.title("DBSCAN-Grid: Noise-Anteil nach Parameterwahl")
    plt.legend()
    plt.grid(alpha=0.3)

    path = os.path.join(PLOT_DIR, "05_dbscan_grid_noise_share.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print("[OK] Plots gespeichert in:", PLOT_DIR)


# =============================================================================
# REPORT
# =============================================================================

def generate_report(
    params,
    market_features,
    high_intensity_threshold,
    usability_report,
    result,
    contingency_summary,
    logistic_summary,
    intensity_tests,
    robustness_df
):
    report_path = os.path.join(RESULTS_DIR, "DBSCAN_clean_statistical_report.txt")

    n_total = len(result)
    n_anomaly = int(result["is_market_anomaly"].sum())
    anomaly_share = 100 * n_anomaly / n_total

    n_conflict = int(result["is_conflict_day"].sum())
    n_non_conflict = int((result["is_conflict_day"] == 0).sum())

    if logistic_summary is not None and not logistic_summary.empty:
        logit_text = logistic_summary.to_string(index=False)
    else:
        logit_text = "Logistische Regression wurde nicht berechnet."

    if intensity_tests is not None and not intensity_tests.empty:
        intensity_text = intensity_tests.to_string(index=False)
    else:
        intensity_text = "Keine Intensitätstests berechenbar."

    if robustness_df is not None and not robustness_df.empty:
        robustness_short = (
            robustness_df
            .groupby("conflict_variable")
            .agg(
                median_odds_ratio=("corrected_odds_ratio", "median"),
                share_significant_chi2=("p_value_chi2", lambda x: np.mean(x < 0.05)),
                median_difference=("difference_conflict_minus_non_conflict", "median"),
            )
            .reset_index()
        )
        robustness_text = robustness_short.to_string(index=False)
    else:
        robustness_text = "Keine Robustheitsanalyse berechenbar."

    report = f"""
================================================================================
DBSCAN-MARKTANOMALIEANALYSE MIT KONTROLLGRUPPE
================================================================================

FRAGESTELLUNG
================================================================================
Es wird untersucht, ob ungewöhnliche Marktreaktionen häufiger an Handelstagen
auftreten, denen geopolitische Konfliktereignisse zugeordnet wurden.

VERWENDETE DATEIEN
================================================================================
- data/processed/indexData.csv
- data/processed/GDEEvent_v25_1.csv

DATENPRÜFUNG
================================================================================
{usability_report.to_string(index=False)}

METHODISCHE LOGIK
================================================================================
DBSCAN wird ausschließlich auf Marktvariablen aus indexData.csv angewendet.

Konfliktinformationen aus GDEEvent_v25_1.csv werden nicht für das Clustering
verwendet. Sie werden erst nachträglich als Konfliktindikatoren auf Handelstage
gemappt.

Beobachtungen mit DBSCAN-Label -1 werden als Marktanomalien interpretiert.

WICHTIG:
Diese Analyse zeigt statistische Zusammenhänge, aber keine Kausalität.

DATENGRUNDLAGE
================================================================================
Beobachtungen gesamt:              {n_total}
Konflikttage:                      {n_conflict}
Nicht-Konflikttage:                {n_non_conflict}
DBSCAN-Marktanomalien:             {n_anomaly}
Anomalieanteil gesamt:             {anomaly_share:.2f} %

High-Intensity-Schwelle:
total_deaths_sum >= {high_intensity_threshold:.2f}

DBSCAN-MARKTFEATURES
================================================================================
{market_features}

GEWÄHLTE DBSCAN-PARAMETER
================================================================================
eps:                               {params["eps"]:.6f}
min_samples:                       {params["min_samples"]}
eps_percentile:                    {params["eps_percentile"]}

KONTINGENZTESTS
================================================================================
{contingency_summary.to_string(index=False)}

LOGISTISCHE REGRESSIONEN
================================================================================
Kontrolliert für:
- Index
- Wochentag
- Jahr

Standardfehler:
- nach Index geclustert

{logit_text}

INTENSITÄTSTESTS INNERHALB DER KONFLIKTTAGE
================================================================================
{intensity_text}

ROBUSTHEITSANALYSE ÜBER DBSCAN-PARAMETER
================================================================================
{robustness_text}

INTERPRETATION FÜR DIE HAUSARBEIT
================================================================================
Ein statistisch signifikanter positiver Zusammenhang würde bedeuten, dass
DBSCAN-Marktanomalien an Konflikttagen oder in Konfliktfenstern häufiger
auftreten als an normalen Handelstagen.

Wenn die p-Werte nicht signifikant sind, kann nicht gezeigt werden, dass
ungewöhnliche Marktreaktionen systematisch häufiger mit geopolitischen
Konfliktereignissen zusammenfallen.

Falls sehr viele Handelstage als Konflikttage markiert werden, ist die einfache
Variable is_conflict_day nur eingeschränkt aussagekräftig. Dann sollten vor allem
High-Intensity-Konflikttage betrachtet werden.

Auch bei Signifikanz gilt:
DBSCAN liefert keine kausale Erklärung. Es handelt sich um eine explorative
Anomalieanalyse mit statistischem Vergleich gegen eine Kontrollgruppe.

DATEIEN
================================================================================
Direkt in results/:
- data_usability_report.csv
- market_feature_quality_check.csv
- clean_market_conflict_panel.csv
- dbscan_market_gridsearch.csv
- dbscan_market_anomaly_results.csv
- dbscan_market_anomaly_cases.csv
- stat_tests_anomaly_vs_conflict_definitions.csv
- logistic_regression_conflict_anomaly_robust.csv
- conflict_intensity_tests_within_conflict_days.csv
- robustness_anomaly_conflict_by_dbscan_params.csv
- DBSCAN_clean_statistical_report.txt

In results/plots/:
- 01_anomaly_rate_by_conflict_definition.png
- 02_return_boxplot_anomaly.png
- 03_volatility_boxplot_anomaly.png
- 04_anomaly_rate_by_index.png
- 05_dbscan_grid_noise_share.png
================================================================================
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("[OK] Report gespeichert:", report_path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    create_directories()

    stock_df = load_stock_data()
    conflict_df = load_conflict_data()

    usability_report = check_input_usability(stock_df, conflict_df)

    market_df, market_features = create_market_features(stock_df)

    panel = map_conflicts_to_next_trading_day(
        market_df=market_df,
        conflict_df=conflict_df
    )

    panel, high_intensity_threshold = add_high_intensity_and_windows(panel)

    X_scaled, scaler = standardize_dbscan_features(
        panel=panel,
        market_features=market_features
    )

    grid_df = run_dbscan_gridsearch(X_scaled)

    best_params = choose_best_dbscan_params(grid_df)

    result = apply_final_dbscan(
        panel=panel,
        X_scaled=X_scaled,
        params=best_params
    )

    contingency_summary, contingency_tables = run_all_contingency_tests(result)

    logistic_summary = logistic_regression_tests(result)

    intensity_tests = conflict_intensity_tests(result)

    robustness_df = robustness_tests_over_dbscan_params(
        panel=panel,
        X_scaled=X_scaled,
        grid_df=grid_df
    )

    create_plots(
        result=result,
        grid_df=grid_df
    )

    generate_report(
        params=best_params,
        market_features=market_features,
        high_intensity_threshold=high_intensity_threshold,
        usability_report=usability_report,
        result=result,
        contingency_summary=contingency_summary,
        logistic_summary=logistic_summary,
        intensity_tests=intensity_tests,
        robustness_df=robustness_df
    )

    print("\n" + "=" * 80)
    print("ANALYSE ABGESCHLOSSEN")
    print("=" * 80)
    print("Ergebnisse gespeichert in:", RESULTS_DIR)
    print("Plots gespeichert in:", PLOT_DIR)


if __name__ == "__main__":
    main()