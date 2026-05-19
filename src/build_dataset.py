"""
build_dataset.py

Joint die Rohdaten (Aktienindizes & GPR) und erzeugt drei aufbereitete Dateien:

    data/processed/stocks_gpr_daily.csv     -- Tages-Panel (Aktien <- GPR)
    data/processed/stocks_gpr_monthly.csv   -- Monats-Panel (aggregiert)
    data/processed/stocks_gpr_features.csv  -- Monats-Panel + Features (analysefertig)

Folgt der Spezifikation aus docs/JOIN_PROMPT_1.md (bzw. dem Aufgaben-Prompt).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Pfade ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw"
OUT_DIR  = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = RAW_DIR / "indexData.csv"
GPR_PATH   = RAW_DIR / "data_gpr_daily_recent.xls"

OUT_DAILY    = OUT_DIR / "stocks_gpr_daily.csv"
OUT_MONTHLY  = OUT_DIR / "stocks_gpr_monthly.csv"
OUT_FEATURES = OUT_DIR / "stocks_gpr_features.csv"


# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 1: Rohdateien einlesen
# ══════════════════════════════════════════════════════════════════════════════
print("[1/9] Lade Rohdateien...")
gpr    = pd.read_excel(GPR_PATH, engine="xlrd")   # PFLICHT: Legacy .xls
stocks = pd.read_csv(INDEX_PATH)

assert gpr.shape    == (15106, 11), f"GPR unerwartet: {gpr.shape}"
assert stocks.shape == (112457, 8), f"Stocks unerwartet: {stocks.shape}"
print(f"      GPR: {gpr.shape}, Stocks: {stocks.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 2: GPR reduzieren + umbenennen
# ══════════════════════════════════════════════════════════════════════════════
print("[2/9] GPR auf 3 Kern-Spalten reduzieren...")
gpr_clean = gpr[['date', 'GPRD', 'GPRD_ACT', 'GPRD_THREAT']].copy()
gpr_clean = gpr_clean.rename(columns={'date': 'Date'})

assert gpr_clean[['GPRD', 'GPRD_ACT', 'GPRD_THREAT']].isna().sum().sum() == 0, \
    "GPR-Kernspalten enthalten NaN!"
assert gpr_clean['Date'].duplicated().sum() == 0, "GPR enthält Datums-Duplikate!"


# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 3: Stock-Daten bereinigen
# ══════════════════════════════════════════════════════════════════════════════
print("[3/9] Stock-Daten bereinigen...")

# 3a) Date: STR -> datetime
stocks['Date'] = pd.to_datetime(stocks['Date'])

# 3b) NaN-Zeilen entfernen (Feiertage / Handelsaussetzungen)
n_before = len(stocks)
stocks = stocks.dropna(subset=['Close']).copy()
n_dropped = n_before - len(stocks)
print(f"      NaN entfernt: {n_dropped} Zeilen (erwartet ~2204)")
assert 2000 <= n_dropped <= 2500, f"Unerwartete NaN-Anzahl: {n_dropped}"

# 3c) Nur benötigte Spalten
stocks = stocks[['Index', 'Date', 'Close', 'Adj Close']].copy()

# 3d) (Index, Date) eindeutig
assert stocks.duplicated(subset=['Index', 'Date']).sum() == 0, \
    "(Index, Date)-Duplikate gefunden!"


# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 4: Gemeinsamer Zeitraum + Index-Auswahl
# ══════════════════════════════════════════════════════════════════════════════
print("[4/9] Zeitraum & Indizes filtern...")
START = pd.Timestamp("2000-01-01")
END   = pd.Timestamp("2021-05-31")

KEEP_INDICES = [
    '000001.SS', '399001.SZ', 'GDAXI', 'GSPTSE',
    'HSI', 'IXIC', 'KS11', 'N100', 'N225', 'NYA', 'SSMI', 'TWII',
]

stocks = stocks[
    (stocks['Index'].isin(KEEP_INDICES))
    & (stocks['Date'] >= START)
    & (stocks['Date'] <= END)
].copy()

gpr_clean = gpr_clean[
    (gpr_clean['Date'] >= START) & (gpr_clean['Date'] <= END)
].copy()

print(f"      Stocks gefiltert: {stocks.shape}, GPR gefiltert: {gpr_clean.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 5: TAGES-JOIN (LEFT JOIN Aktien <- GPR)
# ══════════════════════════════════════════════════════════════════════════════
print("[5/9] Tages-Join...")
df_daily = stocks.merge(gpr_clean, on='Date', how='left')

n_missing = df_daily[['GPRD', 'GPRD_ACT', 'GPRD_THREAT']].isna().sum().sum()
print(f"      Fehlende GPR-Werte nach Join: {n_missing} (erwartet: 0)")
assert n_missing == 0, "GPR-Lücken! Datumskonvertierung prüfen."

df_daily['YearMonth'] = df_daily['Date'].dt.to_period('M').astype(str)

df_daily.to_csv(OUT_DAILY, index=False)
print(f"      -> {OUT_DAILY.name}: {df_daily.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 6: AKTIEN auf Monatsebene (letzter Schlusskurs)
# ══════════════════════════════════════════════════════════════════════════════
print("[6/9] Aktien-Monatsaggregation...")
stocks_monthly = (
    df_daily
    .sort_values(['Index', 'Date'])
    .groupby(['Index', 'YearMonth'], as_index=False)
    .agg(Close_last=('Close', 'last'))
)

# Monatsrendite je Index (kein Leak über Index-Grenzen!)
stocks_monthly['Stock_monthly_pct'] = (
    stocks_monthly
    .groupby('Index')['Close_last']
    .pct_change() * 100
)


# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 7: GPR auf Monatsebene (Durchschnitt)
# ══════════════════════════════════════════════════════════════════════════════
print("[7/9] GPR-Monatsaggregation (Durchschnitt)...")
gpr_monthly = (
    gpr_clean
    .assign(YearMonth=gpr_clean['Date'].dt.to_period('M').astype(str))
    .groupby('YearMonth', as_index=False)
    .agg(
        GPRD_mean=('GPRD', 'mean'),
        GPRD_ACT_mean=('GPRD_ACT', 'mean'),
        GPRD_THREAT_mean=('GPRD_THREAT', 'mean'),
    )
)

gpr_monthly['GPRD_monthly_pct']        = gpr_monthly['GPRD_mean'].pct_change() * 100
gpr_monthly['GPRD_ACT_monthly_pct']    = gpr_monthly['GPRD_ACT_mean'].pct_change() * 100
gpr_monthly['GPRD_THREAT_monthly_pct'] = gpr_monthly['GPRD_THREAT_mean'].pct_change() * 100


# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 8: MONATS-JOIN
# ══════════════════════════════════════════════════════════════════════════════
print("[8/9] Monats-Join...")
df_monthly = stocks_monthly.merge(gpr_monthly, on='YearMonth', how='left')

# Erste Zeile je Index (NaN-Rendite, kein Vormonat) entfernen
df_monthly = df_monthly.dropna(subset=['Stock_monthly_pct', 'GPRD_monthly_pct'])

df_monthly.to_csv(OUT_MONTHLY, index=False)
print(f"      -> {OUT_MONTHLY.name}: {df_monthly.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 9: Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════
print("[9/9] Feature Engineering...")
df = df_monthly.sort_values(['Index', 'YearMonth']).copy()

# (a) Lags je Index
for lag in [1, 2, 3]:
    df[f'gpr_lag{lag}']   = df.groupby('Index')['GPRD_monthly_pct'].shift(lag)
    df[f'stock_lag{lag}'] = df.groupby('Index')['Stock_monthly_pct'].shift(lag)

# (b) Rollvolatilität 6 Monate je Index
df['stock_vol6'] = (
    df.groupby('Index')['Stock_monthly_pct']
      .transform(lambda x: x.rolling(6).std())
)

# (c) Standardisiertes GPR-Level (z-Score)
df['gprd_zscore'] = (df['GPRD_mean'] - df['GPRD_mean'].mean()) / df['GPRD_mean'].std()

# (d) GPR-Spike-Indikator (|%-Veränderung| > 1 SD)
gpr_std = df['GPRD_monthly_pct'].std()
df['gpr_spike'] = (df['GPRD_monthly_pct'].abs() > gpr_std).astype(int)

# (e) Winsorisierung 1./99. Perzentil
for col in ['Stock_monthly_pct', 'GPRD_monthly_pct',
            'GPRD_ACT_monthly_pct', 'GPRD_THREAT_monthly_pct']:
    lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(lo, hi)

# (f) Zielvariablen
df['stock_down']  = (df['Stock_monthly_pct'] < 0).astype(int)
df['gpr_up_next'] = (
    df.groupby('Index')['GPRD_monthly_pct'].shift(-1) > 0
).astype(int)

# NaN am Reihenanfang (durch Lags) entfernen
df = df.dropna().reset_index(drop=True)

df.to_csv(OUT_FEATURES, index=False)
print(f"      -> {OUT_FEATURES.name}: {df.shape}")

print("\nFertig.")
