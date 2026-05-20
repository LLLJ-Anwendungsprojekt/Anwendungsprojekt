"""
Algoritmo 3: Volatilitätsregression (Tagesebene)

Grundfrage: Geht ein hoher GPR-Index mit hoher Marktvolatilität einher?

Methode:
  - Realisierte Volatilität: rollierende 20-Tage-Standardabweichung der
    täglichen Aktienrenditen (annualisiert: × √252)
  - Richtung A: GPR-Level → Volatilität (gleichzeitig, Tag t)
  - Richtung B: Volatilität (Tag t) → GPR-Level (Tag t+1)
  - Multivariate OLS mit HAC-robusten SE (maxlags=20, ≈ ein Handelsmonat)
  - Univariate OLS für Visualisierung + 95%-Konfidenzband
  - Winsorisierung auf 1./99. Perzentil
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 0. KONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA = "data/processed/stocks_gpr_daily.csv"
KEEP = ['000001.SS', '399001.SZ', 'GDAXI', 'GSPTSE', 'HSI', 'IXIC',
        'KS11', 'N100', 'N225', 'NYA', 'SSMI', 'TWII']
START, END = '2000-01-01', '2021-05-31'
OUTPUT_DIR = "results"

VOL_WINDOW = 20     # Tage für rollierende Volatilität
HAC_LAGS   = 20     # ≈ ein Handelsmonat (Vol ist stark autokorreliert)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATENAUFBEREITUNG
# ─────────────────────────────────────────────────────────────────────────────

def prepare():
    """
    Lädt Tagesdaten, berechnet rollierende Volatilität und GPR-Lags.
    Zielvariablen:
      Vol_20d       = rollierende 20-Tage-Std × √252 (annualisiert, %)
      GPRD          = GPR-Indexstand (Level, nicht %-Veränderung)
      GPRD_ACT/THREAT = Sub-Komponenten (Level)
    """
    df = pd.read_csv(DATA)
    df['Date'] = pd.to_datetime(df['Date'])

    df = df[(df['Index'].isin(KEEP)) &
            (df['Date'] >= START) &
            (df['Date'] <= END)].copy()
    df = df.sort_values(['Index', 'Date'])

    # Tägliche Rendite je Index
    df['Stock_daily_pct'] = df.groupby('Index')['Close'].pct_change() * 100

    # Rollierende 20-Tage-Volatilität (annualisiert)
    df['Vol_20d'] = (
        df.groupby('Index')['Stock_daily_pct']
          .transform(lambda x: x.rolling(VOL_WINDOW).std() * np.sqrt(252))
    )

    # GPR-Lags (Level) je Index
    for l in [1, 2]:
        df[f'gpr_lag{l}']     = df.groupby('Index')['GPRD'].shift(l)
        df[f'gpr_act_lag{l}'] = df.groupby('Index')['GPRD_ACT'].shift(l)
        df[f'vol_lag{l}']     = df.groupby('Index')['Vol_20d'].shift(l)

    # Ziel Richtung B: GPR-Level am Folgetag
    df['gpr_next'] = df.groupby('Index')['GPRD'].shift(-1)

    # Winsorisierung
    for c in ['Vol_20d', 'GPRD', 'GPRD_ACT', 'GPRD_THREAT']:
        lo, hi = df[c].quantile(0.01), df[c].quantile(0.99)
        df[c] = df[c].clip(lo, hi)

    return df.dropna(subset=['Vol_20d', 'GPRD', 'gpr_lag1', 'vol_lag1'])

# ─────────────────────────────────────────────────────────────────────────────
# 2. MULTIVARIATES MODELL
# ─────────────────────────────────────────────────────────────────────────────

def fit_multivariate(df, direction):
    """
    Richtung A: Vol_20d = f(GPRD, GPRD_ACT, GPRD_THREAT, gpr_lag1, gpr_lag2)
    Richtung B: GPRD(t+1) = f(Vol_20d, vol_lag1, vol_lag2)
    """
    if direction == 'A':
        feats = ['GPRD', 'GPRD_ACT', 'GPRD_THREAT', 'gpr_lag1', 'gpr_lag2']
        y = df['Vol_20d']
        X = df[feats]
    else:
        feats = ['Vol_20d', 'vol_lag1', 'vol_lag2']
        sub = df.dropna(subset=['gpr_next'] + feats)
        y = sub['gpr_next']
        X = sub[feats]

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})
    return model, feats

# ─────────────────────────────────────────────────────────────────────────────
# 3. UNIVARIATES MODELL + GRAFIK
# ─────────────────────────────────────────────────────────────────────────────

def plot_regression(df, direction, outfile):
    """Streudiagramm + Regressionsgerade + 95%-Konfidenzband."""
    if direction == 'A':
        x_col, y_col = 'GPRD', 'Vol_20d'
        sub = df.copy()
        x_label = 'GPR-Index (Level, Tag t)'
        y_label = f'Realisierte Volatilität {VOL_WINDOW}T ann. (%, Tag t)'
        title = f'Prüfung A: Hoher GPR → Hohe Marktvolatilität?'
    else:
        x_col, y_col = 'Vol_20d', 'gpr_next'
        sub = df.dropna(subset=['gpr_next']).copy()
        x_label = f'Realisierte Volatilität {VOL_WINDOW}T ann. (%, Tag t)'
        y_label = 'GPR-Index (Level, Tag t+1)'
        title = 'Prüfung B: Hohe Volatilität heute → Höherer GPR morgen?'

    x = sub[x_col].values
    y = sub[y_col].values

    X = sm.add_constant(x)
    m = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})
    slope, intercept = m.params[1], m.params[0]
    r2, p_slope = m.rsquared, m.pvalues[1]

    x_line = np.linspace(x.min(), x.max(), 200)
    Xl = sm.add_constant(x_line)
    pred = m.get_prediction(Xl).summary_frame(alpha=0.05)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, s=6, alpha=0.15, color='#2E75B6',
               edgecolors='none', label='Tagesbeobachtungen')
    ax.plot(x_line, pred['mean'], color='#C00000', lw=2.2,
            label=f'Regressionsgerade (y = {slope:.4f}·x + {intercept:.3f})')
    ax.fill_between(x_line, pred['mean_ci_lower'], pred['mean_ci_upper'],
                    color='#C00000', alpha=0.15, label='95%-Konfidenzband')
    ax.axhline(0, color='gray', lw=0.6, ls='--')
    ax.axvline(0, color='gray', lw=0.6, ls='--')
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(f'{title}\nR² = {r2:.4f}  |  p(Steigung) = {p_slope:.2e}  |  n = {len(x):,}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()

    return {'slope': slope, 'intercept': intercept,
            'r2': r2, 'p_slope': p_slope, 'n': len(x)}

# ─────────────────────────────────────────────────────────────────────────────
# 4. ZEITREIHEN-GRAFIK (Vol + GPR im Verlauf)
# ─────────────────────────────────────────────────────────────────────────────

def plot_timeseries(df, outfile):
    """
    Zeigt durchschnittliche Vol_20d und GPRD über alle Indizes im Zeitverlauf.
    Macht die Ko-Bewegung in Krisenzeiten sichtbar.
    """
    ts = (df.groupby('Date')[['Vol_20d', 'GPRD']]
            .mean()
            .reset_index()
            .sort_values('Date'))

    fig, ax1 = plt.subplots(figsize=(14, 5))

    ax1.fill_between(ts['Date'], ts['Vol_20d'], alpha=0.35,
                     color='#2E75B6', label=f'Ø Vol {VOL_WINDOW}T ann. (%)')
    ax1.set_ylabel(f'Ø Realisierte Volatilität {VOL_WINDOW}T ann. (%)',
                   color='#2E75B6', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='#2E75B6')

    ax2 = ax1.twinx()
    ax2.plot(ts['Date'], ts['GPRD'], color='#C00000', lw=1.4,
             alpha=0.85, label='Ø GPR-Index (Level)')
    ax2.set_ylabel('Ø GPR-Index (Level)', color='#C00000', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='#C00000')

    # Krisen-Marker
    krisen = {
        '9/11 (2001)': '2001-09-11',
        'Irak (2003)': '2003-03-20',
        'Finanzkrise\n(2008)': '2008-09-15',
        'Krim (2014)': '2014-03-01',
        'COVID (2020)': '2020-03-01',
    }
    for label, date in krisen.items():
        ax1.axvline(pd.Timestamp(date), color='gray', lw=0.8,
                    ls=':', alpha=0.7)
        ax1.text(pd.Timestamp(date), ax1.get_ylim()[1] * 0.92,
                 label, fontsize=7, color='gray',
                 ha='center', va='top', rotation=90)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper left', fontsize=9)

    ax1.set_title(
        f'Marktvolatilität vs. GPR-Index im Zeitverlauf (Ø über {len(KEEP)} Indizes)',
        fontsize=12, fontweight='bold'
    )
    ax1.set_xlabel('Datum', fontsize=10)
    ax1.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 5. FAKTEN-AUSGABE
# ─────────────────────────────────────────────────────────────────────────────

def print_facts(model, feats, plot_stats, direction):
    print(f"\n{'='*64}")
    print(f"  RICHTUNG {direction} — VOLATILITÄTSREGRESSION (TAGESEBENE)")
    print(f"{'='*64}")

    print(f"\n[Multivariates Modell — belastbare Kennzahlen]")
    print(f"  R²          : {model.rsquared:.4f}")
    print(f"  Adj. R²     : {model.rsquared_adj:.4f}")
    print(f"  F-Statistik : {model.fvalue:.2f}  (p = {model.f_pvalue:.3e})")
    print(f"  Beobachtung.: {int(model.nobs):,}")

    print(f"\n  Koeffizienten (HAC-robuste p-Werte, maxlags={HAC_LAGS}):")
    for name in model.params.index:
        coef = model.params[name]
        pval = model.pvalues[name]
        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        print(f"    {name:26s}: {coef:+.5f}  (p={pval:.4f}) {sig}")

    print(f"\n[Univariates Modell — Basis der Grafik]")
    print(f"  Steigung    : {plot_stats['slope']:+.6f}")
    print(f"  Achsenabschn: {plot_stats['intercept']:+.4f}")
    print(f"  R² (univar.): {plot_stats['r2']:.4f}")
    print(f"  p(Steigung) : {plot_stats['p_slope']:.3e}")
    print(f"  n           : {plot_stats['n']:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

KEEP = ['000001.SS', '399001.SZ', 'GDAXI', 'GSPTSE', 'HSI', 'IXIC',
        'KS11', 'N100', 'N225', 'NYA', 'SSMI', 'TWII']

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*64)
    print("  VOLATILITÄTSREGRESSION — GPR-LEVEL VS. MARKTVOLATILITÄT")
    print("="*64)
    print(f"\nDatenladevorgang...")

    df = prepare()
    print(f"✓ Datensatz geladen: {len(df):,} Beobachtungen")
    print(f"  Indizes: {sorted(df['Index'].unique())}")
    print(f"  Zeitraum: {df['Date'].min().date()} bis {df['Date'].max().date()}")
    print(f"  Rollvolatilität: {VOL_WINDOW}-Tage-Fenster, annualisiert")
    print(f"  Ø Vol_20d: {df['Vol_20d'].mean():.2f}%  |  Ø GPRD: {df['GPRD'].mean():.1f}")

    # ── Zeitreihen-Grafik ──
    plot_timeseries(df, outfile=f"{OUTPUT_DIR}/vol_gpr_timeseries.png")
    print(f"\n✓ Zeitreihen-Grafik: {OUTPUT_DIR}/vol_gpr_timeseries.png")

    # ── Richtung A: GPR-Level → Volatilität ──
    print(f"\n{'─'*64}")
    print(f"Verarbeite Richtung A: GPR-Level → Marktvolatilität...")
    model_a, feats_a = fit_multivariate(df, 'A')
    stats_a = plot_regression(df, 'A',
        outfile=f"{OUTPUT_DIR}/vol_regression_richtung_A.png")
    print_facts(model_a, feats_a, stats_a, 'A')
    print(f"  ✓ Grafik: {OUTPUT_DIR}/vol_regression_richtung_A.png")

    # ── Richtung B: Volatilität → GPR-Level (Folgetag) ──
    print(f"\n{'─'*64}")
    print(f"Verarbeite Richtung B: Volatilität → GPR-Level (Folgetag)...")
    model_b, feats_b = fit_multivariate(df, 'B')
    stats_b = plot_regression(df, 'B',
        outfile=f"{OUTPUT_DIR}/vol_regression_richtung_B.png")
    print_facts(model_b, feats_b, stats_b, 'B')
    print(f"  ✓ Grafik: {OUTPUT_DIR}/vol_regression_richtung_B.png")

    # ── Zusammenfassung ──
    print(f"\n{'='*64}")
    print("  INTERPRETATION")
    print(f"{'='*64}")

    print(f"\n[Richtung A: Hoher GPR → Hohe Marktvolatilität?]")
    print(f"  → Steigung (univariat): {stats_a['slope']:+.6f}")
    print(f"  → R²: {stats_a['r2']:.4f}  |  p: {stats_a['p_slope']:.2e}")
    if stats_a['p_slope'] < 0.05 and stats_a['slope'] > 0:
        print(f"  ✓ BESTÄTIGT: Höherer GPR geht mit höherer Volatilität einher.")
    elif stats_a['p_slope'] < 0.05 and stats_a['slope'] < 0:
        print(f"  ✗ UMGEKEHRT: Höherer GPR geht mit niedrigerer Volatilität einher.")
    else:
        print(f"  ⊗ NICHT SIGNIFIKANT")

    print(f"\n[Richtung B: Hohe Volatilität heute → Höherer GPR morgen?]")
    print(f"  → Steigung (univariat): {stats_b['slope']:+.6f}")
    print(f"  → R²: {stats_b['r2']:.4f}  |  p: {stats_b['p_slope']:.2e}")
    if stats_b['p_slope'] < 0.05 and stats_b['slope'] > 0:
        print(f"  ✓ BESTÄTIGT: Hohe Marktvolatilität geht mit steigendem GPR einher.")
    elif stats_b['p_slope'] < 0.05 and stats_b['slope'] < 0:
        print(f"  ✗ UMGEKEHRT: Hohe Volatilität geht mit sinkendem GPR einher.")
    else:
        print(f"  ⊗ NICHT SIGNIFIKANT")

    print(f"\n{'='*64}")
    print(f"✓ Analyse abgeschlossen!")
    print(f"{'='*64}\n")

if __name__ == "__main__":
    main()
