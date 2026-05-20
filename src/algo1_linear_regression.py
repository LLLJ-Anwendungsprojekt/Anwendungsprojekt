"""
Algoritmo 1: Lineare Regression zur Prüfung der GPR-Grundannahmen

Testet zwei zentrale Richtungen:
  A: Fallen Aktien, wenn der Geopolitical Risk (GPR) Index steigt?
  B: Steigt der GPR, wenn die Aktienmärkte zuvor gefallen sind?

Methode:
  - Multivariate OLS (mit HAC-robusten SE) für die belastbaren Facts
  - Univariate OLS für die Visualisierung (Streudiagramm + Gerade)
  - Winsorisierung auf 1./99. Perzentil
  - 95%-Konfidenzband
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 0. KONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA = "data/processed/stocks_gpr_monthly.csv"
KEEP = ['000001.SS', '399001.SZ', 'GDAXI', 'GSPTSE', 'HSI', 'IXIC',
        'KS11', 'N100', 'N225', 'NYA', 'SSMI', 'TWII']
START, END = '2000-01', '2021-05'
OUTPUT_DIR = "results"

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATENAUFBEREITUNG
# ─────────────────────────────────────────────────────────────────────────────

def prepare():
    """Laden, filtern, Lags bilden, Winsorisierung."""
    df = pd.read_csv(DATA)
    df['YearMonth'] = df['YearMonth'].astype(str)
    
    # Filter nach Indizes und Zeitraum
    df = df[(df['Index'].isin(KEEP)) &
            (df['YearMonth'] >= START) & 
            (df['YearMonth'] <= END)].copy()
    
    df = df.sort_values(['Index', 'YearMonth'])

    # Lags je Index (NEVER über Index-Grenzen)
    for l in [1, 2]:
        df[f'gpr_lag{l}'] = df.groupby('Index')['GPRD_monthly_pct'].shift(l)
        df[f'stock_lag{l}'] = df.groupby('Index')['Stock_monthly_pct'].shift(l)
    
    # Ziel Richtung B: GPR im Folgemonat
    df['gpr_next'] = df.groupby('Index')['GPRD_monthly_pct'].shift(-1)

    # Winsorisierung auf 1./99. Perzentil
    for c in ['Stock_monthly_pct', 'GPRD_monthly_pct',
              'GPRD_ACT_monthly_pct', 'GPRD_THREAT_monthly_pct']:
        lo, hi = df[c].quantile(0.01), df[c].quantile(0.99)
        df[c] = df[c].clip(lo, hi)

    return df.dropna(subset=['gpr_lag1', 'gpr_lag2', 'stock_lag1', 'stock_lag2'])

# ─────────────────────────────────────────────────────────────────────────────
# 2. MULTIVARIATES MODELL (liefert die Facts)
# ─────────────────────────────────────────────────────────────────────────────

def fit_multivariate(df, direction):
    """
    OLS-Regression mit HAC-robusten Standardfehlern.
    
    Richtung A: GPR → Aktienrendite
    Richtung B: Aktienrendite → zukünftiger GPR
    """
    if direction == 'A':
        feats = ['GPRD_monthly_pct', 'gpr_lag1', 'gpr_lag2',
                 'GPRD_ACT_monthly_pct', 'GPRD_THREAT_monthly_pct']
        y = df['Stock_monthly_pct']
        X = df[feats]
    else:  # B
        feats = ['Stock_monthly_pct', 'stock_lag1', 'stock_lag2']
        sub = df.dropna(subset=['gpr_next'])
        y = sub['gpr_next']
        X = sub[feats]

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    return model, feats

# ─────────────────────────────────────────────────────────────────────────────
# 3. UNIVARIATES MODELL + GRAFIK
# ─────────────────────────────────────────────────────────────────────────────

def plot_regression(df, direction, outfile):
    """
    Erzeugt Streudiagramm + Regressionsgerade + 95%-Konfidenzband.
    
    Returnt univariate OLS-Stats für die Grafik-Basis.
    """
    if direction == 'A':
        x_col, y_col = 'GPRD_monthly_pct', 'Stock_monthly_pct'
        sub = df.copy()
        x_label = 'GPR % Veränderung (Monat t)'
        y_label = 'Aktienrendite % (Monat t)'
        title = 'Prüfung A: Fallen Aktien bei steigendem GPR?'
    else:  # B
        x_col, y_col = 'Stock_monthly_pct', 'gpr_next'
        sub = df.dropna(subset=['gpr_next']).copy()
        x_label = 'Aktienrendite % (Monat t)'
        y_label = 'GPR % Veränderung (Monat t+1)'
        title = 'Prüfung B: Steigt der GPR nach fallenden Aktien?'

    x = sub[x_col].values
    y = sub[y_col].values

    # Univariates OLS mit HAC für die Grafik
    X = sm.add_constant(x)
    m = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    slope, intercept = m.params[1], m.params[0]
    r2, p_slope = m.rsquared, m.pvalues[1]

    # Vorhersage + Konfidenzband
    x_line = np.linspace(x.min(), x.max(), 200)
    Xl = sm.add_constant(x_line)
    pred = m.get_prediction(Xl).summary_frame(alpha=0.05)

    # ── PLOT ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Streudiagramm
    ax.scatter(x, y, s=14, alpha=0.35, color='#2E75B6',
               edgecolors='none', label='Monatsbeobachtungen')
    
    # Regressionsgerade
    ax.plot(x_line, pred['mean'], color='#C00000', lw=2.2,
            label=f'Regressionsgerade (y = {slope:.4f}·x + {intercept:.3f})')
    
    # 95%-Konfidenzband
    ax.fill_between(x_line, pred['mean_ci_lower'], pred['mean_ci_upper'],
                    color='#C00000', alpha=0.15, label='95%-Konfidenzband')
    
    # Null-Linien
    ax.axhline(0, color='gray', lw=0.6, ls='--')
    ax.axvline(0, color='gray', lw=0.6, ls='--')
    
    # Labels & Styling
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(f'{title}\nR² = {r2:.4f}  |  p(Steigung) = {p_slope:.2e}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.2)
    
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'slope': slope,
        'intercept': intercept,
        'r2': r2,
        'p_slope': p_slope,
        'n': len(x)
    }

# ─────────────────────────────────────────────────────────────────────────────
# 4. FAKTEN-AUSGABE (strukturiert)
# ─────────────────────────────────────────────────────────────────────────────

def print_facts(model, feats, plot_stats, direction):
    """Gibt multivariate und univariate Facts strukturiert aus."""
    print(f"\n{'='*64}")
    print(f"  RICHTUNG {direction} — LINEARE REGRESSION")
    print(f"{'='*64}")

    # ── Multivariate Facts ──
    print(f"\n[Multivariates Modell — belastbare Kennzahlen]")
    print(f"  R²          : {model.rsquared:.4f}")
    print(f"  Adj. R²     : {model.rsquared_adj:.4f}")
    print(f"  F-Statistik : {model.fvalue:.2f}  (p = {model.f_pvalue:.3e})")
    print(f"  Beobachtung.: {int(model.nobs)}")
    
    print(f"\n  Koeffizienten (HAC-robuste p-Werte):")
    for name in model.params.index:
        coef = model.params[name]
        pval = model.pvalues[name]
        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        print(f"    {name:26s}: {coef:+.5f}  (p={pval:.4f}) {sig}")

    # ── Univariate Plot-Facts ──
    print(f"\n[Univariates Modell — Basis der Grafik]")
    print(f"  Steigung    : {plot_stats['slope']:+.5f}")
    print(f"  Achsenabschn: {plot_stats['intercept']:+.4f}")
    print(f"  R² (univar.): {plot_stats['r2']:.4f}")
    print(f"  p(Steigung) : {plot_stats['p_slope']:.3e}")
    print(f"  n           : {plot_stats['n']}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Führt beide Richtungen (A & B) aus."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*64)
    print("  LINEARE REGRESSION ZUR PRÜFUNG DER GPR-GRUNDANNAHME")
    print("="*64)
    print(f"\nDatenladevorgang...")
    
    df = prepare()
    print(f"✓ Datensatz geladen: {len(df)} Beobachtungen")
    print(f"  Indizes: {sorted(df['Index'].unique())}")
    print(f"  Zeitraum: {df['YearMonth'].min()} bis {df['YearMonth'].max()}")

    # ── Richtung A: GPR → Aktien ──
    print(f"\n{'─'*64}")
    print(f"Verarbeite Richtung A...")
    model_a, feats_a = fit_multivariate(df, 'A')
    stats_a = plot_regression(
        df, 'A',
        outfile=f"{OUTPUT_DIR}/regression_richtung_A.png"
    )
    print_facts(model_a, feats_a, stats_a, 'A')
    print(f"  ✓ Grafik gespeichert: {OUTPUT_DIR}/regression_richtung_A.png")

    # ── Richtung B: Aktien → GPR ──
    print(f"\n{'─'*64}")
    print(f"Verarbeite Richtung B...")
    model_b, feats_b = fit_multivariate(df, 'B')
    stats_b = plot_regression(
        df, 'B',
        outfile=f"{OUTPUT_DIR}/regression_richtung_B.png"
    )
    print_facts(model_b, feats_b, stats_b, 'B')
    print(f"  ✓ Grafik gespeichert: {OUTPUT_DIR}/regression_richtung_B.png")

    # ── Zusammenfassung ──
    print(f"\n{'='*64}")
    print("  INTERPRETATION DER GRUNDANNAHME")
    print(f"{'='*64}")
    
    print(f"\n[Richtung A: Fallen Aktien bei steigendem GPR?]")
    print(f"  → Steigung (univariat): {stats_a['slope']:+.5f}")
    print(f"  → Signifikanz: {stats_a['p_slope']:.2e}")
    if stats_a['p_slope'] < 0.05:
        if stats_a['slope'] < 0:
            print(f"  ✓ BESTÄTIGT: Negative Steigung (statistisch signifikant)")
            print(f"    Interpretation: Ein steigender GPR geht mit fallenden")
            print(f"    Aktienrenditen einher.")
        else:
            print(f"  ✗ WIDERLEGT: Positive Steigung (aber signifikant)")
    else:
        print(f"  ⊗ NICHT SIGNIFIKANT: Zusammenhang nicht belastbar")
    
    print(f"\n[Richtung B: Steigt der GPR nach fallenden Aktien?]")
    print(f"  → Steigung (univariat): {stats_b['slope']:+.5f}")
    print(f"  → Signifikanz: {stats_b['p_slope']:.2e}")
    if stats_b['p_slope'] < 0.05:
        if stats_b['slope'] < 0:
            print(f"  ✓ TEILWEISE BESTÄTIGT: Negative Steigung")
            print(f"    Interpretation: Fallende Aktienkurse (t) gehen mit")
            print(f"    steigendem GPR (t+1) einher.")
        else:
            print(f"  ✗ WIDERLEGT: Positive Steigung (signifikant)")
    else:
        print(f"  ⊗ SCHWACH: Zusammenhang nur schwach oder nicht signifikant")
    
    print(f"\n{'='*64}")
    print(f"✓ Analyse abgeschlossen!")
    print(f"{'='*64}\n")

if __name__ == "__main__":
    main()
