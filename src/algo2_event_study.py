"""
Algoritmo 2: Event Study + Lineare Regression

Isoliert geopolitische Schocks (Richtung A) bzw. Aktienschocks (Richtung B)
und prüft die Reaktion im Ereignisfenster.

Design:
  - Event-Fenster:        Tage -5 bis +5 (11 Handelstage)
  - Schätzfenster:        Tage -30 bis -6 (25 Tage, schätzt "normale" Rendite)
  - Abnormal Return (AR): tatsächliche Rendite - Mittel(Schätzfenster)
  - CAR_post:             Summe AR von Tag +1 bis +5 (Reaktion NACH dem Schock)

Richtung A: Event = GPR-Tagesveränderung im obersten 5%-Perzentil
            → Reagieren die Aktien in den 5 Folgetagen negativ?
Richtung B: Event = Aktienrendite im untersten 5%-Perzentil (großer Crash-Tag)
            → Steigt der GPR in den 5 Folgetagen?

Anschließend lineare Regression der CAR_post auf die Schockstärke
(multivariat HAC + univariat für die Grafik).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats as scistats
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

EVENT_WINDOW = 5            # ±5 Tage um das Event
ESTIM_GAP = 1               # Lücke zwischen Schätz- und Eventfenster
ESTIM_LEN = 25              # Länge des Schätzfensters
SHOCK_QUANTILE = 0.95       # oberes (A) bzw. spiegelverkehrt unteres (B) Perzentil
HAC_LAGS = 5

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATENAUFBEREITUNG
# ─────────────────────────────────────────────────────────────────────────────

def prepare():
    """Laden, filtern, tägliche %-Veränderungen, Winsorisierung."""
    df = pd.read_csv(DATA)
    df['Date'] = pd.to_datetime(df['Date'])

    df = df[(df['Index'].isin(KEEP)) &
            (df['Date'] >= START) &
            (df['Date'] <= END)].copy()
    df = df.sort_values(['Index', 'Date'])

    df['Stock_daily_pct'] = df.groupby('Index')['Close'].pct_change() * 100
    df['GPRD_daily_pct'] = df.groupby('Index')['GPRD'].pct_change() * 100
    df['GPRD_ACT_daily_pct'] = df.groupby('Index')['GPRD_ACT'].pct_change() * 100
    df['GPRD_THREAT_daily_pct'] = df.groupby('Index')['GPRD_THREAT'].pct_change() * 100

    for c in ['Stock_daily_pct', 'GPRD_daily_pct',
              'GPRD_ACT_daily_pct', 'GPRD_THREAT_daily_pct']:
        lo, hi = df[c].quantile(0.01), df[c].quantile(0.99)
        df[c] = df[c].clip(lo, hi)

    return df.dropna(subset=['Stock_daily_pct', 'GPRD_daily_pct'])

# ─────────────────────────────────────────────────────────────────────────────
# 2. EVENT-EXTRAKTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_events(df, direction):
    """
    Findet pro Index Event-Tage und baut für jeden Event die Returns
    (Schätzfenster + Eventfenster) zusammen.

    Returnt DataFrame mit einer Zeile pro Event:
      Index, EventDate, shock_mag, shock_act, shock_threat (nur A),
      AR_[-5..+5] als Spalten
    """
    if direction == 'A':
        shock_col = 'GPRD_daily_pct'
        target_col = 'Stock_daily_pct'
        # Top 5% positive GPR-Schocks
        threshold = df[shock_col].quantile(SHOCK_QUANTILE)
        event_filter = lambda s: s >= threshold
    else:  # B
        shock_col = 'Stock_daily_pct'
        target_col = 'GPRD_daily_pct'
        # Untere 5% (große negative Aktien-Schocks)
        threshold = df[shock_col].quantile(1 - SHOCK_QUANTILE)
        event_filter = lambda s: s <= threshold

    pre_buffer = ESTIM_GAP + ESTIM_LEN + EVENT_WINDOW  # Tage VOR Event nötig
    post_buffer = EVENT_WINDOW                          # Tage NACH Event nötig
    rel_days = np.arange(-EVENT_WINDOW, EVENT_WINDOW + 1)
    ar_cols = [f'AR_{d:+d}' for d in rel_days]

    events = []
    for idx, grp in df.groupby('Index'):
        grp = grp.sort_values('Date').reset_index(drop=True)
        mask = event_filter(grp[shock_col])
        event_pos = np.where(mask.values)[0]

        for ei in event_pos:
            if ei < pre_buffer or ei + post_buffer >= len(grp):
                continue

            est_slice = grp[target_col].iloc[
                ei - pre_buffer : ei - EVENT_WINDOW - ESTIM_GAP
            ]
            if est_slice.isna().any() or len(est_slice) < ESTIM_LEN:
                continue
            expected = est_slice.mean()

            win = grp[target_col].iloc[
                ei - EVENT_WINDOW : ei + EVENT_WINDOW + 1
            ].values
            if np.isnan(win).any():
                continue
            ar = win - expected

            row = {
                'Index': idx,
                'EventDate': grp['Date'].iloc[ei],
                'shock_mag': grp[shock_col].iloc[ei],
            }
            if direction == 'A':
                row['shock_act'] = grp['GPRD_ACT_daily_pct'].iloc[ei]
                row['shock_threat'] = grp['GPRD_THREAT_daily_pct'].iloc[ei]
            else:
                row['stock_lag1'] = grp['Stock_daily_pct'].iloc[ei - 1]
                row['stock_lag2'] = grp['Stock_daily_pct'].iloc[ei - 2]
            row.update(dict(zip(ar_cols, ar)))
            events.append(row)

    events_df = pd.DataFrame(events)
    events_df['CAR_pre'] = events_df[[f'AR_{d:+d}' for d in range(-EVENT_WINDOW, 0)]].sum(axis=1)
    events_df['CAR_post'] = events_df[[f'AR_{d:+d}' for d in range(1, EVENT_WINDOW + 1)]].sum(axis=1)
    events_df['CAR_total'] = events_df[ar_cols].sum(axis=1)
    return events_df, ar_cols, rel_days, threshold

# ─────────────────────────────────────────────────────────────────────────────
# 3. EVENT-STUDY-GRAFIK (AVG CAR um t=0)
# ─────────────────────────────────────────────────────────────────────────────

def plot_event_study(events_df, ar_cols, rel_days, direction, outfile):
    """Durchschnittliche AR + kumuliertes CAR mit 95%-Konfidenzband."""
    ar_mat = events_df[ar_cols].values
    n = len(ar_mat)
    mean_ar = ar_mat.mean(axis=0)
    sem_ar = ar_mat.std(axis=0, ddof=1) / np.sqrt(n)
    ci_ar = 1.96 * sem_ar

    mean_car = mean_ar.cumsum()
    # CI für CAR via aufsummierte Varianzen (vereinfacht, unabhängige Tage)
    var_car = (sem_ar ** 2).cumsum()
    ci_car = 1.96 * np.sqrt(var_car)

    if direction == 'A':
        target = 'Aktien'
        title = (f'Event Study A: Aktienreaktion auf GPR-Schock\n'
                 f'n = {n} Events  |  Schock-Schwelle: {SHOCK_QUANTILE*100:.0f}.-Perzentil GPR')
    else:
        target = 'GPR'
        title = (f'Event Study B: GPR-Reaktion auf Aktien-Crash\n'
                 f'n = {n} Events  |  Schock-Schwelle: {(1-SHOCK_QUANTILE)*100:.0f}.-Perzentil Aktien')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Plot 1: durchschnittliches AR pro Tag
    ax1.bar(rel_days, mean_ar, color='#2E75B6', alpha=0.75,
            label=f'Ø Abnormal Return {target}')
    ax1.errorbar(rel_days, mean_ar, yerr=ci_ar, fmt='none',
                 ecolor='#1F4E79', capsize=3, lw=1.0)
    ax1.axhline(0, color='gray', lw=0.6, ls='--')
    ax1.axvline(0, color='#C00000', lw=1.2, ls=':', label='Event-Tag (t=0)')
    ax1.set_xlabel('Tage relativ zum Event', fontsize=11)
    ax1.set_ylabel(f'Ø Abnormal Return {target} (%)', fontsize=11)
    ax1.set_title('Tägliche durchschnittliche AR (±95% KI)', fontsize=11, fontweight='bold')
    ax1.set_xticks(rel_days)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.2)

    # Plot 2: kumuliertes CAR
    ax2.plot(rel_days, mean_car, color='#C00000', lw=2.2, marker='o',
             label=f'Ø CAR {target}')
    ax2.fill_between(rel_days, mean_car - ci_car, mean_car + ci_car,
                     color='#C00000', alpha=0.15, label='95%-Konfidenzband')
    ax2.axhline(0, color='gray', lw=0.6, ls='--')
    ax2.axvline(0, color='#1F4E79', lw=1.2, ls=':', label='Event-Tag (t=0)')
    ax2.set_xlabel('Tage relativ zum Event', fontsize=11)
    ax2.set_ylabel(f'Ø Kumulatives AR {target} (%)', fontsize=11)
    ax2.set_title('Kumuliertes CAR im Ereignisfenster', fontsize=11, fontweight='bold')
    ax2.set_xticks(rel_days)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.2)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'n': n,
        'mean_ar': mean_ar,
        'mean_car_final': mean_car[-1],
        'car_post_avg': events_df['CAR_post'].mean(),
        'car_post_t_pval': scistats.ttest_1samp(events_df['CAR_post'], 0).pvalue,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 4. LINEARE REGRESSION (CAR_post ~ Schockstärke)
# ─────────────────────────────────────────────────────────────────────────────

def fit_multivariate(events_df, direction):
    """OLS mit HAC-SE: CAR_post = f(shock_mag, Kontrollen)."""
    if direction == 'A':
        feats = ['shock_mag', 'shock_act', 'shock_threat']
    else:
        feats = ['shock_mag', 'stock_lag1', 'stock_lag2']

    sub = events_df.dropna(subset=feats + ['CAR_post'])
    y = sub['CAR_post']
    X = sm.add_constant(sub[feats])
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})
    return model, feats

def plot_regression(events_df, direction, outfile):
    """Univariate Regression CAR_post ~ shock_mag mit Streudiagramm + KI."""
    sub = events_df.dropna(subset=['shock_mag', 'CAR_post'])
    x = sub['shock_mag'].values
    y = sub['CAR_post'].values

    X = sm.add_constant(x)
    m = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})
    slope, intercept = m.params[1], m.params[0]
    r2, p_slope = m.rsquared, m.pvalues[1]

    x_line = np.linspace(x.min(), x.max(), 200)
    Xl = sm.add_constant(x_line)
    pred = m.get_prediction(Xl).summary_frame(alpha=0.05)

    if direction == 'A':
        x_label = 'GPR-Schock am Event-Tag (%)'
        y_label = 'CAR Aktien (t+1 … t+5, %)'
        title = 'Regression A: Aktien-CAR nach GPR-Schock'
    else:
        x_label = 'Aktien-Schock am Event-Tag (%)'
        y_label = 'CAR GPR (t+1 … t+5, %)'
        title = 'Regression B: GPR-CAR nach Aktien-Schock'

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, s=24, alpha=0.55, color='#2E75B6',
               edgecolors='none', label='Events')
    ax.plot(x_line, pred['mean'], color='#C00000', lw=2.2,
            label=f'Regressionsgerade (y = {slope:.4f}·x + {intercept:.3f})')
    ax.fill_between(x_line, pred['mean_ci_lower'], pred['mean_ci_upper'],
                    color='#C00000', alpha=0.15, label='95%-Konfidenzband')
    ax.axhline(0, color='gray', lw=0.6, ls='--')
    ax.axvline(0, color='gray', lw=0.6, ls='--')
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(f'{title}\nR² = {r2:.4f}  |  p(Steigung) = {p_slope:.2e}  |  n = {len(x)}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()

    return {'slope': slope, 'intercept': intercept,
            'r2': r2, 'p_slope': p_slope, 'n': len(x)}

# ─────────────────────────────────────────────────────────────────────────────
# 5. FAKTEN-AUSGABE
# ─────────────────────────────────────────────────────────────────────────────

def print_facts(es_stats, model, feats, reg_stats, direction):
    print(f"\n{'='*64}")
    print(f"  RICHTUNG {direction} — EVENT STUDY + LINEARE REGRESSION")
    print(f"{'='*64}")

    print(f"\n[Event-Study-Kennzahlen]")
    print(f"  Anzahl Events            : {es_stats['n']}")
    print(f"  Ø CAR im Gesamtfenster   : {es_stats['mean_car_final']:+.4f} %")
    print(f"  Ø CAR_post (t+1..t+5)    : {es_stats['car_post_avg']:+.4f} %")
    print(f"  p(CAR_post ≠ 0, t-Test)  : {es_stats['car_post_t_pval']:.3e}")

    print(f"\n[Multivariates Modell — CAR_post ~ Schockfaktoren]")
    print(f"  R²          : {model.rsquared:.4f}")
    print(f"  Adj. R²     : {model.rsquared_adj:.4f}")
    print(f"  F-Statistik : {model.fvalue:.2f}  (p = {model.f_pvalue:.3e})")
    print(f"  Beobachtung.: {int(model.nobs)}")
    print(f"\n  Koeffizienten (HAC-robuste p-Werte, maxlags={HAC_LAGS}):")
    for name in model.params.index:
        coef = model.params[name]
        pval = model.pvalues[name]
        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        print(f"    {name:26s}: {coef:+.5f}  (p={pval:.4f}) {sig}")

    print(f"\n[Univariates Modell — Basis der Grafik]")
    print(f"  Steigung    : {reg_stats['slope']:+.5f}")
    print(f"  Achsenabschn: {reg_stats['intercept']:+.4f}")
    print(f"  R² (univar.): {reg_stats['r2']:.4f}")
    print(f"  p(Steigung) : {reg_stats['p_slope']:.3e}")
    print(f"  n           : {reg_stats['n']}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. RICHTUNG B — ANALYSE NACH EINZELNEM INDEX
# ─────────────────────────────────────────────────────────────────────────────

def run_direction_b_by_index(df):
    """
    Führt Event-Study + Regression für Richtung B je Index durch.
    Schwelle = unteres 5%-Perzentil der eigenen Indexrendite (per Index).
    Gibt Liste von Dicts zurück (je Index ein Dict mit Kennzahlen + AR-Matrix).
    """
    rel_days = np.arange(-EVENT_WINDOW, EVENT_WINDOW + 1)
    ar_cols = [f'AR_{d:+d}' for d in rel_days]
    results = []

    for idx in sorted(df['Index'].unique()):
        grp = df[df['Index'] == idx].copy().sort_values('Date').reset_index(drop=True)

        threshold = grp['Stock_daily_pct'].quantile(1 - SHOCK_QUANTILE)
        event_pos = np.where(grp['Stock_daily_pct'].values <= threshold)[0]

        pre_buffer = ESTIM_GAP + ESTIM_LEN + EVENT_WINDOW
        events = []
        for ei in event_pos:
            if ei < pre_buffer or ei + EVENT_WINDOW >= len(grp):
                continue
            est_slice = grp['GPRD_daily_pct'].iloc[
                ei - pre_buffer : ei - EVENT_WINDOW - ESTIM_GAP
            ]
            if est_slice.isna().any() or len(est_slice) < ESTIM_LEN:
                continue
            expected = est_slice.mean()
            win = grp['GPRD_daily_pct'].iloc[ei - EVENT_WINDOW : ei + EVENT_WINDOW + 1].values
            if np.isnan(win).any():
                continue
            ar = win - expected
            events.append({
                'shock_mag': grp['Stock_daily_pct'].iloc[ei],
                **dict(zip(ar_cols, ar))
            })

        if len(events) < 10:
            continue

        ev = pd.DataFrame(events)
        ev['CAR_post'] = ev[[f'AR_{d:+d}' for d in range(1, EVENT_WINDOW + 1)]].sum(axis=1)

        ar_mat = ev[ar_cols].values
        mean_ar = ar_mat.mean(axis=0)

        t_pval = scistats.ttest_1samp(ev['CAR_post'], 0).pvalue

        # Univariate Regression
        sub = ev.dropna(subset=['shock_mag', 'CAR_post'])
        X = sm.add_constant(sub['shock_mag'].values)
        m = sm.OLS(sub['CAR_post'].values, X).fit(
            cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})

        results.append({
            'index': idx,
            'n': len(ev),
            'threshold': threshold,
            'mean_ar': mean_ar,
            'car_post_avg': ev['CAR_post'].mean(),
            'car_post_t_pval': t_pval,
            'slope': m.params[1],
            'p_slope': m.pvalues[1],
            'r2': m.rsquared,
        })

    return results, rel_days


def plot_car_panel(results, rel_days, outfile):
    """4×3 Panel: avg CAR-Kurve pro Index (Richtung B)."""
    n_idx = len(results)
    ncols = 3
    nrows = int(np.ceil(n_idx / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3.8), sharey=False)
    axes = axes.flatten()

    for i, r in enumerate(results):
        ax = axes[i]
        mean_car = r['mean_ar'].cumsum()
        n = r['n']
        sem = r['mean_ar'].std(ddof=1) / np.sqrt(n)
        ci = 1.96 * np.sqrt(np.cumsum(np.full(len(rel_days), sem ** 2)))

        sig = '***' if r['car_post_t_pval'] < 0.01 else \
              '**'  if r['car_post_t_pval'] < 0.05 else \
              '*'   if r['car_post_t_pval'] < 0.1  else ''

        color = '#C00000' if r['car_post_avg'] > 0 else '#2E75B6'
        ax.plot(rel_days, mean_car, color=color, lw=2.0, marker='o', ms=3)
        ax.fill_between(rel_days, mean_car - ci, mean_car + ci,
                        color=color, alpha=0.15)
        ax.axhline(0, color='gray', lw=0.6, ls='--')
        ax.axvline(0, color='black', lw=0.8, ls=':')
        ax.set_title(
            f"{r['index']}  (n={r['n']})  {sig}\n"
            f"CAR_post = {r['car_post_avg']:+.2f}%  |  β={r['slope']:+.3f}",
            fontsize=9, fontweight='bold'
        )
        ax.set_xticks(rel_days[::2])
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        'Event Study B (per Index): GPR-Reaktion auf Aktien-Schock\n'
        f'Schwelle: unteres {(1-SHOCK_QUANTILE)*100:.0f}%-Perzentil je Index  |  '
        f'Fenster ±{EVENT_WINDOW} Tage  |  Sig.: * p<0.1  ** p<0.05  *** p<0.01',
        fontsize=12, fontweight='bold'
    )
    fig.supxlabel('Tage relativ zum Event', fontsize=10)
    fig.supylabel('Ø CAR GPR (%)', fontsize=10)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()


def print_index_summary(results):
    """Gibt Vergleichstabelle aller Indizes aus."""
    print(f"\n{'='*64}")
    print(f"  RICHTUNG B — ZUSAMMENFASSUNG JE INDEX")
    print(f"{'='*64}")
    header = f"  {'Index':<12} {'n':>5} {'Ø CAR_post':>11} {'p(t-Test)':>11} {'Steigung':>10} {'p(β)':>9} {'Sig':>4}"
    print(f"\n{header}")
    print("  " + "─" * 62)
    for r in sorted(results, key=lambda x: x['car_post_avg'], reverse=True):
        sig = '***' if r['car_post_t_pval'] < 0.01 else \
              '**'  if r['car_post_t_pval'] < 0.05 else \
              '*'   if r['car_post_t_pval'] < 0.1  else ''
        print(
            f"  {r['index']:<12} {r['n']:>5} "
            f"{r['car_post_avg']:>+10.3f}% "
            f"{r['car_post_t_pval']:>11.3e} "
            f"{r['slope']:>+10.4f} "
            f"{r['p_slope']:>9.3e} "
            f"{sig:>4}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. GPRD_ACT vs. GPRD_THREAT — SCHOCKTYP-VERGLEICH (Richtung A)
# ─────────────────────────────────────────────────────────────────────────────

def extract_events_component(df, shock_col, target_col):
    """
    Generalisierter Event-Extraktor: top SHOCK_QUANTILE-Schocks in shock_col,
    Reaktion gemessen in target_col. Gibt (events_df, ar_cols, rel_days, threshold) zurück.
    """
    rel_days = np.arange(-EVENT_WINDOW, EVENT_WINDOW + 1)
    ar_cols = [f'AR_{d:+d}' for d in rel_days]
    pre_buffer = ESTIM_GAP + ESTIM_LEN + EVENT_WINDOW

    threshold = df[shock_col].quantile(SHOCK_QUANTILE)
    events = []

    for idx, grp in df.groupby('Index'):
        grp = grp.sort_values('Date').reset_index(drop=True)
        event_pos = np.where(grp[shock_col].values >= threshold)[0]

        for ei in event_pos:
            if ei < pre_buffer or ei + EVENT_WINDOW >= len(grp):
                continue
            est_slice = grp[target_col].iloc[
                ei - pre_buffer : ei - EVENT_WINDOW - ESTIM_GAP
            ]
            if est_slice.isna().any() or len(est_slice) < ESTIM_LEN:
                continue
            expected = est_slice.mean()
            win = grp[target_col].iloc[ei - EVENT_WINDOW : ei + EVENT_WINDOW + 1].values
            if np.isnan(win).any():
                continue
            ar = win - expected
            events.append({
                'Index': idx,
                'EventDate': grp['Date'].iloc[ei],
                'shock_mag': grp[shock_col].iloc[ei],
                **dict(zip(ar_cols, ar))
            })

    ev = pd.DataFrame(events)
    ev['CAR_post'] = ev[[f'AR_{d:+d}' for d in range(1, EVENT_WINDOW + 1)]].sum(axis=1)
    ev['CAR_pre']  = ev[[f'AR_{d:+d}' for d in range(-EVENT_WINDOW, 0)]].sum(axis=1)
    return ev, ar_cols, rel_days, threshold


def _component_stats(ev, ar_cols):
    """Berechnet Kennzahlen + univariate Regression für einen Schocktyp."""
    ar_mat = ev[ar_cols].values
    mean_ar = ar_mat.mean(axis=0)
    sem_ar  = ar_mat.std(axis=0, ddof=1) / np.sqrt(len(ev))
    mean_car = mean_ar.cumsum()
    ci_car = 1.96 * np.sqrt(np.cumsum(sem_ar ** 2))

    t_pval = scistats.ttest_1samp(ev['CAR_post'], 0).pvalue

    sub = ev.dropna(subset=['shock_mag', 'CAR_post'])
    X = sm.add_constant(sub['shock_mag'].values)
    m = sm.OLS(sub['CAR_post'].values, X).fit(
        cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})

    return {
        'n': len(ev),
        'mean_ar': mean_ar,
        'mean_car': mean_car,
        'ci_car': ci_car,
        'car_post_avg': ev['CAR_post'].mean(),
        't_pval': t_pval,
        'slope': m.params[1],
        'p_slope': m.pvalues[1],
        'r2': m.rsquared,
        'intercept': m.params[0],
    }


def plot_component_comparison(stats_act, stats_threat, rel_days,
                               ev_act, ev_threat, outfile):
    """
    2×2-Panel-Vergleich: ACT vs. THREAT
      Zeile 1: durchschn. AR (Balken) + CAR-Kurve
      Zeile 2: Scatter + Regressionsgerade (CAR_post ~ shock_mag)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'ACT': '#C00000', 'THREAT': '#2E75B6'}
    for col, (label, s, ev) in enumerate([
        ('ACT', stats_act, ev_act),
        ('THREAT', stats_threat, ev_threat)
    ]):
        color = colors[label]
        sig = ('***' if s['t_pval'] < 0.01 else '**' if s['t_pval'] < 0.05
               else '*' if s['t_pval'] < 0.1 else 'n.s.')

        # ── Zeile 0: CAR-Kurve ────────────────────────────────────────────
        ax0 = axes[0, col]
        ax0.plot(rel_days, s['mean_car'], color=color, lw=2.2,
                 marker='o', ms=4, label=f'Ø CAR Aktien ({label})')
        ax0.fill_between(rel_days,
                         s['mean_car'] - s['ci_car'],
                         s['mean_car'] + s['ci_car'],
                         color=color, alpha=0.15, label='95%-KI')
        ax0.axhline(0, color='gray', lw=0.6, ls='--')
        ax0.axvline(0, color='black', lw=0.8, ls=':', label='Event-Tag')
        ax0.set_title(
            f'GPRD_{label}-Schock → Aktien-CAR\n'
            f'n={s["n"]}  |  Ø CAR_post={s["car_post_avg"]:+.3f}%  |  {sig}',
            fontsize=10, fontweight='bold'
        )
        ax0.set_xlabel('Tage relativ zum Event', fontsize=9)
        ax0.set_ylabel('Ø Kumulatives AR Aktien (%)', fontsize=9)
        ax0.set_xticks(rel_days)
        ax0.legend(fontsize=8)
        ax0.grid(alpha=0.2)

        # ── Zeile 1: Regression scatter ───────────────────────────────────
        ax1 = axes[1, col]
        sub = ev.dropna(subset=['shock_mag', 'CAR_post'])
        x, y = sub['shock_mag'].values, sub['CAR_post'].values
        x_line = np.linspace(x.min(), x.max(), 200)
        Xl = sm.add_constant(x_line)
        X = sm.add_constant(x)
        m = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})
        pred = m.get_prediction(Xl).summary_frame(alpha=0.05)

        ax1.scatter(x, y, s=10, alpha=0.3, color=color, edgecolors='none',
                    label='Events')
        ax1.plot(x_line, pred['mean'], color=color, lw=2.0,
                 label=f'β={s["slope"]:+.4f}  p={s["p_slope"]:.2e}')
        ax1.fill_between(x_line, pred['mean_ci_lower'], pred['mean_ci_upper'],
                         color=color, alpha=0.12, label='95%-KI')
        ax1.axhline(0, color='gray', lw=0.6, ls='--')
        ax1.axvline(0, color='gray', lw=0.6, ls='--')
        ax1.set_title(
            f'Regression: CAR_post ~ GPRD_{label}-Schock\n'
            f'R²={s["r2"]:.4f}  |  p(β)={s["p_slope"]:.2e}',
            fontsize=10, fontweight='bold'
        )
        ax1.set_xlabel(f'GPRD_{label} % Veränderung am Event-Tag', fontsize=9)
        ax1.set_ylabel('CAR Aktien (t+1 … t+5, %)', fontsize=9)
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.2)

    fig.suptitle(
        'GPRD_ACT vs. GPRD_THREAT: Reagieren Aktien unterschiedlich?\n'
        f'Schock-Schwelle: {SHOCK_QUANTILE*100:.0f}.-Perzentil je Komponente  |  '
        f'Fenster ±{EVENT_WINDOW} Tage',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()


def print_component_comparison(stats_act, stats_threat):
    """Gibt ACT/THREAT-Vergleichstabelle aus."""
    print(f"\n{'='*64}")
    print(f"  SCHOCKTYP-VERGLEICH: GPRD_ACT vs. GPRD_THREAT → AKTIEN")
    print(f"{'='*64}")

    header = f"  {'Typ':<10} {'n':>5} {'Ø CAR_post':>11} {'p(t-Test)':>11} {'Steigung':>10} {'p(β)':>9} {'Sig':>4}"
    print(f"\n{header}")
    print("  " + "─" * 58)

    for label, s in [('GPRD_ACT', stats_act), ('GPRD_THREAT', stats_threat)]:
        sig = ('***' if s['t_pval'] < 0.01 else '**' if s['t_pval'] < 0.05
               else '*' if s['t_pval'] < 0.1 else '')
        print(
            f"  {label:<10} {s['n']:>5} "
            f"{s['car_post_avg']:>+10.3f}% "
            f"{s['t_pval']:>11.3e} "
            f"{s['slope']:>+10.4f} "
            f"{s['p_slope']:>9.3e} "
            f"{sig:>4}"
        )

    diff = stats_act['car_post_avg'] - stats_threat['car_post_avg']
    print(f"\n  Differenz ACT − THREAT: {diff:+.3f}%")
    if abs(diff) > 0.1:
        if diff < 0:
            print("  → ACT-Schocks lösen stärkere negative Aktienreaktion aus.")
        else:
            print("  → THREAT-Schocks lösen stärkere negative Aktienreaktion aus")
            print("    ('Buy the rumor, sell the news'-Muster möglich).")
    else:
        print("  → Kein wesentlicher Unterschied zwischen den Schocktypen.")


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*64)
    print("  EVENT STUDY + LINEARE REGRESSION — GPR-GRUNDANNAHME")
    print("="*64)
    print(f"\nDatenladevorgang...")

    df = prepare()
    print(f"✓ Datensatz geladen: {len(df)} Tagesbeobachtungen")
    print(f"  Indizes: {sorted(df['Index'].unique())}")
    print(f"  Zeitraum: {df['Date'].min().date()} bis {df['Date'].max().date()}")
    print(f"  Event-Fenster: ±{EVENT_WINDOW} Tage  |  Schätzfenster: {ESTIM_LEN} Tage")

    for direction, tag in [('A', 'GPR-Schock → Aktien'),
                           ('B', 'Aktien-Schock → GPR')]:
        print(f"\n{'─'*64}")
        print(f"Verarbeite Richtung {direction}: {tag}")

        events_df, ar_cols, rel_days, threshold = extract_events(df, direction)
        print(f"  ✓ {len(events_df)} Events extrahiert (Schwelle = {threshold:+.3f})")

        es_stats = plot_event_study(
            events_df, ar_cols, rel_days, direction,
            outfile=f"{OUTPUT_DIR}/event_study_richtung_{direction}.png"
        )
        print(f"  ✓ Event-Study-Grafik: {OUTPUT_DIR}/event_study_richtung_{direction}.png")

        model, feats = fit_multivariate(events_df, direction)
        reg_stats = plot_regression(
            events_df, direction,
            outfile=f"{OUTPUT_DIR}/event_study_regression_{direction}.png"
        )
        print(f"  ✓ Regressionsgrafik:  {OUTPUT_DIR}/event_study_regression_{direction}.png")

        print_facts(es_stats, model, feats, reg_stats, direction)

    # ── Richtung B nach einzelnem Index ──
    print(f"\n{'─'*64}")
    print(f"Verarbeite Richtung B — gruppiert nach Index...")
    index_results, rel_days_idx = run_direction_b_by_index(df)
    print(f"  ✓ {len(index_results)} Indizes analysiert")

    plot_car_panel(
        index_results, rel_days_idx,
        outfile=f"{OUTPUT_DIR}/event_study_B_by_index.png"
    )
    print(f"  ✓ Panel-Grafik: {OUTPUT_DIR}/event_study_B_by_index.png")
    print_index_summary(index_results)

    # ── GPRD_ACT vs. GPRD_THREAT ──
    print(f"\n{'─'*64}")
    print(f"Verarbeite Schocktyp-Vergleich: GPRD_ACT vs. GPRD_THREAT → Aktien...")

    ev_act,    ar_cols_c, rel_days_c, thr_act    = extract_events_component(
        df, 'GPRD_ACT_daily_pct', 'Stock_daily_pct')
    ev_threat, _,         _,          thr_threat  = extract_events_component(
        df, 'GPRD_THREAT_daily_pct', 'Stock_daily_pct')

    print(f"  ✓ ACT-Events:    {len(ev_act)}    (Schwelle = {thr_act:+.3f})")
    print(f"  ✓ THREAT-Events: {len(ev_threat)}  (Schwelle = {thr_threat:+.3f})")

    stats_act    = _component_stats(ev_act,    ar_cols_c)
    stats_threat = _component_stats(ev_threat, ar_cols_c)

    plot_component_comparison(
        stats_act, stats_threat, rel_days_c,
        ev_act, ev_threat,
        outfile=f"{OUTPUT_DIR}/event_study_act_vs_threat.png"
    )
    print(f"  ✓ Vergleichsgrafik: {OUTPUT_DIR}/event_study_act_vs_threat.png")
    print_component_comparison(stats_act, stats_threat)

    print(f"\n{'='*64}")
    print("  INTERPRETATION")
    print(f"{'='*64}")
    print("\n• Richtung A: signifikantes negatives Ø CAR_post bei Aktien")
    print("  → Aktien reagieren NACH einem GPR-Schock fallend.")
    print("• Richtung B (gepooled): signifikantes positives Ø CAR_post beim GPR")
    print("  → GPR steigt NACH einem Aktien-Crash.")
    print("• Richtung B (je Index): Heterogenität über Märkte sichtbar.")
    print("• ACT vs. THREAT: zeigt, ob Bedrohungslagen oder reale Ereignisse")
    print("  stärker auf Aktienmärkte wirken.")

    print(f"\n{'='*64}")
    print(f"✓ Analyse abgeschlossen!")
    print(f"{'='*64}\n")

if __name__ == "__main__":
    main()
