"""
KNN-Klassifikation mit Feature Engineering, MI-Selektion und PCA.

Richtung A: GPR-Features    -> Aktien fallen?    (stock_down)
Richtung B: Aktien-Features -> GPR steigt t+1?   (gpr_up_next)

Pipeline: StandardScaler -> SelectKBest(MI) -> PCA(0.95) -> KNN
          (jeder Schritt wird pro CV-Fold nur auf Trainingsdaten gefittet)

Input:  data/processed/stocks_gpr_features.csv
Output: results/knn/knn_roc_curves.png, knn_confusion_matrices.png,
        knn_feature_importance.png, knn_results.txt
"""

import sys
import os
import warnings
from functools import partial

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    confusion_matrix, roc_curve,
)

warnings.filterwarnings('ignore')
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── 0. Konfiguration ────────────────────────────────────────────────────────

DATA       = "data/processed/stocks_gpr_features.csv"
OUTPUT_DIR = "results/knn"
START, END = '2000-01', '2021-05'
KEEP = ['000001.SS', '399001.SZ', 'GDAXI', 'GSPTSE', 'HSI', 'IXIC',
        'KS11', 'N100', 'N225', 'NYA', 'SSMI', 'TWII']

HOLDOUT_FRAC  = 0.20
N_SPLITS_CV   = 5
GAP           = 1
ACCURACY_BASE = 0.58
PCA_VAR       = 0.95   # erkläre mindestens 95 % Varianz in PCA

# 48 × 2 × 3 × 4 = 192 Kombinationen × 5 Folds = 960 Fits pro Richtung
PARAM_GRID = {
    'select__k':        [5, 7, 10, 'all'],
    'knn__n_neighbors': [3, 5, 7, 10, 15, 20, 30, 50],
    'knn__weights':     ['uniform', 'distance'],
    'knn__metric':      ['euclidean', 'manhattan', 'chebyshev'],
}

# Basis-Features (aus stocks_gpr_features.csv)
_BASE_A = ['GPRD_monthly_pct', 'gpr_lag1', 'gpr_lag2', 'gpr_lag3',
           'GPRD_ACT_monthly_pct', 'GPRD_THREAT_monthly_pct', 'gprd_zscore']
_BASE_B = ['Stock_monthly_pct', 'stock_lag1', 'stock_lag2', 'stock_lag3', 'stock_vol6']

# Erweitert durch engineer_features(); wird dort gesetzt
FEATS_A: list = []
FEATS_B: list = []
TARGET_A = 'stock_down'
TARGET_B = 'gpr_up_next'

# ── 1. Datenaufbereitung + Feature Engineering ──────────────────────────────

def prepare():
    df = pd.read_csv(DATA)
    df['YearMonth'] = df['YearMonth'].astype(str)
    df = df[(df['Index'].isin(KEEP)) &
            (df['YearMonth'] >= START) &
            (df['YearMonth'] <= END)].copy()
    df = df.sort_values(['YearMonth', 'Index']).reset_index(drop=True)
    return df


def engineer_features(df):
    """Leitet Features arithmetisch aus den Lag-Spalten ab (kein Index-Leak)."""
    # ── GPR-Dynamik ──────────────────────────────────────────────────────────
    df['gpr_momentum']        = df['GPRD_monthly_pct'] - df['gpr_lag1']
    df['gpr_momentum2']       = df['gpr_lag1'] - df['gpr_lag2']
    df['gpr_accel']           = df['GPRD_monthly_pct'] - 2*df['gpr_lag1'] + df['gpr_lag2']
    df['gpr_cumchange_3']     = df['GPRD_monthly_pct'] + df['gpr_lag1'] + df['gpr_lag2']

    # ── ACT vs. THREAT Divergenz & Interaktion ───────────────────────────────
    df['act_vs_threat']       = df['GPRD_ACT_monthly_pct'] - df['GPRD_THREAT_monthly_pct']
    df['gpr_act_x_zscore']    = df['GPRD_ACT_monthly_pct']    * df['gprd_zscore']
    df['gpr_threat_x_zscore'] = df['GPRD_THREAT_monthly_pct'] * df['gprd_zscore']

    # ── Aktien-Dynamik ───────────────────────────────────────────────────────
    df['stock_momentum']      = df['Stock_monthly_pct'] - df['stock_lag1']
    df['stock_momentum2']     = df['stock_lag1'] - df['stock_lag2']
    df['stock_accel']         = df['Stock_monthly_pct'] - 2*df['stock_lag1'] + df['stock_lag2']
    df['stock_cumchange_3']   = df['Stock_monthly_pct'] + df['stock_lag1'] + df['stock_lag2']
    df['stock_x_vol']         = df['Stock_monthly_pct'] * df['stock_vol6']
    df['stock_down_lag1']     = (df['stock_lag1'] < 0).astype(int)
    df['stock_down_lag2']     = (df['stock_lag2'] < 0).astype(int)
    df['stock_bear_3m']       = (df['stock_cumchange_3'] < 0).astype(int)

    # ── Kreuz-Features (GPR × Aktien) ────────────────────────────────────────
    df['gpr_x_stock']         = df['GPRD_monthly_pct'] * df['Stock_monthly_pct']
    df['vol_x_zscore']        = df['stock_vol6'] * df['gprd_zscore']

    return df


def set_feature_lists(df):
    """Setzt FEATS_A und FEATS_B global nach dem Feature-Engineering."""
    global FEATS_A, FEATS_B
    # gpr_x_stock ausgeschlossen: enthält Stock_monthly_pct, aus dem
    # stock_down direkt abgeleitet wird → würde Datenleck erzeugen.
    FEATS_A = [c for c in _BASE_A + [
        'gpr_momentum', 'gpr_momentum2', 'gpr_accel', 'gpr_cumchange_3',
        'act_vs_threat', 'gpr_act_x_zscore', 'gpr_threat_x_zscore',
        'vol_x_zscore',
    ] if c in df.columns]

    FEATS_B = [c for c in _BASE_B + [
        'stock_momentum', 'stock_momentum2', 'stock_accel', 'stock_cumchange_3',
        'stock_x_vol', 'stock_down_lag1', 'stock_down_lag2', 'stock_bear_3m',
        'gpr_x_stock', 'vol_x_zscore',
    ] if c in df.columns]


def check_features(df, feats, target, direction):
    missing = [c for c in feats + [target] if c not in df.columns]
    if missing:
        print(f"  [WARNUNG] Richtung {direction}: Fehlende Spalten {missing} — wird übersprungen")
        return False
    return True


def temporal_split(df, feats, target):
    sub = df[feats + [target]].dropna().reset_index(drop=True)
    n_test = max(1, int(len(sub) * HOLDOUT_FRAC))
    X = sub[feats].values
    y = sub[target].values
    return X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]

# ── 2. GridSearch + Evaluation ──────────────────────────────────────────────

def _mi_func(X, y):
    """Wrapper mit festem random_state für reproduzierbare MI-Scores."""
    return mutual_info_classif(X, y, random_state=42)


def run_gridsearch(X_train, y_train):
    """GridSearch über die KNN-Pipeline (siehe Modul-Docstring); TimeSeriesSplit."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(_mi_func)),
        ('pca',    PCA(n_components=PCA_VAR, random_state=42)),
        ('knn',    KNeighborsClassifier(n_jobs=-1)),
    ])
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV, gap=GAP)
    gs = GridSearchCV(
        pipe, PARAM_GRID,
        cv=tscv, scoring='roc_auc',
        n_jobs=-1, refit=True, verbose=0,
    )
    gs.fit(X_train, y_train)
    return gs


def evaluate(gs, X_test, y_test, feats):
    y_pred  = gs.predict(X_test)
    y_proba = gs.predict_proba(X_test)[:, 1]

    # Ausgewählte Features aus dem besten Estimator extrahieren
    selector = gs.best_estimator_.named_steps['select']
    selected = [feats[i] for i in range(len(feats)) if selector.get_support()[i]]
    mi_scores = dict(zip(feats, selector.scores_))

    # PCA-Komponenten nach Selektion
    pca_step = gs.best_estimator_.named_steps['pca']
    n_pca_components = pca_step.n_components_

    return {
        'cv_auc':           gs.best_score_,
        'roc_auc':          roc_auc_score(y_test, y_proba),
        'f1':               f1_score(y_test, y_pred),
        'accuracy':         accuracy_score(y_test, y_pred),
        'cm':               confusion_matrix(y_test, y_pred),
        'best_params':      gs.best_params_,
        'selected_feats':   selected,
        'mi_scores':        mi_scores,
        'n_pca':            n_pca_components,
        'y_test':           y_test,
        'y_pred':           y_pred,
        'y_proba':          y_proba,
    }

# ── 3. Grafiken ─────────────────────────────────────────────────────────────

def plot_roc_curves(res_a, res_b, outfile):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    configs = [
        (axes[0], res_a, 'A', '#C00000', 'Richtung A: GPR-Features\n→ Aktien fallen?'),
        (axes[1], res_b, 'B', '#2E75B6', 'Richtung B: Aktien-Features\n→ GPR steigt (t+1)?'),
    ]
    for ax, res, _d, color, title in configs:
        fpr, tpr, _ = roc_curve(res['y_test'], res['y_proba'])
        ax.plot(fpr, tpr, color=color, lw=2.2,
                label=f"KNN  AUC = {res['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label="Zufall  AUC = 0.500")
        ax.fill_between(fpr, tpr, alpha=0.10, color=color)

        delta = res['roc_auc'] - 0.5
        sign  = '+' if delta >= 0 else ''
        p = res['best_params']
        ax.set_title(
            f"{title}\nCV-AUC={res['cv_auc']:.3f}  Test-AUC={res['roc_auc']:.3f}  Δ={sign}{delta:.3f}\n"
            f"k={p['knn__n_neighbors']}  sel={p['select__k']}  PCA={res['n_pca']}",
            fontsize=9, fontweight='bold',
        )
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(alpha=0.25)

    fig.suptitle("KNN (v2 — Feature Engineering + MI-Selektion + PCA): ROC-Kurven",
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(res_a, res_b, outfile):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    configs = [
        (axes[0], res_a, 'A', 'Reds',  ['Steigt (0)', 'Fällt (1)'],
         'Richtung A\nZiel: stock_down'),
        (axes[1], res_b, 'B', 'Blues', ['GPR fällt (0)', 'GPR steigt (1)'],
         'Richtung B\nZiel: gpr_up_next'),
    ]
    for ax, res, _d, cmap, labels, title in configs:
        cm = res['cm'].astype(float)
        cm_pct = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=1, aspect='auto')
        for i in range(2):
            for j in range(2):
                col = 'white' if cm_pct[i, j] > 0.55 else 'black'
                ax.text(j, i, f"{int(res['cm'][i,j])}\n({cm_pct[i,j]:.0%})",
                        ha='center', va='center', fontsize=11, fontweight='bold',
                        color=col)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels([f"Pred: {l}" for l in labels], fontsize=9)
        ax.set_yticklabels([f"True: {l}" for l in labels], fontsize=9)
        ax.set_title(
            f"{title}\nAcc={res['accuracy']:.3f}  F1={res['f1']:.3f}",
            fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.75)

    fig.suptitle("KNN (v2): Konfusionsmatrizen — Richtung A & B",
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(res_a, res_b, outfile):
    """
    MI-Wichtigkeit aller Features beider Richtungen.
    Rot (dunkel) = vom GridSearch ausgewählt, Grau = verworfen.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    configs = [
        (axes[0], res_a, 'A', '#C00000', '#DDDDDD',
         'Richtung A — MI-Score (SelectKBest)'),
        (axes[1], res_b, 'B', '#2E75B6', '#DDDDDD',
         'Richtung B — MI-Score (SelectKBest)'),
    ]

    for ax, res, _d, sel_color, rej_color, title in configs:
        mi = res['mi_scores']
        sel = set(res['selected_feats'])

        # Absteigend nach MI sortieren
        sorted_feats = sorted(mi, key=lambda f: mi[f], reverse=True)
        scores       = [mi[f] for f in sorted_feats]
        colors       = [sel_color if f in sel else rej_color for f in sorted_feats]

        y_pos = range(len(sorted_feats))
        ax.barh(list(y_pos), scores, color=colors, edgecolor='none', height=0.7)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(sorted_feats, fontsize=8)
        ax.set_xlabel("Mutual Information Score", fontsize=10)
        ax.invert_yaxis()

        k    = res['best_params']['select__k']
        npca = res['n_pca']
        ax.set_title(
            f"{title}\n"
            f"Ausgewählt: k={k}  PCA-Komp.: {npca}\n"
            f"Dunkel = selektiert, Grau = verworfen",
            fontsize=9, fontweight='bold',
        )
        ax.grid(axis='x', alpha=0.3)

        # Trennlinie nach k selektierten Features
        if k != 'all' and isinstance(k, int):
            ax.axhline(k - 0.5, color='black', lw=1.2, ls='--', alpha=0.6)

    fig.suptitle("Feature-Wichtigkeit (Mutual Information) — Richtung A & B",
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()

# ── 4. Textbericht ──────────────────────────────────────────────────────────

def save_text_report(res_a, res_b, outfile):
    delta_auc = res_a['roc_auc'] - res_b['roc_auc']

    lines = [
        "=" * 68,
        "  KNN (v2): GPR <-> AKTIENMARKT — FEATURE ENGINEERING + MI + PCA",
        "  Daten   : data/processed/stocks_gpr_features.csv",
        "  Zeitraum: 2000-01 bis 2021-05  |  12 Indizes",
        "=" * 68,
    ]

    for res, direction, feats, target, frage in [
        (res_a, 'A', FEATS_A, TARGET_A,
         "Fallen Aktien wenn GPR-Features hoch?"),
        (res_b, 'B', FEATS_B, TARGET_B,
         "Steigt GPR naechsten Monat wenn Aktien fallen?"),
    ]:
        p  = res['best_params']
        cm = res['cm']
        mi_sorted = sorted(res['mi_scores'].items(), key=lambda x: x[1], reverse=True)

        lines += [
            "",
            "─" * 68,
            f"RICHTUNG {direction}: {frage}",
            f"  Zielvariable  : {target}",
            f"  Features total: {len(feats)}",
            "─" * 68,
            "BESTE HYPERPARAMETER (GridSearch, Scoring: ROC-AUC):",
            f"  select__k       : {p['select__k']}",
            f"  n_pca_components: {res['n_pca']}  (aus PCA(variance={PCA_VAR}))",
            f"  n_neighbors     : {p['knn__n_neighbors']}",
            f"  weights         : {p['knn__weights']}",
            f"  metric          : {p['knn__metric']}",
            "",
            "AUSGEWAEHLTE FEATURES (SelectKBest):",
        ]
        for f in res['selected_feats']:
            lines.append(f"  [+] {f:<28s}  MI={res['mi_scores'][f]:.4f}")

        lines += [
            "",
            "VERWORFENE FEATURES (MI zu gering):",
        ]
        for f in feats:
            if f not in res['selected_feats']:
                lines.append(f"  [-] {f:<28s}  MI={res['mi_scores'][f]:.4f}")

        above = "JA" if res['roc_auc'] > 0.5 else "NEIN"
        lines += [
            "",
            "METRIKEN:",
            f"  CV-AUC   (TimeSeriesSplit n=5, gap=1) : {res['cv_auc']:.4f}",
            f"  Test-AUC (temporaler Holdout 20%)     : {res['roc_auc']:.4f}",
            f"  F1-Score (Holdout)                    : {res['f1']:.4f}",
            f"  Accuracy (Holdout)                    : {res['accuracy']:.4f}",
            f"  Majority-Baseline Accuracy            : {ACCURACY_BASE:.2f}",
            f"  AUC ueber Zufall-Baseline?            : {above} "
            f"(Delta={res['roc_auc']-0.5:+.4f})",
            "",
            "KONFUSIONSMATRIX (Holdout-Testmenge):",
            f"  {'':20s}  Pred 0  Pred 1",
            f"  {'True 0':20s}  {cm[0,0]:6d}  {cm[0,1]:6d}",
            f"  {'True 1':20s}  {cm[1,0]:6d}  {cm[1,1]:6d}",
        ]

    lines += [
        "",
        "=" * 68,
        "ASYMMETRIE-ANALYSE",
        "─" * 68,
        f"  Delta AUC (A - B) = {res_a['roc_auc']:.4f} - {res_b['roc_auc']:.4f} = {delta_auc:+.4f}",
        "",
    ]
    if abs(delta_auc) < 0.02:
        lines.append("  Interpretation: Keine klare Richtungsasymmetrie.")
    elif delta_auc > 0:
        lines.append("  Interpretation: GPR -> Aktien besser klassifizierbar.")
    else:
        lines.append("  Interpretation: Aktien -> GPR besser klassifizierbar.")
        lines.append("  Konsistent mit algo1-3: Aktienmärkte fuehren den GPR.")

    lines += [
        "",
        "PIPELINE:",
        "  StandardScaler -> SelectKBest(MI) -> PCA(0.95) -> KNN",
        "  TimeSeriesSplit(n=5, gap=1) | Temporaler Holdout (20%)",
        "=" * 68,
        "",
    ]
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

# ── 5. Richtung ausführen ───────────────────────────────────────────────────

def run_direction(df, feats, target, direction):
    if not check_features(df, feats, target, direction):
        return None

    X_train, X_test, y_train, y_test = temporal_split(df, feats, target)
    n_total = len(X_train) + len(X_test)

    print(f"  Features: {len(feats)}  |  Beobachtungen: {n_total} "
          f"(Train={len(X_train)}, Test={len(X_test)})")
    print(f"  Klassen Test — 0: {(y_test==0).sum()}  1: {(y_test==1).sum()}")

    n_combis = (len(PARAM_GRID['select__k'])
                * len(PARAM_GRID['knn__n_neighbors'])
                * len(PARAM_GRID['knn__weights'])
                * len(PARAM_GRID['knn__metric']))
    print(f"  GridSearch: {n_combis} Kombinationen × {N_SPLITS_CV} Folds...")

    gs = run_gridsearch(X_train, y_train)

    p = gs.best_params_
    print(f"  Beste Parameter: sel-k={p['select__k']}  "
          f"k={p['knn__n_neighbors']}  "
          f"weights={p['knn__weights']}  metric={p['knn__metric']}")
    print(f"  Bestes CV-AUC:   {gs.best_score_:.4f}")

    res = evaluate(gs, X_test, y_test, feats)
    print(f"  Test-AUC:        {res['roc_auc']:.4f}  "
          f"(Zufall=0.500, Delta={res['roc_auc']-0.5:+.4f})")
    print(f"  PCA-Komponenten: {res['n_pca']}")
    print(f"  Selektierte Features ({len(res['selected_feats'])}): "
          f"{res['selected_feats']}")
    print(f"  Accuracy:        {res['accuracy']:.4f}  "
          f"(Majority-Baseline={ACCURACY_BASE:.2f})")
    print(f"  F1-Score:        {res['f1']:.4f}")
    return res

# ── 6. Main ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 68)
    print("  KNN v2: GPR <-> AKTIENMARKT — FEATURE ENGINEERING + MI + PCA")
    print("=" * 68)

    print(f"\nLade und bereite Daten vor: {DATA}")
    df = prepare()
    df = engineer_features(df)
    set_feature_lists(df)

    print(f"  {len(df)} Beobachtungen | {df['YearMonth'].min()} bis {df['YearMonth'].max()}")
    print(f"  Neue Features: {len(FEATS_A)} für A, {len(FEATS_B)} für B")

    # ── Richtung A ──────────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("RICHTUNG A: GPR-Features -> Aktien fallen? (stock_down)")
    res_a = run_direction(df, FEATS_A, TARGET_A, 'A')

    # ── Richtung B ──────────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("RICHTUNG B: Aktien-Features -> GPR steigt t+1? (gpr_up_next)")
    res_b = run_direction(df, FEATS_B, TARGET_B, 'B')

    if res_a is None or res_b is None:
        print("\nMindestens eine Richtung fehlgeschlagen. Abbruch.")
        return

    # ── Ausgaben ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("Speichere Ergebnisse...")

    roc_path = f"{OUTPUT_DIR}/knn_roc_curves.png"
    cm_path  = f"{OUTPUT_DIR}/knn_confusion_matrices.png"
    fi_path  = f"{OUTPUT_DIR}/knn_feature_importance.png"
    txt_path = f"{OUTPUT_DIR}/knn_results.txt"

    plot_roc_curves(res_a, res_b, roc_path)
    print(f"  Gespeichert: {roc_path}")

    plot_confusion_matrices(res_a, res_b, cm_path)
    print(f"  Gespeichert: {cm_path}")

    plot_feature_importance(res_a, res_b, fi_path)
    print(f"  Gespeichert: {fi_path}")

    save_text_report(res_a, res_b, txt_path)
    print(f"  Gespeichert: {txt_path}")

    # ── Zusammenfassung ──────────────────────────────────────────────────────
    delta = res_a['roc_auc'] - res_b['roc_auc']

    print(f"\n{'='*68}")
    print("  ZUSAMMENFASSUNG")
    print(f"{'─'*68}")
    print(f"  Richtung A — CV-AUC={res_a['cv_auc']:.4f} | "
          f"Test-AUC={res_a['roc_auc']:.4f} | "
          f"F1={res_a['f1']:.4f} | Acc={res_a['accuracy']:.4f}")
    print(f"  Richtung B — CV-AUC={res_b['cv_auc']:.4f} | "
          f"Test-AUC={res_b['roc_auc']:.4f} | "
          f"F1={res_b['f1']:.4f} | Acc={res_b['accuracy']:.4f}")
    print(f"  Delta AUC (A-B): {delta:+.4f}")

    if delta < -0.02:
        print("  Asymmetrie: Aktienmarkt erklaert GPR besser.")
    elif delta > 0.02:
        print("  Asymmetrie: GPR erklaert Aktienmarkt besser.")
    else:
        print("  Keine klare Richtungsasymmetrie (|DeltaAUC| < 0.02).")

    print(f"{'='*68}")
    print("  Analyse abgeschlossen.")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
