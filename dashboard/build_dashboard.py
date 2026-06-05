"""
Build-Skript für das statische HTML-Dashboard "GPR & Aktienmärkte".

Liest die verarbeiteten CSV-Dateien, berechnet alle Analysen
(K-Means, KNN, Event-Studie, Korrelationen, Regionen) und schreibt das
Ergebnis als JavaScript-Datei `data.js` (window.DASHBOARD_DATA = {...}).

Die `index.html` lädt `data.js` per <script>-Tag und rendert daraus mit
Plotly.js alle Diagramme. Dadurch ist das Dashboard vollständig statisch
und ohne Server / Python-Backend lauffähig (Doppelklick auf index.html).

Aufruf aus dem Projektverzeichnis:
    python dashboard/build_dashboard.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scistats

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix, roc_curve

warnings.filterwarnings("ignore")

# ── Pfade ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / "data" / "processed" / "stocks_gpr_features.csv"
DAILY_CSV = ROOT / "data" / "processed" / "stocks_gpr_daily.csv"
OUT_JS = Path(__file__).resolve().parent / "data.js"

# ── Stammdaten ───────────────────────────────────────────────────────────────
INDICES = {
    "000001.SS": ("Shanghai SSE", "China"),
    "399001.SZ": ("Shenzhen SZSE", "China"),
    "GDAXI": ("DAX", "Europa"),
    "N100": ("Euronext 100", "Europa"),
    "SSMI": ("Swiss SMI", "Europa"),
    "GSPTSE": ("TSX Composite", "Nordamerika"),
    "IXIC": ("NASDAQ", "Nordamerika"),
    "NYA": ("NYSE Composite", "Nordamerika"),
    "HSI": ("Hang Seng", "Asien-Pazifik"),
    "KS11": ("KOSPI", "Asien-Pazifik"),
    "N225": ("Nikkei 225", "Asien-Pazifik"),
    "TWII": ("TAIEX Taiwan", "Asien-Pazifik"),
}
KEEP = list(INDICES.keys())
REGION_ORDER = ["China", "Europa", "Nordamerika", "Asien-Pazifik"]

EVENTS = [
    {"month": "2001-09", "label": "9/11"},
    {"month": "2003-03", "label": "Irakkrieg"},
    {"month": "2008-09", "label": "Finanzkrise"},
    {"month": "2020-03", "label": "COVID-19"},
]

KMEANS_FEATURES = [
    "GPRD_mean", "GPRD_ACT_mean", "GPRD_THREAT_mean",
    "GPRD_monthly_pct", "Stock_monthly_pct",
    "gpr_lag1", "stock_lag1", "stock_vol6", "gprd_zscore",
]
PROFILE_FEATURES = ["GPRD_mean", "Stock_monthly_pct", "stock_vol6", "GPRD_ACT_mean", "GPRD_THREAT_mean"]


def jclean(obj):
    """Rekursiv numpy/pandas Typen in JSON-fähige Python-Objekte umwandeln."""
    if isinstance(obj, dict):
        return {str(k): jclean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jclean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if (np.isnan(obj) or np.isinf(obj)) else round(float(obj), 6)
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else round(obj, 6)
    if isinstance(obj, np.ndarray):
        return jclean(obj.tolist())
    return obj


# ── Daten laden ──────────────────────────────────────────────────────────────
def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV)
    df["YearMonth"] = df["YearMonth"].astype(str)
    df = df[df["Index"].isin(KEEP)].copy()
    df["Region"] = df["Index"].map(lambda i: INDICES[i][1])
    df["Name"] = df["Index"].map(lambda i: INDICES[i][0])
    df = df.sort_values(["YearMonth", "Index"]).reset_index(drop=True)
    return df


def load_daily() -> pd.DataFrame:
    df = pd.read_csv(DAILY_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Index"].isin(KEEP)].copy()
    df = df.sort_values(["Index", "Date"]).reset_index(drop=True)
    return df


# ── Abschnitt 1 + 2: Übersicht / Zeitreihen ──────────────────────────────────
def build_overview(df: pd.DataFrame) -> dict:
    months = sorted(df["YearMonth"].unique())

    by_month = df.groupby("YearMonth")
    gprd = by_month["GPRD_mean"].mean().reindex(months)
    gprd_act = by_month["GPRD_ACT_mean"].mean().reindex(months)
    gprd_threat = by_month["GPRD_THREAT_mean"].mean().reindex(months)
    avg_stock = by_month["Stock_monthly_pct"].mean().reindex(months)

    # KPIs
    spike = df[df["gpr_spike"] == 1]["Stock_monthly_pct"]
    normal = df[df["gpr_spike"] == 0]["Stock_monthly_pct"]
    corr_threat = df[["GPRD_THREAT_monthly_pct", "Stock_monthly_pct"]].corr().iloc[0, 1]
    corr_act = df[["GPRD_ACT_monthly_pct", "Stock_monthly_pct"]].corr().iloc[0, 1]

    # Scatter (pro Region) GPRD_monthly_pct vs Stock_monthly_pct
    sc = df.dropna(subset=["GPRD_monthly_pct", "Stock_monthly_pct"])
    scatter = {}
    for region in REGION_ORDER:
        r = sc[sc["Region"] == region]
        scatter[region] = {
            "x": r["GPRD_monthly_pct"].tolist(),
            "y": r["Stock_monthly_pct"].tolist(),
            "name": r["Name"].tolist(),
            "month": r["YearMonth"].tolist(),
        }
    # OLS-Gerade
    x = sc["GPRD_monthly_pct"].values
    y = sc["Stock_monthly_pct"].values
    slope, intercept, r_val, p_val, _ = scistats.linregress(x, y)
    xline = [float(np.nanmin(x)), float(np.nanmax(x))]
    yline = [intercept + slope * xline[0], intercept + slope * xline[1]]

    return {
        "months": months,
        "gprd_mean": gprd.tolist(),
        "gprd_act_mean": gprd_act.tolist(),
        "gprd_threat_mean": gprd_threat.tolist(),
        "avg_stock_return": avg_stock.tolist(),
        "kpis": {
            "n_months": len(months),
            "n_indices": len(KEEP),
            "n_regions": len(REGION_ORDER),
            "avg_return_spike": float(spike.mean()),
            "avg_return_normal": float(normal.mean()),
            "spike_share": float((df["gpr_spike"] == 1).mean()),
            "corr_threat": float(corr_threat),
            "corr_act": float(corr_act),
        },
        "scatter": scatter,
        "ols": {"x": xline, "y": yline, "slope": float(slope), "r2": float(r_val ** 2), "p": float(p_val)},
        "spike_box": {
            "spike": spike.dropna().tolist(),
            "normal": normal.dropna().tolist(),
        },
    }


def build_timeseries(df: pd.DataFrame, months: list) -> dict:
    # Monatliche rebasierbare Kurse je Index (Close_last)
    close = {}
    pivot = df.pivot_table(index="YearMonth", columns="Index", values="Close_last").reindex(months)
    for code in KEEP:
        if code in pivot.columns:
            close[code] = [None if pd.isna(v) else round(float(v), 4) for v in pivot[code].values]

    # 12-Monats-Rolling-Korrelation GPRD_mean vs Stock_monthly_pct je Region
    rolling = {}
    for region in REGION_ORDER:
        sub = df[df["Region"] == region].groupby("YearMonth").agg(
            gprd=("GPRD_mean", "mean"), ret=("Stock_monthly_pct", "mean")
        ).reindex(months)
        rc = sub["gprd"].rolling(12).corr(sub["ret"])
        rolling[region] = [None if pd.isna(v) else round(float(v), 4) for v in rc.values]

    return {"close": close, "rolling_corr": rolling}


# ── Abschnitt 3: Regionen ─────────────────────────────────────────────────────
def build_regions(df: pd.DataFrame, months: list) -> dict:
    order = sorted(KEEP, key=lambda c: (REGION_ORDER.index(INDICES[c][1]), INDICES[c][0]))
    labels = [INDICES[c][0] for c in order]
    regions = [INDICES[c][1] for c in order]

    # Heatmap der Monatsrenditen
    pivot = df.pivot_table(index="Index", columns="YearMonth", values="Stock_monthly_pct")
    pivot = pivot.reindex(index=order, columns=months)
    z = [[None if pd.isna(v) else round(float(v), 3) for v in row] for row in pivot.values]

    # Korrelationsmatrix der Monatsrenditen
    ret_pivot = df.pivot_table(index="YearMonth", columns="Index", values="Stock_monthly_pct")
    ret_pivot = ret_pivot.reindex(columns=order)
    corr = ret_pivot.corr()
    corr_z = [[round(float(v), 3) for v in row] for row in corr.values]

    # Region-Vergleich
    region_stats = []
    for region in REGION_ORDER:
        r = df[df["Region"] == region]
        gcorr = r[["GPRD_mean", "Stock_monthly_pct"]].corr().iloc[0, 1]
        region_stats.append({
            "region": region,
            "avg_return": float(r["Stock_monthly_pct"].mean()),
            "loss_share": float((r["Stock_monthly_pct"] < 0).mean()),
            "gpr_corr": float(gcorr),
        })

    return {
        "order_codes": order,
        "labels": labels,
        "regions": regions,
        "heatmap_z": z,
        "corr_labels": labels,
        "corr_z": corr_z,
        "region_stats": region_stats,
    }


# ── Abschnitt 4: K-Means ──────────────────────────────────────────────────────
def build_kmeans(df: pd.DataFrame, months: list) -> dict:
    sub = df.copy()
    X = sub[KMEANS_FEATURES].copy()
    imp = SimpleImputer(strategy="median")
    Xi = imp.fit_transform(X)
    Xs = StandardScaler().fit_transform(Xi)

    # Optimales k via Silhouette (2-5)
    from sklearn.metrics import silhouette_score
    best_k, best_sil, sil_scores = 3, -1.0, {}
    for k in range(2, 6):
        labels = KMeans(n_clusters=k, n_init=50, random_state=42).fit_predict(Xs)
        s = silhouette_score(Xs, labels)
        sil_scores[k] = round(float(s), 4)
        if s > best_sil:
            best_sil, best_k = s, k

    model = KMeans(n_clusters=best_k, n_init=50, random_state=42)
    labels = model.fit_predict(Xs)
    sub["cluster"] = labels

    # Cluster nach mittlerer Aktienrendite ordnen (0 = schwächster Markt)
    order = sub.groupby("cluster")["Stock_monthly_pct"].mean().sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    sub["cluster"] = sub["cluster"].map(remap)
    labels = sub["cluster"].values

    # PCA
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(Xs)
    var = pca.explained_variance_ratio_

    pca_pts = {}
    for c in range(best_k):
        m = labels == c
        pca_pts[c] = {
            "x": xy[m, 0].round(3).tolist(),
            "y": xy[m, 1].round(3).tolist(),
            "name": sub.loc[m, "Name"].tolist(),
            "month": sub.loc[m, "YearMonth"].tolist(),
        }

    # Cluster-Profile (z-Score gegen globalen Mittelwert)
    gmean = sub[PROFILE_FEATURES].mean()
    gstd = sub[PROFILE_FEATURES].std().replace(0, np.nan)
    profiles = {}
    for c in range(best_k):
        cm = sub[sub["cluster"] == c][PROFILE_FEATURES].mean()
        profiles[c] = ((cm - gmean) / gstd).round(3).tolist()

    # Cluster-Statistiken
    stats = []
    for c in range(best_k):
        r = sub[sub["cluster"] == c]
        stats.append({
            "cluster": c,
            "n": int(len(r)),
            "avg_return": float(r["Stock_monthly_pct"].mean()),
            "avg_gpr": float(r["GPRD_mean"].mean()),
            "avg_vol": float(r["stock_vol6"].mean()),
            "loss_share": float((r["Stock_monthly_pct"] < 0).mean()),
            "spike_share": float((r["gpr_spike"] == 1).mean()),
        })

    # Regime-Zeitstrahl: dominanter Cluster + Ø Rendite je Monat
    timeline = sub.groupby("YearMonth").agg(
        avg_return=("Stock_monthly_pct", "mean"),
        dominant=("cluster", lambda s: int(s.value_counts().idxmax())),
    ).reindex(months)
    timeline_dominant = [None if pd.isna(v) else int(v) for v in timeline["dominant"].values]
    timeline_return = [None if pd.isna(v) else round(float(v), 4) for v in timeline["avg_return"].values]

    # Automatische Cluster-Labels aus Profilen
    cluster_labels = []
    for c in range(best_k):
        gpr_z = (sub[sub["cluster"] == c]["GPRD_mean"].mean() - gmean["GPRD_mean"]) / gstd["GPRD_mean"]
        ret = sub[sub["cluster"] == c]["Stock_monthly_pct"].mean()
        gpr_word = "Hohe GPR" if gpr_z > 0.3 else ("Niedrige GPR" if gpr_z < -0.3 else "Mittlere GPR")
        ret_word = "starke Märkte" if ret > 0.5 else ("schwache Märkte" if ret < 0 else "neutrale Märkte")
        cluster_labels.append(f"Regime {c}: {gpr_word} / {ret_word}")

    return {
        "best_k": best_k,
        "silhouette": round(float(best_sil), 4),
        "sil_scores": sil_scores,
        "pca_var": [round(float(var[0]), 4), round(float(var[1]), 4)],
        "pca_points": pca_pts,
        "profile_features": PROFILE_FEATURES,
        "profiles": profiles,
        "stats": stats,
        "timeline_months": months,
        "timeline_dominant": timeline_dominant,
        "timeline_return": timeline_return,
        "cluster_labels": cluster_labels,
    }


# ── Abschnitt 5: KNN (echte, reduzierte GridSearch) ──────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gpr_momentum"] = df["GPRD_monthly_pct"] - df["gpr_lag1"]
    df["gpr_momentum2"] = df["gpr_lag1"] - df["gpr_lag2"]
    df["gpr_accel"] = df["GPRD_monthly_pct"] - 2 * df["gpr_lag1"] + df["gpr_lag2"]
    df["gpr_cumchange_3"] = df["GPRD_monthly_pct"] + df["gpr_lag1"] + df["gpr_lag2"]
    df["act_vs_threat"] = df["GPRD_ACT_monthly_pct"] - df["GPRD_THREAT_monthly_pct"]
    df["gpr_act_x_zscore"] = df["GPRD_ACT_monthly_pct"] * df["gprd_zscore"]
    df["gpr_threat_x_zscore"] = df["GPRD_THREAT_monthly_pct"] * df["gprd_zscore"]
    df["stock_momentum"] = df["Stock_monthly_pct"] - df["stock_lag1"]
    df["stock_momentum2"] = df["stock_lag1"] - df["stock_lag2"]
    df["stock_accel"] = df["Stock_monthly_pct"] - 2 * df["stock_lag1"] + df["stock_lag2"]
    df["stock_cumchange_3"] = df["Stock_monthly_pct"] + df["stock_lag1"] + df["stock_lag2"]
    df["stock_x_vol"] = df["Stock_monthly_pct"] * df["stock_vol6"]
    df["stock_down_lag1"] = (df["stock_lag1"] < 0).astype(int)
    df["stock_down_lag2"] = (df["stock_lag2"] < 0).astype(int)
    df["stock_bear_3m"] = (df["stock_cumchange_3"] < 0).astype(int)
    df["gpr_x_stock"] = df["GPRD_monthly_pct"] * df["Stock_monthly_pct"]
    df["vol_x_zscore"] = df["stock_vol6"] * df["gprd_zscore"]
    return df


def run_knn_direction(df: pd.DataFrame, feats: list, target: str) -> dict:
    sub = df[feats + [target]].dropna().reset_index(drop=True)
    n_test = max(1, int(len(sub) * 0.20))
    X, y = sub[feats].values, sub[target].values
    Xtr, Xte, ytr, yte = X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]

    def mi(Xv, yv):
        return mutual_info_classif(Xv, yv, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(mi)),
        ("pca", PCA(n_components=0.95, random_state=42)),
        ("knn", KNeighborsClassifier()),
    ])
    grid = {
        "select__k": [7, "all"],
        "knn__n_neighbors": [5, 7, 10, 20],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    }
    gs = GridSearchCV(pipe, grid, cv=TimeSeriesSplit(n_splits=5, gap=1),
                      scoring="roc_auc", n_jobs=-1, refit=True)
    gs.fit(Xtr, ytr)

    ypred = gs.predict(Xte)
    yproba = gs.predict_proba(Xte)[:, 1]
    sel = gs.best_estimator_.named_steps["select"]
    selected = [feats[i] for i in range(len(feats)) if sel.get_support()[i]]
    mi_scores = {feats[i]: round(float(sel.scores_[i]), 4) for i in range(len(feats))}
    fpr, tpr, _ = roc_curve(yte, yproba)
    cm = confusion_matrix(yte, ypred)
    p = gs.best_params_

    return {
        "cv_auc": round(float(gs.best_score_), 4),
        "test_auc": round(float(roc_auc_score(yte, yproba)), 4),
        "f1": round(float(f1_score(yte, ypred)), 4),
        "accuracy": round(float(accuracy_score(yte, ypred)), 4),
        "best_k": int(p["knn__n_neighbors"]),
        "best_metric": p["knn__metric"],
        "best_weights": p["knn__weights"],
        "select_k": p["select__k"],
        "n_pca": int(gs.best_estimator_.named_steps["pca"].n_components_),
        "selected_features": selected,
        "mi_scores": mi_scores,
        "roc": {"fpr": [round(float(v), 4) for v in fpr], "tpr": [round(float(v), 4) for v in tpr]},
        "cm": [[int(cm[0, 0]), int(cm[0, 1])], [int(cm[1, 0]), int(cm[1, 1])]],
        "n_test": int(n_test),
    }


def build_knn(df_feat: pd.DataFrame) -> dict:
    df = engineer_features(df_feat)
    base_a = ["GPRD_monthly_pct", "gpr_lag1", "gpr_lag2", "gpr_lag3",
              "GPRD_ACT_monthly_pct", "GPRD_THREAT_monthly_pct", "gprd_zscore"]
    feats_a = base_a + ["gpr_momentum", "gpr_momentum2", "gpr_accel", "gpr_cumchange_3",
                        "act_vs_threat", "gpr_act_x_zscore", "gpr_threat_x_zscore", "vol_x_zscore"]
    base_b = ["Stock_monthly_pct", "stock_lag1", "stock_lag2", "stock_lag3", "stock_vol6"]
    feats_b = base_b + ["stock_momentum", "stock_momentum2", "stock_accel", "stock_cumchange_3",
                        "stock_x_vol", "stock_down_lag1", "stock_down_lag2", "stock_bear_3m",
                        "gpr_x_stock", "vol_x_zscore"]

    print("  KNN Richtung A...")
    a = run_knn_direction(df, feats_a, "stock_down")
    a["label"] = "Richtung A: GPR → Aktienmarkt"
    print(f"    Test-AUC={a['test_auc']}  F1={a['f1']}  Acc={a['accuracy']}")
    print("  KNN Richtung B...")
    b = run_knn_direction(df, feats_b, "gpr_up_next")
    b["label"] = "Richtung B: Aktienmarkt → GPR"
    print(f"    Test-AUC={b['test_auc']}  F1={b['f1']}  Acc={b['accuracy']}")
    return {"A": a, "B": b, "delta_auc": round(a["test_auc"] - b["test_auc"], 4)}


# ── Abschnitt 6: Event-Studie (Replikation event_regression.py) ──────────────
EVENT_WINDOW = 5
ESTIM_GAP = 1
ESTIM_LEN = 25
SHOCK_Q = 0.95
AR_COLS = [f"AR_{d:+d}" for d in range(-EVENT_WINDOW, EVENT_WINDOW + 1)]
REL_DAYS = list(range(-EVENT_WINDOW, EVENT_WINDOW + 1))


def prepare_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for pct_col, base_col in [("Stock_daily_pct", "Close"), ("GPRD_daily_pct", "GPRD"),
                              ("GPRD_ACT_daily_pct", "GPRD_ACT"), ("GPRD_THREAT_daily_pct", "GPRD_THREAT")]:
        df[pct_col] = df.groupby("Index")[base_col].pct_change() * 100
        lo, hi = df[pct_col].quantile(0.01), df[pct_col].quantile(0.99)
        df[pct_col] = df[pct_col].clip(lo, hi)
    df = df.dropna(subset=["Stock_daily_pct", "GPRD_daily_pct"])
    return df


def extract_events(df: pd.DataFrame, shock_col: str, target_col: str, top=True):
    threshold = df[shock_col].quantile(SHOCK_Q if top else 1 - SHOCK_Q)
    pre_buf = ESTIM_GAP + ESTIM_LEN + EVENT_WINDOW
    events = []
    for _, grp in df.groupby("Index"):
        grp = grp.sort_values("Date").reset_index(drop=True)
        vals = grp[shock_col].values
        hits = vals >= threshold if top else vals <= threshold
        for ei in np.where(hits)[0]:
            if ei < pre_buf or ei + EVENT_WINDOW >= len(grp):
                continue
            est = grp[target_col].iloc[ei - pre_buf: ei - EVENT_WINDOW - ESTIM_GAP]
            if est.isna().any() or len(est) < ESTIM_LEN:
                continue
            win = grp[target_col].iloc[ei - EVENT_WINDOW: ei + EVENT_WINDOW + 1].values
            if np.isnan(win).any():
                continue
            events.append({"shock_mag": grp[shock_col].iloc[ei], **dict(zip(AR_COLS, win - est.mean()))})
    ev = pd.DataFrame(events)
    ev["CAR_post"] = ev[[f"AR_{d:+d}" for d in range(1, EVENT_WINDOW + 1)]].sum(axis=1)
    return ev, float(threshold)


def event_stats(ev: pd.DataFrame) -> dict:
    ar = ev[AR_COLS].values
    n = len(ar)
    mean_ar = ar.mean(axis=0)
    sem = ar.std(axis=0, ddof=1) / np.sqrt(n)
    car = mean_ar.cumsum()
    ci_ar = 1.96 * sem
    ci_car = 1.96 * np.sqrt((sem ** 2).cumsum())
    car_post = ev["CAR_post"]
    pval = scistats.ttest_1samp(car_post, 0).pvalue
    # Regression CAR_post ~ shock_mag
    sl, ic, r, pr, _ = scistats.linregress(ev["shock_mag"], car_post)
    return {
        "n": int(n),
        "rel_days": REL_DAYS,
        "mean_ar": mean_ar.round(4).tolist(),
        "ci_ar": ci_ar.round(4).tolist(),
        "mean_car": car.round(4).tolist(),
        "ci_car": ci_car.round(4).tolist(),
        "car_post": round(float(car_post.mean()), 4),
        "p": float(pval),
        "reg_slope": round(float(sl), 5),
        "reg_r2": round(float(r ** 2), 4),
        "reg_p": float(pr),
    }


def build_events(daily: pd.DataFrame) -> dict:
    df = prepare_daily(daily)
    print("  Event A: GPR-Schock -> Aktien")
    ev_a, thr_a = extract_events(df, "GPRD_daily_pct", "Stock_daily_pct", top=True)
    print("  Event B: Aktien-Crash -> GPR")
    ev_b, thr_b = extract_events(df, "Stock_daily_pct", "GPRD_daily_pct", top=False)
    print("  Event ACT / THREAT")
    ev_act, _ = extract_events(df, "GPRD_ACT_daily_pct", "Stock_daily_pct", top=True)
    ev_thr, _ = extract_events(df, "GPRD_THREAT_daily_pct", "Stock_daily_pct", top=True)

    # Regression-B Scatter (Teilmenge für Plot)
    rb = ev_b.dropna(subset=["shock_mag", "CAR_post"])
    sl, ic, r, pr, _ = scistats.linregress(rb["shock_mag"], rb["CAR_post"])
    xl = [float(rb["shock_mag"].min()), float(rb["shock_mag"].max())]
    yl = [ic + sl * xl[0], ic + sl * xl[1]]

    return {
        "A": event_stats(ev_a),
        "B": event_stats(ev_b),
        "ACT": event_stats(ev_act),
        "THREAT": event_stats(ev_thr),
        "thr_a": round(thr_a, 3),
        "thr_b": round(thr_b, 3),
        "regB_scatter": {
            "x": rb["shock_mag"].round(3).tolist(),
            "y": rb["CAR_post"].round(3).tolist(),
            "line_x": [round(v, 3) for v in xl],
            "line_y": [round(float(v), 4) for v in yl],
            "slope": round(float(sl), 5),
            "r2": round(float(r ** 2), 4),
            "p": float(pr),
        },
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Lade Daten...")
    df = load_features()
    daily = load_daily()
    months = sorted(df["YearMonth"].unique())

    print("Berechne Übersicht / Zeitreihen...")
    overview = build_overview(df)
    timeseries = build_timeseries(df, months)

    print("Berechne Regionen...")
    regions = build_regions(df, months)

    print("Berechne K-Means...")
    kmeans = build_kmeans(df, months)
    print(f"  Optimales k={kmeans['best_k']} (Silhouette={kmeans['silhouette']})")

    print("Berechne KNN (echte reduzierte GridSearch)...")
    knn = build_knn(df)

    print("Berechne Event-Studie...")
    events = build_events(daily)

    data = {
        "meta": {
            "indices": {c: {"name": INDICES[c][0], "region": INDICES[c][1]} for c in KEEP},
            "region_order": REGION_ORDER,
            "events": EVENTS,
            "months": months,
            "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        },
        "overview": overview,
        "timeseries": timeseries,
        "regions": regions,
        "kmeans": kmeans,
        "knn": knn,
        "events": events,
    }

    data = jclean(data)
    js = "window.DASHBOARD_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n"
    OUT_JS.write_text(js, encoding="utf-8")
    size_kb = OUT_JS.stat().st_size / 1024
    print(f"\nGeschrieben: {OUT_JS}  ({size_kb:.0f} KB)")
    print("Öffne dashboard/index.html im Browser.")


if __name__ == "__main__":
    main()
