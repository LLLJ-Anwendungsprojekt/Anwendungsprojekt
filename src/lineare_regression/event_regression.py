"""
Event Study + Lineare Regression
GPR-Schock -> Aktienreaktion (A) | Aktien-Crash -> GPR-Reaktion (B)

Erzeugt genau 4 PDFs:
  results/event_study_richtung_A.pdf
  results/event_study_richtung_B.pdf
  results/event_study_regression_B.pdf
  results/event_study_act_vs_threat.pdf
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats as scistats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

# ── Konfiguration ──────────────────────────────────────────────────────────────
DATA = "data/processed/stocks_gpr_daily.csv"
KEEP = ["000001.SS", "399001.SZ", "GDAXI", "GSPTSE", "HSI", "IXIC",
        "KS11", "N100", "N225", "NYA", "SSMI", "TWII"]
START, END   = "2000-01-01", "2021-05-31"
OUTPUT_DIR   = "results/lineare_regression"
EVENT_WINDOW = 5
ESTIM_GAP    = 1
ESTIM_LEN    = 25
SHOCK_Q      = 0.95
HAC_LAGS     = 5

AR_COLS  = [f"AR_{d:+d}" for d in range(-EVENT_WINDOW, EVENT_WINDOW + 1)]
REL_DAYS = np.arange(-EVENT_WINDOW, EVENT_WINDOW + 1)


# ── 1. Daten laden ─────────────────────────────────────────────────────────────
def prepare():
    assert os.path.isfile(DATA), f"Datendatei nicht gefunden: {DATA}"
    df = pd.read_csv(DATA)
    missing_cols = {"Date", "Index", "Close", "GPRD", "GPRD_ACT", "GPRD_THREAT"} - set(df.columns)
    assert not missing_cols, f"Fehlende Spalten: {missing_cols}"

    df["Date"] = pd.to_datetime(df["Date"])
    df = (df[(df["Index"].isin(KEEP)) & (df["Date"] >= START) & (df["Date"] <= END)]
          .sort_values(["Index", "Date"]).copy())

    missing_idx = set(KEEP) - set(df["Index"].unique())
    assert not missing_idx, f"Fehlende Indizes: {missing_idx}"

    for pct_col, base_col in [("Stock_daily_pct", "Close"), ("GPRD_daily_pct", "GPRD"),
                               ("GPRD_ACT_daily_pct", "GPRD_ACT"), ("GPRD_THREAT_daily_pct", "GPRD_THREAT")]:
        df[pct_col] = df.groupby("Index")[base_col].pct_change() * 100
        lo, hi = df[pct_col].quantile(0.01), df[pct_col].quantile(0.99)
        df[pct_col] = df[pct_col].clip(lo, hi)

    df = df.dropna(subset=["Stock_daily_pct", "GPRD_daily_pct"])
    assert len(df) > 10_000, f"Zu wenige Zeilen: {len(df)}"
    assert df.duplicated(["Index", "Date"]).sum() == 0, "Duplikate (Index, Date)"
    assert df["Date"].min() <= pd.Timestamp("2001-01-01"), "Startdatum unerwartet spaet"
    assert df["Date"].max() >= pd.Timestamp("2020-01-01"), "Enddatum unerwartet frueh"
    return df


# ── 2. Events extrahieren ──────────────────────────────────────────────────────
def extract_events(df, shock_col, target_col, top=True):
    """Extrahiert Event-Fenster: obere (top=True) oder untere (top=False) SHOCK_Q-Schocks."""
    threshold = df[shock_col].quantile(SHOCK_Q if top else 1 - SHOCK_Q)
    pre_buf   = ESTIM_GAP + ESTIM_LEN + EVENT_WINDOW
    events    = []

    for _, grp in df.groupby("Index"):
        grp  = grp.sort_values("Date").reset_index(drop=True)
        vals = grp[shock_col].values
        hits = vals >= threshold if top else vals <= threshold
        for ei in np.where(hits)[0]:
            if ei < pre_buf or ei + EVENT_WINDOW >= len(grp):
                continue
            est = grp[target_col].iloc[ei - pre_buf : ei - EVENT_WINDOW - ESTIM_GAP]
            if est.isna().any() or len(est) < ESTIM_LEN:
                continue
            win = grp[target_col].iloc[ei - EVENT_WINDOW : ei + EVENT_WINDOW + 1].values
            if np.isnan(win).any():
                continue
            events.append({"shock_mag": grp[shock_col].iloc[ei],
                           **dict(zip(AR_COLS, win - est.mean()))})

    ev = pd.DataFrame(events)
    ev["CAR_post"] = ev[[f"AR_{d:+d}" for d in range(1, EVENT_WINDOW + 1)]].sum(axis=1)

    assert len(ev) >= 50, f"Zu wenige Events ({shock_col}, top={top}): {len(ev)}"
    assert np.isfinite(ev[AR_COLS].values).all(), "Nicht-finite AR-Werte"
    assert ev["CAR_post"].notna().all(), "NaN in CAR_post"
    return ev, threshold


# ── 3. Event-Study-Grafik ──────────────────────────────────────────────────────
def plot_event_study(ev, direction, outfile):
    ar_mat   = ev[AR_COLS].values
    n        = len(ar_mat)
    mean_ar  = ar_mat.mean(axis=0)
    sem_ar   = ar_mat.std(axis=0, ddof=1) / np.sqrt(n)
    mean_car = mean_ar.cumsum()
    ci_ar    = 1.96 * sem_ar
    ci_car   = 1.96 * np.sqrt((sem_ar ** 2).cumsum())

    if direction == "A":
        target, title = "Aktien", (
            f"Event Study A: Aktienreaktion auf GPR-Schock\n"
            f"n = {n} Events  |  Schock-Schwelle: {SHOCK_Q*100:.0f}.-Perzentil GPR")
    else:
        target, title = "GPR", (
            f"Event Study B: GPR-Reaktion auf Aktien-Crash\n"
            f"n = {n} Events  |  Schock-Schwelle: {(1-SHOCK_Q)*100:.0f}.-Perzentil Aktien")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.bar(REL_DAYS, mean_ar, color="#2E75B6", alpha=0.75, label=f"Ø AR {target}")
    ax1.errorbar(REL_DAYS, mean_ar, yerr=ci_ar, fmt="none", ecolor="#1F4E79", capsize=3, lw=1.0)
    ax1.axhline(0, color="gray", lw=0.6, ls="--")
    ax1.axvline(0, color="#C00000", lw=1.2, ls=":", label="Event-Tag (t=0)")
    ax1.set_xlabel("Tage relativ zum Event", fontsize=11)
    ax1.set_ylabel(f"Ø Abnormal Return {target} (%)", fontsize=11)
    ax1.set_title("Tägliche durchschnittliche AR (±95% KI)", fontsize=11, fontweight="bold")
    ax1.set_xticks(REL_DAYS); ax1.legend(fontsize=9); ax1.grid(alpha=0.2)

    ax2.plot(REL_DAYS, mean_car, color="#C00000", lw=2.2, marker="o", label=f"Ø CAR {target}")
    ax2.fill_between(REL_DAYS, mean_car - ci_car, mean_car + ci_car,
                     color="#C00000", alpha=0.15, label="95%-Konfidenzband")
    ax2.axhline(0, color="gray", lw=0.6, ls="--")
    ax2.axvline(0, color="#1F4E79", lw=1.2, ls=":", label="Event-Tag (t=0)")
    ax2.set_xlabel("Tage relativ zum Event", fontsize=11)
    ax2.set_ylabel(f"Ø Kumulatives AR {target} (%)", fontsize=11)
    ax2.set_title("Kumuliertes CAR im Ereignisfenster", fontsize=11, fontweight="bold")
    ax2.set_xticks(REL_DAYS); ax2.legend(fontsize=9); ax2.grid(alpha=0.2)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close()

    t_pval = scistats.ttest_1samp(ev["CAR_post"], 0).pvalue
    print(f"  n={n}, CAR_post={ev['CAR_post'].mean():+.4f}%, p={t_pval:.3e}")


# ── 4. Regression B: CAR_post ~ shock_mag ─────────────────────────────────────
def plot_regression_B(ev, outfile):
    sub = ev.dropna(subset=["shock_mag", "CAR_post"])
    x, y = sub["shock_mag"].values, sub["CAR_post"].values
    assert len(x) >= 30, f"Zu wenige Punkte fuer Regression: {len(x)}"

    m = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    slope, intercept, r2, p = m.params[1], m.params[0], m.rsquared, m.pvalues[1]
    x_line = np.linspace(x.min(), x.max(), 200)
    pred   = m.get_prediction(sm.add_constant(x_line)).summary_frame(alpha=0.05)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, s=24, alpha=0.55, color="#2E75B6", edgecolors="none", label="Events")
    ax.plot(x_line, pred["mean"], color="#C00000", lw=2.2,
            label=f"Regressionsgerade (y = {slope:.4f}·x + {intercept:.3f})")
    ax.fill_between(x_line, pred["mean_ci_lower"], pred["mean_ci_upper"],
                    color="#C00000", alpha=0.15, label="95%-Konfidenzband")
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.axvline(0, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("Aktien-Schock am Event-Tag (%)", fontsize=11)
    ax.set_ylabel("CAR GPR (t+1 … t+5, %)", fontsize=11)
    ax.set_title(f"Regression B: GPR-CAR nach Aktien-Schock\n"
                 f"R² = {r2:.4f}  |  p(Steigung) = {p:.2e}  |  n = {len(x)}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close()
    print(f"  slope={slope:+.5f}, R²={r2:.4f}, p={p:.3e}, n={len(x)}")


# ── 5. ACT vs. THREAT ─────────────────────────────────────────────────────────
def plot_act_vs_threat(ev_act, ev_thr, outfile):
    def _stats(ev):
        ar_mat   = ev[AR_COLS].values
        sem_ar   = ar_mat.std(axis=0, ddof=1) / np.sqrt(len(ev))
        mean_car = ar_mat.mean(axis=0).cumsum()
        ci_car   = 1.96 * np.sqrt((sem_ar ** 2).cumsum())
        assert np.isfinite(mean_car).all(), "Nicht-finite CAR-Werte"
        t_pval   = scistats.ttest_1samp(ev["CAR_post"], 0).pvalue
        sub = ev.dropna(subset=["shock_mag", "CAR_post"])
        m   = sm.OLS(sub["CAR_post"].values,
                     sm.add_constant(sub["shock_mag"].values)
                     ).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        return {"n": len(ev), "mean_car": mean_car, "ci_car": ci_car,
                "car_post_avg": ev["CAR_post"].mean(), "t_pval": t_pval,
                "slope": m.params[1], "p_slope": m.pvalues[1], "r2": m.rsquared}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, (label, ev) in enumerate([("ACT", ev_act), ("THREAT", ev_thr)]):
        s     = _stats(ev)
        color = "#C00000" if label == "ACT" else "#2E75B6"
        sig   = "***" if s["t_pval"] < 0.01 else "**" if s["t_pval"] < 0.05 \
                else "*" if s["t_pval"] < 0.1 else "n.s."
        print(f"  {label}: n={s['n']}, CAR_post={s['car_post_avg']:+.3f}%, "
              f"slope={s['slope']:+.4f}, R²={s['r2']:.4f}, p={s['p_slope']:.3e}")

        ax0 = axes[0, col]
        ax0.plot(REL_DAYS, s["mean_car"], color=color, lw=2.2,
                 marker="o", ms=4, label=f"Ø CAR Aktien ({label})")
        ax0.fill_between(REL_DAYS, s["mean_car"] - s["ci_car"], s["mean_car"] + s["ci_car"],
                         color=color, alpha=0.15, label="95%-KI")
        ax0.axhline(0, color="gray", lw=0.6, ls="--")
        ax0.axvline(0, color="black", lw=0.8, ls=":", label="Event-Tag")
        ax0.set_title(f"GPRD_{label}-Schock → Aktien-CAR\n"
                      f"n={s['n']}  |  Ø CAR_post={s['car_post_avg']:+.3f}%  |  {sig}",
                      fontsize=10, fontweight="bold")
        ax0.set_xlabel("Tage relativ zum Event", fontsize=9)
        ax0.set_ylabel("Ø Kumulatives AR Aktien (%)", fontsize=9)
        ax0.set_xticks(REL_DAYS); ax0.legend(fontsize=8); ax0.grid(alpha=0.2)

        ax1 = axes[1, col]
        sub    = ev.dropna(subset=["shock_mag", "CAR_post"])
        x, y   = sub["shock_mag"].values, sub["CAR_post"].values
        x_line = np.linspace(x.min(), x.max(), 200)
        m      = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        pred   = m.get_prediction(sm.add_constant(x_line)).summary_frame(alpha=0.05)

        line_color = "#111111"  # kontrastiert klar mit den farbigen Streupunkten dahinter
        ax1.scatter(x, y, s=10, alpha=0.3, color=color, edgecolors="none", label="Events")
        ax1.plot(x_line, pred["mean"], color=line_color, lw=2.4,
                 label=f"β={s['slope']:+.4f}  p={s['p_slope']:.2e}")
        ax1.fill_between(x_line, pred["mean_ci_lower"], pred["mean_ci_upper"],
                         color=line_color, alpha=0.15, label="95%-KI")
        ax1.axhline(0, color="gray", lw=0.6, ls="--")
        ax1.axvline(0, color="gray", lw=0.6, ls="--")
        ax1.set_title(f"Regression: CAR_post ~ GPRD_{label}-Schock\n"
                      f"R²={s['r2']:.4f}  |  p(β)={s['p_slope']:.2e}",
                      fontsize=10, fontweight="bold")
        ax1.set_xlabel(f"GPRD_{label} % Veränderung am Event-Tag", fontsize=9)
        ax1.set_ylabel("CAR Aktien (t+1 … t+5, %)", fontsize=9)
        ax1.legend(fontsize=8); ax1.grid(alpha=0.2)

    fig.suptitle(
        f"GPRD_ACT vs. GPRD_THREAT: Reagieren Aktien unterschiedlich?\n"
        f"Schock-Schwelle: {SHOCK_Q*100:.0f}.-Perzentil je Komponente  |  "
        f"Fenster ±{EVENT_WINDOW} Tage",
        fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Lade Daten...")
    df = prepare()
    print(f"  {len(df):,} Beobachtungen | {df['Index'].nunique()} Indizes | "
          f"{df['Date'].min().date()} - {df['Date'].max().date()}")

    print("\nRichtung A: GPR-Schock -> Aktien")
    ev_a, thr_a = extract_events(df, "GPRD_daily_pct", "Stock_daily_pct", top=True)
    print(f"  {len(ev_a)} Events (Schwelle={thr_a:+.3f})")
    plot_event_study(ev_a, "A", f"{OUTPUT_DIR}/event_study_richtung_A.pdf")

    print("\nRichtung B: Aktien-Crash -> GPR")
    ev_b, thr_b = extract_events(df, "Stock_daily_pct", "GPRD_daily_pct", top=False)
    print(f"  {len(ev_b)} Events (Schwelle={thr_b:+.3f})")
    plot_event_study(ev_b, "B", f"{OUTPUT_DIR}/event_study_richtung_B.pdf")
    plot_regression_B(ev_b, f"{OUTPUT_DIR}/event_study_regression_B.pdf")

    print("\nACT vs. THREAT:")
    ev_act, thr_act = extract_events(df, "GPRD_ACT_daily_pct",    "Stock_daily_pct", top=True)
    ev_thr, thr_thr = extract_events(df, "GPRD_THREAT_daily_pct", "Stock_daily_pct", top=True)
    print(f"  ACT {len(ev_act)} Events (Schwelle={thr_act:+.3f}), "
          f"THREAT {len(ev_thr)} Events (Schwelle={thr_thr:+.3f})")
    plot_act_vs_threat(ev_act, ev_thr, f"{OUTPUT_DIR}/event_study_act_vs_threat.pdf")

    print("\nPruefe Output-Dateien...")
    for path in [f"{OUTPUT_DIR}/event_study_richtung_A.pdf",
                 f"{OUTPUT_DIR}/event_study_richtung_B.pdf",
                 f"{OUTPUT_DIR}/event_study_regression_B.pdf",
                 f"{OUTPUT_DIR}/event_study_act_vs_threat.pdf"]:
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        assert size > 5_000, f"Output fehlt oder zu klein: {path} ({size} Bytes)"
        print(f"  OK  {path}  ({size:,} Bytes)")

    print(f"\nFertig. Alle 4 PDFs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()