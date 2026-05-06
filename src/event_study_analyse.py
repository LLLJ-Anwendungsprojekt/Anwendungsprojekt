"""Event Study fuer den Zusammenhang zwischen Konflikten und Aktienrenditen.

Die Analyse ist indexzentriert:
- Event-Tage werden auf Tagesebene definiert (z. B. hohe Konfliktschwere).
- Abnormale Rendite = Rendite am Tag minus indexspezifischer Mittelwert aus
  dem Schaetzfenster vor dem Event.
- Ausgabe: AAR/CAAR Tabelle, Plot und Summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Event Study auf index_conflict_features")
    parser.add_argument(
        "--data-path",
        default="data/processed/index_conflict_features.csv",
        help="Pfad zur indexzentrierten CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="results_event_study",
        help="Ausgabeordner fuer Tabellen und Plot",
    )
    parser.add_argument(
        "--event-mode",
        choices=["high_fatality", "has_conflict"],
        default="high_fatality",
        help="Definition der Event-Tage",
    )
    parser.add_argument(
        "--fatality-quantile",
        type=float,
        default=0.9,
        help="Quantil fuer high_fatality Event-Definition",
    )
    parser.add_argument("--event-window", type=int, default=5, help="Fenster +/- Handelstage")
    parser.add_argument("--est-window", type=int, default=60, help="Laenge Schaetzfenster")
    parser.add_argument("--gap", type=int, default=5, help="Gap zwischen Schaetz- und Eventfenster")
    parser.add_argument(
        "--min-est-obs",
        type=int,
        default=30,
        help="Minimale Beobachtungen je Index im Schaetzfenster",
    )
    return parser.parse_args()


def get_event_dates(date_level: pd.DataFrame, mode: str, fatality_quantile: float) -> pd.DatetimeIndex:
    if mode == "has_conflict":
        return pd.DatetimeIndex(date_level.loc[date_level["has_conflict"] == 1, "Date"]).sort_values()

    # mode == high_fatality
    conflict_days = date_level[date_level["has_conflict"] == 1].copy()
    if conflict_days.empty:
        return pd.DatetimeIndex([])
    threshold = conflict_days["fatalities_best_sum"].quantile(fatality_quantile)
    event_dates = conflict_days.loc[conflict_days["fatalities_best_sum"] >= threshold, "Date"]
    return pd.DatetimeIndex(event_dates).sort_values()


def run_event_study(
    df: pd.DataFrame,
    event_dates: pd.DatetimeIndex,
    event_window: int,
    est_window: int,
    gap: int,
    min_est_obs: int,
) -> pd.DataFrame:
    returns = (
        df.pivot_table(index="Date", columns="Index", values="daily_return", aggfunc="first")
        .sort_index()
    )
    trade_dates = returns.index
    date_to_pos = {d: i for i, d in enumerate(trade_dates)}

    rel_days = list(range(-event_window, event_window + 1))
    ar_by_rel: dict[int, list[float]] = {r: [] for r in rel_days}

    for event_date in event_dates:
        if event_date not in date_to_pos:
            continue
        pos = date_to_pos[event_date]

        est_end = pos - gap - 1
        est_start = est_end - est_window + 1
        if est_start < 0 or pos + event_window >= len(trade_dates):
            continue

        est_slice = returns.iloc[est_start : est_end + 1]
        est_mean = est_slice.mean(axis=0, skipna=True)
        est_count = est_slice.notna().sum(axis=0)

        valid_indices = est_count[est_count >= min_est_obs].index
        if len(valid_indices) == 0:
            continue

        for rel in rel_days:
            target_pos = pos + rel
            target_ret = returns.iloc[target_pos][valid_indices]
            abnormal = target_ret - est_mean[valid_indices]
            abnormal = abnormal.dropna()
            if not abnormal.empty:
                ar_by_rel[rel].append(float(abnormal.mean()))

    rows = []
    for rel in rel_days:
        vals = np.array(ar_by_rel[rel], dtype=float)
        n = len(vals)
        if n == 0:
            rows.append(
                {
                    "rel_day": rel,
                    "aar": np.nan,
                    "std": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "n_events_used": 0,
                }
            )
            continue

        aar = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        se = std / np.sqrt(n) if n > 1 else np.nan
        t_stat = float(aar / se) if n > 1 and se and not np.isnan(se) else np.nan
        p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))) if n > 1 and not np.isnan(t_stat) else np.nan

        rows.append(
            {
                "rel_day": rel,
                "aar": aar,
                "std": std,
                "t_stat": t_stat,
                "p_value": p_value,
                "n_events_used": n,
            }
        )

    result = pd.DataFrame(rows).sort_values("rel_day").reset_index(drop=True)
    result["caar"] = result["aar"].fillna(0).cumsum()
    return result


def save_outputs(result: pd.DataFrame, out_dir: Path, summary_lines: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "event_study_aar_caar.csv"
    txt_path = out_dir / "event_study_summary.txt"
    png_path = out_dir / "event_study_plot.png"

    result.to_csv(csv_path, index=False)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(result["rel_day"], result["aar"], marker="o")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_ylabel("AAR")
    axes[0].set_title("Average Abnormal Return")

    axes[1].plot(result["rel_day"], result["caar"], marker="o")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_ylabel("CAAR")
    axes[1].set_xlabel("Relativer Handelstag")
    axes[1].set_title("Cumulative Average Abnormal Return")

    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    out_dir = Path(args.output_dir)

    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    date_level = (
        df.groupby("Date", as_index=False)[["has_conflict", "fatalities_best_sum"]]
        .max()
        .sort_values("Date")
    )

    event_dates = get_event_dates(date_level, args.event_mode, args.fatality_quantile)

    result = run_event_study(
        df=df,
        event_dates=event_dates,
        event_window=args.event_window,
        est_window=args.est_window,
        gap=args.gap,
        min_est_obs=args.min_est_obs,
    )

    # zentrale Fenster fuer schnelle Interpretation
    def sum_window(a: int, b: int) -> float:
        sub = result[(result["rel_day"] >= a) & (result["rel_day"] <= b)]["aar"].fillna(0)
        return float(sub.sum())

    caar_m1_p1 = sum_window(-1, 1)
    caar_m5_p5 = sum_window(-args.event_window, args.event_window)

    summary_lines = [
        "Event Study Summary",
        "===================",
        f"data_path: {data_path}",
        f"event_mode: {args.event_mode}",
        f"fatality_quantile: {args.fatality_quantile}",
        f"event_window: +/-{args.event_window}",
        f"est_window: {args.est_window}",
        f"gap: {args.gap}",
        f"min_est_obs: {args.min_est_obs}",
        f"n_unique_event_dates: {len(event_dates)}",
        f"caar_-1_+1: {caar_m1_p1:.6f}",
        f"caar_-{args.event_window}_+{args.event_window}: {caar_m5_p5:.6f}",
        "",
        "Significance by rel_day is in event_study_aar_caar.csv (t_stat, p_value).",
    ]

    save_outputs(result=result, out_dir=out_dir, summary_lines=summary_lines)

    print("Event Study abgeschlossen")
    print(f"Event-Daten: {len(event_dates)}")
    print(f"Output: {out_dir / 'event_study_aar_caar.csv'}")
    print(f"Output: {out_dir / 'event_study_summary.txt'}")
    print(f"Output: {out_dir / 'event_study_plot.png'}")


if __name__ == "__main__":
    main()
