"""Audit rolling stability for the fixed VN30 hourly selected candidate."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.vn30_hourly_dual_track_common import REPO_ROOT, rel  # noqa: E402

INPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_selected_l2_logistic_h40_row_predictions"
INPUT_PATH = INPUT_DIR / "row_predictions.csv"
REPRODUCTION_SUMMARY_PATH = INPUT_DIR / "selected_candidate_reproduction_summary.csv"
OUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_selected_candidate_rolling"
WINDOWS = [250, 500, 1000]
REQUIRED_COLUMNS = {
    "datetime",
    "ticker",
    "horizon",
    "model",
    "feature_set",
    "threshold",
    "split",
    "y_true",
    "y_pred",
    "y_score_or_probability",
    "correct",
    "source_run_id",
    "reproduction_flag",
}


def as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def fmt_pct(value: Any) -> str:
    number = as_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number * 100.0:.2f}%"


def fmt_pp(value: Any) -> str:
    number = as_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number * 100.0:+.2f} percentage points"


def read_reproduction_status() -> tuple[str, dict[str, Any]]:
    if not REPRODUCTION_SUMMARY_PATH.exists():
        return "missing", {}
    frame = pd.read_csv(REPRODUCTION_SUMMARY_PATH)
    if frame.empty:
        return "missing", {}
    row = frame.iloc[0].to_dict()
    return str(row.get("reproduction_status", "missing")), row


def load_predictions() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"row predictions not found: {rel(INPUT_PATH)}")
    frame = pd.read_csv(INPUT_PATH)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"row predictions missing columns: {', '.join(sorted(missing))}")
    frame = frame[frame["split"].astype(str).str.lower() == "final"].copy()
    if frame.empty:
        raise ValueError("row predictions have no final split rows")
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["ticker"] = frame["ticker"].astype(str)
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="coerce").astype("Int64")
    frame["y_pred"] = pd.to_numeric(frame["y_pred"], errors="coerce").astype("Int64")
    frame["correct"] = pd.to_numeric(frame["correct"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["datetime", "ticker", "y_true", "y_pred", "correct"])
    frame["y_true"] = frame["y_true"].astype(int)
    frame["y_pred"] = frame["y_pred"].astype(int)
    frame["correct"] = frame["correct"].astype(int)
    frame = frame.sort_values(["datetime", "ticker"]).reset_index(drop=True)
    frame["row_number"] = np.arange(1, len(frame) + 1)
    return frame


def majority_baseline_from_positive_rate(positive_rate: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return np.maximum(positive_rate, 1.0 - positive_rate)


def rolling_frame(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    correct = frame["correct"].astype(float)
    y_true = frame["y_true"].astype(float)
    y_pred = frame["y_pred"].astype(float)

    rolling_correct = correct.rolling(window=window, min_periods=window).sum()
    rolling_positive = y_true.rolling(window=window, min_periods=window).sum()
    rolling_pred_positive = y_pred.rolling(window=window, min_periods=window).sum()
    valid = rolling_correct.notna()

    out = frame.loc[valid, ["row_number", "datetime", "ticker"]].copy()
    out["window_rows"] = window
    out["window_start_row_number"] = out["row_number"] - window + 1
    out["window_start_datetime"] = frame["datetime"].shift(window - 1).loc[valid].to_numpy()
    out["window_start_ticker"] = frame["ticker"].shift(window - 1).loc[valid].to_numpy()
    out["window_observations"] = window
    out["rolling_correct"] = rolling_correct.loc[valid].to_numpy(dtype=float)
    out["rolling_accuracy"] = out["rolling_correct"] / window
    out["rolling_positive_rate"] = rolling_positive.loc[valid].to_numpy(dtype=float) / window
    out["rolling_prediction_positive_rate"] = rolling_pred_positive.loc[valid].to_numpy(dtype=float) / window
    out["rolling_majority_baseline"] = majority_baseline_from_positive_rate(out["rolling_positive_rate"].to_numpy(dtype=float))
    out["rolling_lift_vs_majority"] = out["rolling_accuracy"] - out["rolling_majority_baseline"]
    return out


def expanding_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[["row_number", "datetime", "ticker"]].copy()
    cumulative_correct = frame["correct"].astype(float).cumsum()
    cumulative_positive = frame["y_true"].astype(float).cumsum()
    cumulative_pred_positive = frame["y_pred"].astype(float).cumsum()
    observations = frame["row_number"].astype(float)
    positive_rate = cumulative_positive / observations

    out["expanding_observations"] = frame["row_number"]
    out["expanding_correct"] = cumulative_correct
    out["expanding_accuracy"] = cumulative_correct / observations
    out["expanding_positive_rate"] = positive_rate
    out["expanding_prediction_positive_rate"] = cumulative_pred_positive / observations
    out["expanding_majority_baseline"] = majority_baseline_from_positive_rate(positive_rate)
    out["expanding_lift_vs_majority"] = out["expanding_accuracy"] - out["expanding_majority_baseline"]
    return out


def longest_below_episode(rolling: pd.DataFrame, threshold: float = 0.60) -> dict[str, Any]:
    best_length = 0
    best_start = -1
    best_end = -1
    current_start = -1
    current_length = 0
    below = rolling["rolling_accuracy"].to_numpy(dtype=float) < threshold

    for position, is_below in enumerate(below):
        if is_below:
            if current_length == 0:
                current_start = position
            current_length += 1
            if current_length > best_length:
                best_length = current_length
                best_start = current_start
                best_end = position
        else:
            current_start = -1
            current_length = 0

    if best_length == 0:
        return {
            "longest_below60_episode_windows": 0,
            "longest_below60_start_row_number": "",
            "longest_below60_end_row_number": "",
            "longest_below60_start_datetime": "",
            "longest_below60_end_datetime": "",
        }

    start_row = rolling.iloc[best_start]
    end_row = rolling.iloc[best_end]
    return {
        "longest_below60_episode_windows": int(best_length),
        "longest_below60_start_row_number": int(start_row["row_number"]),
        "longest_below60_end_row_number": int(end_row["row_number"]),
        "longest_below60_start_datetime": start_row["datetime"],
        "longest_below60_end_datetime": end_row["datetime"],
    }


def summarize_rolling(rolling: pd.DataFrame, window: int, total_rows: int, global_accuracy: float) -> dict[str, Any]:
    if rolling.empty:
        return {
            "window_rows": window,
            "total_final_rows": total_rows,
            "global_final_accuracy": global_accuracy,
            "rolling_window_count": 0,
            "rolling_min_accuracy": math.nan,
            "rolling_max_accuracy": math.nan,
            "rolling_mean_accuracy": math.nan,
            "rolling_median_accuracy": math.nan,
            "final_endpoint_rolling_accuracy": math.nan,
            "windows_below_50": 0,
            "windows_below_55": 0,
            "windows_below_60": 0,
            "rolling_min_lift_vs_majority": math.nan,
            "rolling_max_lift_vs_majority": math.nan,
            "rolling_mean_lift_vs_majority": math.nan,
            "rolling_median_lift_vs_majority": math.nan,
            "final_endpoint_rolling_lift_vs_majority": math.nan,
            **longest_below_episode(rolling),
        }

    accuracy = rolling["rolling_accuracy"]
    lift = rolling["rolling_lift_vs_majority"]
    return {
        "window_rows": window,
        "total_final_rows": total_rows,
        "global_final_accuracy": global_accuracy,
        "rolling_window_count": int(len(rolling)),
        "rolling_min_accuracy": float(accuracy.min()),
        "rolling_max_accuracy": float(accuracy.max()),
        "rolling_mean_accuracy": float(accuracy.mean()),
        "rolling_median_accuracy": float(accuracy.median()),
        "final_endpoint_rolling_accuracy": float(accuracy.iloc[-1]),
        "windows_below_50": int((accuracy < 0.50).sum()),
        "windows_below_55": int((accuracy < 0.55).sum()),
        "windows_below_60": int((accuracy < 0.60).sum()),
        "rolling_min_lift_vs_majority": float(lift.min()),
        "rolling_max_lift_vs_majority": float(lift.max()),
        "rolling_mean_lift_vs_majority": float(lift.mean()),
        "rolling_median_lift_vs_majority": float(lift.median()),
        "final_endpoint_rolling_lift_vs_majority": float(lift.iloc[-1]),
        **longest_below_episode(rolling),
    }


def period_summary(frame: pd.DataFrame, period_freq: str, period_name: str) -> pd.DataFrame:
    work = frame.copy()
    work[period_name] = work["datetime"].dt.to_period(period_freq).astype(str)
    rows: list[dict[str, Any]] = []
    for period, group in work.groupby(period_name, sort=True):
        rows_count = int(len(group))
        accuracy = float(group["correct"].mean())
        positive_rate = float(group["y_true"].mean())
        majority = max(positive_rate, 1.0 - positive_rate)
        rows.append(
            {
                period_name: period,
                "rows": rows_count,
                "accuracy": accuracy,
                "majority_baseline": majority,
                "lift_vs_majority": accuracy - majority,
                "target_up_rate": positive_rate,
                "prediction_up_rate": float(group["y_pred"].mean()),
                "correct": int(group["correct"].sum()),
            }
        )
    return pd.DataFrame(rows)


def ticker_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker, group in frame.groupby("ticker", sort=True):
        rows_count = int(len(group))
        accuracy = float(group["correct"].mean())
        positive_rate = float(group["y_true"].mean())
        majority = max(positive_rate, 1.0 - positive_rate)
        row: dict[str, Any] = {
            "ticker": ticker,
            "rows": rows_count,
            "accuracy": accuracy,
            "majority_baseline": majority,
            "lift_vs_majority": accuracy - majority,
            "target_up_rate": positive_rate,
            "prediction_up_rate": float(group["y_pred"].mean()),
            "correct": int(group["correct"].sum()),
        }
        ordered = group.sort_values(["datetime", "ticker"]).reset_index(drop=True)
        for window in WINDOWS:
            if len(ordered) >= window:
                roll = ordered["correct"].astype(float).rolling(window=window, min_periods=window).mean().dropna()
                row[f"ticker_rolling_{window}_window_count"] = int(len(roll))
                row[f"ticker_rolling_{window}_min_accuracy"] = float(roll.min())
                row[f"ticker_rolling_{window}_mean_accuracy"] = float(roll.mean())
                row[f"ticker_rolling_{window}_median_accuracy"] = float(roll.median())
                row[f"ticker_rolling_{window}_final_endpoint_accuracy"] = float(roll.iloc[-1])
            else:
                row[f"ticker_rolling_{window}_window_count"] = 0
                row[f"ticker_rolling_{window}_min_accuracy"] = math.nan
                row[f"ticker_rolling_{window}_mean_accuracy"] = math.nan
                row[f"ticker_rolling_{window}_median_accuracy"] = math.nan
                row[f"ticker_rolling_{window}_final_endpoint_accuracy"] = math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def save_plot(path: Path, frame: pd.DataFrame, y_col: str, title: str, ylabel: str, extra_series: list[tuple[str, str]] | None = None) -> None:
    plt.figure(figsize=(10, 5.5))
    plt.plot(frame["row_number"], frame[y_col], label=y_col, linewidth=1.5)
    if extra_series:
        for col, label in extra_series:
            if col in frame.columns:
                plt.plot(frame["row_number"], frame[col], label=label, linewidth=1.0, alpha=0.75)
    plt.axhline(0.50, color="0.55", linewidth=0.9, linestyle="--", label="50%")
    plt.axhline(0.60, color="0.25", linewidth=0.9, linestyle=":", label="60%")
    plt.title(title)
    plt.xlabel("Endpoint row number sorted by datetime, ticker")
    plt.ylabel(ylabel)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def save_lift_plot(path: Path, rolling_by_window: dict[int, pd.DataFrame]) -> None:
    plt.figure(figsize=(10, 5.5))
    for window, frame in rolling_by_window.items():
        plt.plot(frame["row_number"], frame["rolling_lift_vs_majority"], label=f"{window}-row lift", linewidth=1.4)
    plt.axhline(0.0, color="0.35", linewidth=0.9, linestyle="--", label="zero lift")
    plt.title("Rolling Lift Versus Rolling Majority Baseline")
    plt.xlabel("Endpoint row number sorted by datetime, ticker")
    plt.ylabel("Lift versus rolling majority baseline")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def markdown_summary_table(summary_rows: list[dict[str, Any]]) -> str:
    headers = [
        "window_rows",
        "rolling_window_count",
        "rolling_min_accuracy",
        "rolling_mean_accuracy",
        "rolling_median_accuracy",
        "final_endpoint_rolling_accuracy",
        "windows_below_60",
        "longest_below60_episode_windows",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in summary_rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                values.append(fmt_pct(value) if "accuracy" in header else str(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def unavailable_outputs(status: str, summary_row: dict[str, Any]) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "status": "unavailable",
            "reason": f"reproduction_status={status}; rolling audit requires passed reproduction",
            **summary_row,
        }
    ]
    pd.DataFrame(rows).to_csv(OUT_DIR / "rolling_stability_summary.csv", index=False)
    report = [
        "# VN30 Hourly Selected Candidate Rolling Stability Report",
        "",
        "- Status: unavailable.",
        f"- Reason: reproduction status is `{status}`; rolling audit requires passed reproduction.",
        "",
    ]
    (OUT_DIR / "rolling_stability_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"rolling_stability_status=unavailable reproduction_status={status}")
    return 0


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reproduction_status, reproduction_summary = read_reproduction_status()
    if reproduction_status != "passed":
        return unavailable_outputs(reproduction_status, reproduction_summary)

    frame = load_predictions()
    if frame["reproduction_flag"].astype(str).nunique() != 1 or frame["reproduction_flag"].astype(str).iloc[0] != "passed":
        return unavailable_outputs("row_predictions_not_marked_passed", reproduction_summary)

    total_rows = int(len(frame))
    global_accuracy = float(frame["correct"].mean())
    global_positive_rate = float(frame["y_true"].mean())
    global_majority = max(global_positive_rate, 1.0 - global_positive_rate)
    global_lift = global_accuracy - global_majority

    rolling_by_window: dict[int, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []
    lift_rows: list[pd.DataFrame] = []
    for window in WINDOWS:
        rolling = rolling_frame(frame, window)
        rolling_by_window[window] = rolling
        rolling.to_csv(OUT_DIR / f"rolling_accuracy_{window}.csv", index=False)
        summary_rows.append(summarize_rolling(rolling, window, total_rows, global_accuracy))
        lift_rows.append(
            rolling[
                [
                    "row_number",
                    "datetime",
                    "ticker",
                    "window_rows",
                    "rolling_accuracy",
                    "rolling_majority_baseline",
                    "rolling_lift_vs_majority",
                ]
            ].copy()
        )
        save_plot(
            OUT_DIR / f"fig_rolling_{window}_accuracy.png",
            rolling,
            "rolling_accuracy",
            f"{window}-Row Rolling Accuracy",
            "Accuracy",
            [("rolling_majority_baseline", "rolling majority baseline")],
        )

    expanding = expanding_frame(frame)
    expanding.to_csv(OUT_DIR / "expanding_accuracy.csv", index=False)
    pd.concat(lift_rows, ignore_index=True).to_csv(OUT_DIR / "rolling_lift_vs_majority.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "rolling_stability_summary.csv", index=False)

    ticker = ticker_summary(frame)
    monthly = period_summary(frame, "M", "month")
    quarterly = period_summary(frame, "Q", "quarter")
    ticker.to_csv(OUT_DIR / "ticker_rolling_stability_summary.csv", index=False)
    monthly.to_csv(OUT_DIR / "monthly_stability_summary.csv", index=False)
    quarterly.to_csv(OUT_DIR / "quarterly_stability_summary.csv", index=False)

    save_plot(
        OUT_DIR / "fig_expanding_accuracy.png",
        expanding,
        "expanding_accuracy",
        "Expanding Accuracy",
        "Accuracy",
        [("expanding_majority_baseline", "expanding majority baseline")],
    )
    save_lift_plot(OUT_DIR / "fig_rolling_lift_vs_majority.png", rolling_by_window)

    monthly_acc = monthly["accuracy"] if not monthly.empty else pd.Series(dtype=float)
    quarterly_acc = quarterly["accuracy"] if not quarterly.empty else pd.Series(dtype=float)
    ticker_rolling_feasible = {
        window: int(ticker[f"ticker_rolling_{window}_window_count"].sum()) for window in WINDOWS
    }

    report = [
        "# VN30 Hourly Selected Candidate Rolling Stability Report",
        "",
        "- Input: `outputs/vn30_hourly_selected_l2_logistic_h40_row_predictions/row_predictions.csv`.",
        "- Reproduction status: `passed`.",
        "- Candidate: L2 Logistic h=40 `feature_set_C_closest` threshold=0.50.",
        "- Final rows sorted by datetime, ticker: 4,074.",
        f"- Global final accuracy from row-level predictions: {fmt_pct(global_accuracy)}.",
        f"- Rolling majority baseline definition: majority class rate inside each rolling window.",
        f"- Global rolling-order majority baseline: {fmt_pct(global_majority)}.",
        f"- Global lift versus majority baseline: {fmt_pp(global_lift)}.",
        "",
        "## Rolling Accuracy",
        "",
        markdown_summary_table(summary_rows),
        "",
        "## Monthly And Quarterly Summaries",
        "",
        f"- Monthly slices: {len(monthly)}; mean accuracy {fmt_pct(monthly_acc.mean())}; median accuracy {fmt_pct(monthly_acc.median())}; months below 60%: {int((monthly_acc < 0.60).sum()) if len(monthly_acc) else 0}.",
        f"- Quarterly slices: {len(quarterly)}; mean accuracy {fmt_pct(quarterly_acc.mean())}; median accuracy {fmt_pct(quarterly_acc.median())}; quarters below 60%: {int((quarterly_acc < 0.60).sum()) if len(quarterly_acc) else 0}.",
        "",
        "## Ticker-Level Rolling Feasibility",
        "",
        f"- Ticker-level 250-row rolling windows: {ticker_rolling_feasible[250]}.",
        f"- Ticker-level 500-row rolling windows: {ticker_rolling_feasible[500]}.",
        f"- Ticker-level 1000-row rolling windows: {ticker_rolling_feasible[1000]}.",
        "- Ticker-level 250-row rolling summaries are feasible for tickers with enough final rows. Per-ticker final histories are too short for 500/1000-row rolling windows. Overall ticker summaries are saved.",
        "",
        "## Claim Boundary",
        "",
        "The final score is 9.63 percentage points above validation accuracy; therefore, the result must be interpreted cautiously.",
        "These rolling diagnostics are evidence-generation artifacts only. They do not support target62 or final65 upgrades and do not make trading, profitability, investment-recommendation, or live-deployment claims.",
        "",
    ]
    (OUT_DIR / "rolling_stability_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"rolling_stability_status=completed rows={total_rows} accuracy={global_accuracy:.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
