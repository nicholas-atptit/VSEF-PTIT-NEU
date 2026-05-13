"""Validate VN100 regime diagnostics with an ex-ante regime proxy.

The official regime diagnostics are post-hoc artifact rows. This script derives
an ex-ante return-regime proxy from prior realized rows only, then compares it
with the post-hoc regime diagnostics and global benchmark summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ARTIFACT_DIR = REPO_ROOT / "outputs" / "vn100_hybrid_official_2025_confidence_sweep_traincutoff"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "generated" / "evidence_gap_closure"
REGIME_WINDOW = 20
MIN_PRIOR_OBS = 5
BEAR_THRESHOLD = -0.03
BULL_THRESHOLD = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VN100 ex-ante regime validation from existing predictions.")
    parser.add_argument("--artifact-dir", type=Path, default=OFFICIAL_ARTIFACT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_predictions(artifact_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for frequency in ("daily", "hourly"):
        path = artifact_dir / frequency / "predicted_vs_actual.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "frequency" not in frame.columns:
            frame["frequency"] = frequency
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data["timestamp_sort"] = pd.to_datetime(
        data["timestamp"] if "timestamp" in data.columns else data["date"],
        errors="coerce",
    )
    for column in ("horizon", "actual_return", "is_correct"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["timestamp_sort", "horizon", "actual_return", "is_correct"])
    data["model"] = data["model"].astype(str)
    data["frequency"] = data["frequency"].astype(str)
    data["ticker"] = data["ticker"].astype(str)
    return data


def add_exante_regimes(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    data = data.sort_values(["frequency", "model", "horizon", "ticker", "timestamp_sort"]).copy()
    group_cols = ["frequency", "model", "horizon", "ticker"]
    data["prior_actual_return"] = data.groupby(group_cols)["actual_return"].shift(1)
    data["exante_trailing_return"] = (
        data.groupby(group_cols)["prior_actual_return"]
        .rolling(REGIME_WINDOW, min_periods=MIN_PRIOR_OBS)
        .mean()
        .reset_index(level=group_cols, drop=True)
    )
    data["exante_regime"] = "insufficient_history"
    data.loc[data["exante_trailing_return"] <= BEAR_THRESHOLD, "exante_regime"] = "bear"
    data.loc[data["exante_trailing_return"] >= BULL_THRESHOLD, "exante_regime"] = "bull"
    data.loc[
        data["exante_trailing_return"].notna()
        & (data["exante_trailing_return"] > BEAR_THRESHOLD)
        & (data["exante_trailing_return"] < BULL_THRESHOLD),
        "exante_regime",
    ] = "sideways"
    return data


def build_exante_summary(data: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if data.empty:
        return rows
    for (frequency, regime, model, horizon), group in data.groupby(
        ["frequency", "exante_regime", "model", "horizon"], sort=True
    ):
        n_obs = int(len(group))
        accuracy = float(group["is_correct"].mean()) if n_obs else None
        rows.append(
            {
                "frequency": frequency,
                "exante_regime": regime,
                "model": model,
                "horizon": int(horizon),
                "n_obs": n_obs,
                "accuracy": accuracy,
                "passed_60pct": bool(accuracy is not None and accuracy >= 0.60),
                "passed_63pct": bool(accuracy is not None and accuracy >= 0.63),
                "reliable": bool(n_obs >= 50),
            }
        )
    return rows


def posthoc_summary_rows(artifact_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for frequency in ("daily", "hourly"):
        rows.extend(read_csv_rows(artifact_dir / frequency / "regime_accuracy_summary.csv"))
    return rows


def build_comparison(artifact_dir: Path, exante_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exante_index = {
        (
            str(row["frequency"]),
            str(row["model"]),
            int(row["horizon"]),
            str(row["exante_regime"]),
        ): row
        for row in exante_summary
    }
    comparison: list[dict[str, Any]] = []
    for row in posthoc_summary_rows(artifact_dir):
        frequency = row.get("frequency", "")
        model = row.get("model", "")
        horizon = int(float(row.get("horizon", 0)))
        posthoc_regime = row.get("regime", "")
        exante = exante_index.get((frequency, model, horizon, posthoc_regime), {})
        comparison.append(
            {
                "frequency": frequency,
                "model": model,
                "horizon": horizon,
                "posthoc_regime": posthoc_regime,
                "posthoc_n_obs": row.get("n_obs", ""),
                "posthoc_accuracy": row.get("accuracy", ""),
                "posthoc_passed_60pct": row.get("passed_60pct", ""),
                "exante_regime": posthoc_regime if exante else "",
                "exante_n_obs": exante.get("n_obs", ""),
                "exante_accuracy": exante.get("accuracy", ""),
                "exante_passed_60pct": exante.get("passed_60pct", ""),
                "exante_passed_63pct": exante.get("passed_63pct", ""),
                "survives_exante_63pct": bool(exante.get("passed_63pct", False) and exante.get("reliable", False)),
                "comparison_status": "matched_exante_label" if exante else "no_matching_exante_label",
            }
        )

    for frequency in ("daily", "hourly"):
        summary = read_json(artifact_dir / frequency / "benchmark_summary.json")
        comparison.append(
            {
                "frequency": frequency,
                "model": "global",
                "horizon": "all",
                "posthoc_regime": "global",
                "posthoc_n_obs": summary.get("n_predictions", ""),
                "posthoc_accuracy": summary.get("overall_accuracy", ""),
                "posthoc_passed_60pct": summary.get("passed", ""),
                "exante_regime": "global",
                "exante_n_obs": summary.get("n_predictions", ""),
                "exante_accuracy": summary.get("overall_accuracy", ""),
                "exante_passed_60pct": summary.get("passed", ""),
                "exante_passed_63pct": False,
                "survives_exante_63pct": False,
                "comparison_status": "global_not_regime_filtered",
            }
        )
    return comparison


def ticker_stability(data: pd.DataFrame, frequency: str, model: str, horizon: int, regime: str) -> dict[str, Any]:
    subset = data[
        (data["frequency"] == frequency)
        & (data["model"] == model)
        & (data["horizon"] == horizon)
        & (data["exante_regime"] == regime)
    ]
    if subset.empty:
        return {"ticker_count": 0, "reliable_tickers": 0, "tickers_passing_63pct": 0}
    ticker_rows = []
    for ticker, group in subset.groupby("ticker"):
        n_obs = int(len(group))
        accuracy = float(group["is_correct"].mean()) if n_obs else 0.0
        ticker_rows.append({"ticker": ticker, "n_obs": n_obs, "accuracy": accuracy})
    return {
        "ticker_count": len(ticker_rows),
        "reliable_tickers": sum(1 for row in ticker_rows if row["n_obs"] >= 20),
        "tickers_passing_63pct": sum(1 for row in ticker_rows if row["n_obs"] >= 20 and row["accuracy"] >= 0.63),
    }


def markdown_table(headers: list[str], rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        if 0 <= value <= 1:
            return f"{value * 100:.2f}%"
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def write_report(
    path: Path,
    artifact_dir: Path,
    data: pd.DataFrame,
    exante_summary: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    key_rows = [
        row
        for row in exante_summary
        if row["frequency"] == "daily"
        and row["model"] in {"lightgbm", "xgboost"}
        and int(row["horizon"]) == 20
        and row["exante_regime"] == "bear"
    ]
    key_rows = sorted(key_rows, key=lambda row: row["model"])
    stability_rows = [
        {
            "slice": "daily lightgbm h=20 exante bear",
            **ticker_stability(data, "daily", "lightgbm", 20, "bear"),
        },
        {
            "slice": "daily xgboost h=20 exante bear",
            **ticker_stability(data, "daily", "xgboost", 20, "bear"),
        },
    ]
    survives = [row for row in key_rows if row["passed_63pct"] and row["reliable"]]
    content = [
        "# VN100 Ex-Ante Regime Validation Report",
        "",
        "## Source",
        "",
        f"- Official artifact directory: `{rel(artifact_dir)}`.",
        "- Prediction inputs: `daily/predicted_vs_actual.csv` and `hourly/predicted_vs_actual.csv`.",
        "- No model training, provider fetch, or benchmark rerun was performed.",
        "",
        "## Ex-Ante Rule",
        "",
        f"For each ticker/frequency/model/horizon sequence, the ex-ante regime uses the mean of the prior {REGIME_WINDOW}",
        f"realized target returns, shifted by one row. At least {MIN_PRIOR_OBS} prior observations are required. The bear",
        f"threshold is <= {BEAR_THRESHOLD:.2%}; the bull threshold is >= {BULL_THRESHOLD:.2%}; remaining labeled rows are sideways.",
        "Rows without enough prior history are marked `insufficient_history`.",
        "",
        "## Key Bear-Regime Diagnostic Recheck",
        "",
        markdown_table(
            ["frequency", "exante_regime", "model", "horizon", "n_obs", "accuracy", "passed_60pct", "passed_63pct", "reliable"],
            key_rows,
        )
        if key_rows
        else "No ex-ante daily bear h=20 rows were available for LightGBM or XGBoost.",
        "",
        "## Ticker Stability of Key Ex-Ante Bear Slices",
        "",
        markdown_table(["slice", "ticker_count", "reliable_tickers", "tickers_passing_63pct"], stability_rows),
        "",
        "## Required Answers",
        "",
        f"- Bear-regime 63%+ survives ex-ante validation: {'yes for the listed reliable key slice(s)' if survives else 'not established for the key slices'}." ,
        "- Regime effect stability across windows: not established because 2022-2024 official windows are unavailable.",
        "- Regime effect stability across tickers: partially checkable from the 2025 rows; see ticker-stability table.",
        "- Regime rules usable before prediction time: the derived rule is lagged and therefore usable before the current prediction row, but it remains a proxy requiring deployment-quality validation.",
        "",
        "## Comparison Output",
        "",
        f"- Ex-ante summary CSV: `{rel(path.parent / 'vn100_exante_regime_accuracy_summary.csv')}`.",
        f"- Post-hoc versus ex-ante comparison CSV: `{rel(path.parent / 'vn100_regime_posthoc_vs_exante_comparison.csv')}`.",
        "",
        "## Claim Boundary",
        "",
        "This analysis can upgrade the regime evidence from purely post-hoc to lagged-rule diagnostic evidence where rows are reliable.",
        "It still does not establish a global 60% pass, multi-window stability, full-market representativeness, or trading readiness.",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> int:
    args = parse_args()
    data = add_exante_regimes(load_predictions(args.artifact_dir))
    exante_summary = build_exante_summary(data)
    comparison = build_comparison(args.artifact_dir, exante_summary)
    summary_path = args.report_dir / "vn100_exante_regime_accuracy_summary.csv"
    comparison_path = args.report_dir / "vn100_regime_posthoc_vs_exante_comparison.csv"
    report_path = args.report_dir / "vn100_exante_regime_validation_report.md"
    write_csv(summary_path, exante_summary)
    write_csv(comparison_path, comparison)
    write_report(report_path, args.artifact_dir, data, exante_summary, comparison)
    print(f"Wrote {rel(summary_path)}")
    print(f"Wrote {rel(comparison_path)}")
    print(f"Wrote {rel(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
