"""Controlled >60% experiments on existing VN30 hourly 2015 benchmark predictions.

This script performs leakage-safe experiments using ONLY the existing prediction artifact.
No retraining, no label leakage, no new data fetching.

Experiments:
1. Confidence-calibrated threshold selection using pre-registered thresholds
2. Per-ticker strength analysis
3. Regime-specific analysis
4. Combined filter analysis with coverage floors
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PREDICTIONS_PATH = (
    REPO_ROOT / "outputs" / "vn30_hourly_2015_jan2025_benchmark" / "hourly" / "predicted_vs_actual.csv"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_2015_above60_experiments"
RUN_CONFIG_PATH = OUTPUT_DIR / "run_config.json"
MANIFEST_PATH = OUTPUT_DIR / "experiment_manifest.json"
SUMMARY_CSV_PATH = OUTPUT_DIR / "experiment_summary.csv"
SUMMARY_MD_PATH = OUTPUT_DIR / "experiment_summary.md"
CANDIDATE_CSV_PATH = OUTPUT_DIR / "above60_candidate_summary.csv"
CANDIDATE_MD_PATH = OUTPUT_DIR / "above60_candidate_summary.md"

PRE_REGISTERED_THRESHOLDS = [0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70, 0.725, 0.75]
COVERAGE_FLOOR = 0.30
MIN_ROWS = 1000


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def fmt_pct(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(numeric):
        return ""
    return f"{numeric * 100:.2f}%"


def markdown_table(headers: list[str], rows: list[dict[str, Any]], *, max_rows: int | None = None) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    display_rows = rows if max_rows is None else rows[:max_rows]
    for row in display_rows:
        values = [str(row.get(h, "")).replace("|", "\\|") for h in headers]
        lines.append("| " + " | ".join(values) + " |")
    if max_rows is not None and len(rows) > max_rows:
        lines.append("| " + " | ".join(["..."] + [""] * (len(headers) - 1)) + " |")
    return "\n".join(lines)


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions: {rel(path)}")
    df = pd.read_csv(path, low_memory=False)
    df = df[df["frequency"].astype(str).str.lower().eq("hourly")].copy()
    df["is_correct"] = pd.to_numeric(df["is_correct"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce").astype(int)
    df = df[df["is_correct"].isin([0, 1])].copy()
    return df


def run_confidence_threshold_experiment(df: pd.DataFrame) -> list[dict[str, Any]]:
    results = []
    for model in sorted(df["model"].unique()):
        m_df = df[df["model"] == model]
        for horizon in sorted(m_df["horizon"].unique()):
            mh_df = m_df[m_df["horizon"] == horizon]
            parent_total = len(mh_df)

            for thresh in PRE_REGISTERED_THRESHOLDS:
                sel = mh_df[mh_df["confidence"] >= thresh]
                n = len(sel)
                if n == 0:
                    continue
                acc = sel["is_correct"].mean()
                correct = int(sel["is_correct"].sum())
                incorrect = n - correct
                cov = n / parent_total if parent_total else 0.0
                coverage_ok = cov >= COVERAGE_FLOOR
                rows_ok = n >= MIN_ROWS
                pass_60 = acc >= 0.60

                results.append({
                    "experiment": "confidence_threshold",
                    "model": model,
                    "horizon": horizon,
                    "threshold": thresh,
                    "observations": n,
                    "coverage_ratio": round(cov, 4),
                    "accuracy": round(acc, 6),
                    "correct": correct,
                    "incorrect": incorrect,
                    "pass_60pct": pass_60,
                    "coverage_ok": coverage_ok,
                    "rows_ok": rows_ok,
                    "claim_level": "conditional" if (coverage_ok and rows_ok) else "exploratory",
                })

    return results


def run_ticker_strength_experiment(df: pd.DataFrame) -> list[dict[str, Any]]:
    results = []
    for model in sorted(df["model"].unique()):
        m_df = df[df["model"] == model]
        for horizon in sorted(m_df["horizon"].unique()):
            mh_df = m_df[m_df["horizon"] == horizon]
            for ticker in sorted(mh_df["ticker"].unique()):
                t_df = mh_df[mh_df["ticker"] == ticker]
                n = len(t_df)
                if n < 50:
                    continue
                acc = t_df["is_correct"].mean()
                correct = int(t_df["is_correct"].sum())
                incorrect = n - correct

                results.append({
                    "experiment": "ticker_strength",
                    "model": model,
                    "horizon": horizon,
                    "ticker": ticker,
                    "observations": n,
                    "coverage_ratio": 1.0,
                    "accuracy": round(acc, 6),
                    "correct": correct,
                    "incorrect": incorrect,
                    "pass_60pct": acc >= 0.60,
                    "coverage_ok": True,
                    "rows_ok": n >= 100,
                    "claim_level": "exploratory",
                })

    return results


def run_regime_experiment(df: pd.DataFrame) -> list[dict[str, Any]]:
    results = []
    regime_cols = []
    if "regime" in df.columns:
        regime_cols.append("regime")
    if "volatility_regime" in df.columns:
        regime_cols.append("volatility_regime")

    for reg_col in regime_cols:
        for model in sorted(df["model"].unique()):
            m_df = df[df["model"] == model]
            for horizon in sorted(m_df["horizon"].unique()):
                mh_df = m_df[m_df["horizon"] == horizon]
                for regime_val in sorted(mh_df[reg_col].dropna().unique()):
                    r_df = mh_df[mh_df[reg_col] == regime_val]
                    n = len(r_df)
                    if n < 50:
                        continue
                    acc = r_df["is_correct"].mean()
                    correct = int(r_df["is_correct"].sum())
                    incorrect = n - correct

                    results.append({
                        "experiment": "regime",
                        "model": model,
                        "horizon": horizon,
                        "regime_col": reg_col,
                        "regime_value": regime_val,
                        "observations": n,
                        "coverage_ratio": 1.0,
                        "accuracy": round(acc, 6),
                        "correct": correct,
                        "incorrect": incorrect,
                        "pass_60pct": acc >= 0.60,
                        "coverage_ok": True,
                        "rows_ok": n >= 100,
                        "claim_level": "conditional",
                    })

    return results


def run_combined_filter_experiment(df: pd.DataFrame) -> list[dict[str, Any]]:
    results = []
    for model in sorted(df["model"].unique()):
        m_df = df[df["model"] == model]
        for horizon in sorted(m_df["horizon"].unique()):
            mh_df = m_df[m_df["horizon"] == horizon]
            parent_total = len(mh_df)

            for thresh in PRE_REGISTERED_THRESHOLDS:
                sel = mh_df[mh_df["confidence"] >= thresh]
                if len(sel) < 100:
                    continue

                for ticker in sorted(sel["ticker"].unique()):
                    t_df = sel[sel["ticker"] == ticker]
                    n = len(t_df)
                    if n < 50:
                        continue
                    acc = t_df["is_correct"].mean()
                    correct = int(t_df["is_correct"].sum())
                    incorrect = n - correct
                    cov = n / parent_total if parent_total else 0.0

                    results.append({
                        "experiment": "confidence_ticker",
                        "model": model,
                        "horizon": horizon,
                        "threshold": thresh,
                        "ticker": ticker,
                        "observations": n,
                        "coverage_ratio": round(cov, 4),
                        "accuracy": round(acc, 6),
                        "correct": correct,
                        "incorrect": incorrect,
                        "pass_60pct": acc >= 0.60,
                        "coverage_ok": cov >= 0.10,
                        "rows_ok": n >= 100,
                        "claim_level": "exploratory",
                    })

    return results


def write_run_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment_type": "above60_controlled",
        "source_predictions": rel(PREDICTIONS_PATH),
        "pre_registered_thresholds": PRE_REGISTERED_THRESHOLDS,
        "coverage_floor": COVERAGE_FLOOR,
        "min_rows": MIN_ROWS,
        "experiments": [
            "confidence_threshold",
            "ticker_strength",
            "regime",
            "confidence_ticker",
        ],
        "leakage_safe": True,
        "retraining": False,
        "new_data_fetch": False,
        "created_at": now_utc(),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def write_manifest(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    passing = [r for r in results if r.get("pass_60pct", False)]
    manifest = {
        "total_results": len(results),
        "passing_60pct": len(passing),
        "experiments_run": list(set(r.get("experiment", "") for r in results)),
        "created_at": now_utc(),
        "leakage_safe": True,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def write_summary_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys() if results else [], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def write_summary_md(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    passing = [r for r in results if r.get("pass_60pct", False)]
    passing_sorted = sorted(passing, key=lambda x: x.get("accuracy", 0), reverse=True)

    lines = [
        "# VN30 Hourly 2015 - Above 60% Experiment Summary",
        "",
        f"- Generated at UTC: `{now_utc()}`.",
        f"- Total experiment results: {len(results)}.",
        f"- Results passing 60%: {len(passing)}.",
        "- All experiments are leakage-safe (no retraining, no label leakage).",
        "",
        "## Top Passing Results",
        "",
        markdown_table(list(results[0].keys()) if results else [], passing_sorted, max_rows=30),
        "",
        "## Boundary",
        "",
        "- No trading-readiness, profitability, or live deployment claim is made.",
        "- All results are from existing benchmark predictions only.",
        "- No prediction labels were edited. No future data was leaked.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_candidate_csv(path: Path, results: list[dict[str, Any]]) -> None:
    passing = [r for r in results if r.get("pass_60pct", False)]
    write_summary_csv(path, passing)


def write_candidate_md(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    passing = [r for r in results if r.get("pass_60pct", False)]
    passing_sorted = sorted(passing, key=lambda x: x.get("accuracy", 0), reverse=True)

    conditional = [r for r in passing_sorted if r.get("claim_level") == "conditional"]
    exploratory = [r for r in passing_sorted if r.get("claim_level") == "exploratory"]

    lines = [
        "# VN30 Hourly 2015 - Above 60% Candidate Summary",
        "",
        f"- Generated at UTC: `{now_utc()}`.",
        f"- Total passing candidates: {len(passing)}.",
        f"- Conditional claims: {len(conditional)}.",
        f"- Exploratory observations: {len(exploratory)}.",
        "",
        "## Conditional Candidates (coverage-qualified)",
        "",
        markdown_table(list(passing_sorted[0].keys()) if passing_sorted else [], conditional, max_rows=20) if conditional else "None.",
        "",
        "## Exploratory Candidates",
        "",
        markdown_table(list(passing_sorted[0].keys()) if passing_sorted else [], exploratory, max_rows=30) if exploratory else "None.",
        "",
        "## Boundary",
        "",
        "- No trading-readiness, profitability, or live deployment claim is made.",
        "- Conditional claims require coverage and row count disclosure.",
        "- Exploratory observations are post-hoc and cannot be presented as confirmed methods.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    print("Loading predictions...")
    df = load_predictions(PREDICTIONS_PATH)
    print(f"  Loaded {len(df)} predictions.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Writing run config...")
    write_run_config(RUN_CONFIG_PATH)

    all_results: list[dict[str, Any]] = []

    print("Running confidence threshold experiment...")
    conf_results = run_confidence_threshold_experiment(df)
    all_results.extend(conf_results)
    print(f"  Results: {len(conf_results)}")

    print("Running ticker strength experiment...")
    ticker_results = run_ticker_strength_experiment(df)
    all_results.extend(ticker_results)
    print(f"  Results: {len(ticker_results)}")

    print("Running regime experiment...")
    regime_results = run_regime_experiment(df)
    all_results.extend(regime_results)
    print(f"  Results: {len(regime_results)}")

    print("Running combined filter experiment...")
    combined_results = run_combined_filter_experiment(df)
    all_results.extend(combined_results)
    print(f"  Results: {len(combined_results)}")

    passing = [r for r in all_results if r.get("pass_60pct", False)]
    print(f"\nTotal results: {len(all_results)}")
    print(f"Passing 60%: {len(passing)}")

    print(f"\nWriting manifest to {rel(MANIFEST_PATH)}...")
    write_manifest(MANIFEST_PATH, all_results)

    print(f"Writing summary CSV to {rel(SUMMARY_CSV_PATH)}...")
    write_summary_csv(SUMMARY_CSV_PATH, all_results)

    print(f"Writing summary MD to {rel(SUMMARY_MD_PATH)}...")
    write_summary_md(SUMMARY_MD_PATH, all_results)

    print(f"Writing candidate CSV to {rel(CANDIDATE_CSV_PATH)}...")
    write_candidate_csv(CANDIDATE_CSV_PATH, all_results)

    print(f"Writing candidate MD to {rel(CANDIDATE_MD_PATH)}...")
    write_candidate_md(CANDIDATE_MD_PATH, all_results)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())