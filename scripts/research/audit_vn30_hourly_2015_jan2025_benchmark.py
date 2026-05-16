"""Audit the VN30 hourly 2015 Jan-2025 benchmark outputs."""

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


ACTIVE_UNIVERSE_SOURCE = "VN30 January 2025 review"
UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
READINESS_MANIFEST_PATH = (
    REPO_ROOT
    / "reports"
    / "generated"
    / "vn30_hourly_2015_benchmark_readiness"
    / "vn30_2015_benchmark_readiness_manifest.json"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_2015_jan2025_benchmark"
REPORT_ROOT = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015_benchmark"
AUDIT_DIR = REPORT_ROOT / "audit"
AUDIT_CSV_PATH = AUDIT_DIR / "vn30_hourly_2015_benchmark_audit.csv"
AUDIT_MD_PATH = AUDIT_DIR / "vn30_hourly_2015_benchmark_audit.md"
SUMMARY_REPORT_PATH = REPO_ROOT / "reports" / "VN30_HOURLY_2015_JAN2025_BENCHMARK_RESULT_SUMMARY.md"

TRAIN_CUTOFF_TEXT = "2024-12-31 23:59:59"
EVAL_START_TEXT = "2025-01-01 00:00:00"
REQUIRED_PREDICTION_COLUMNS = {
    "ticker",
    "model",
    "horizon",
    "actual_direction",
    "predicted_direction",
    "is_correct",
}


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def bool_text(value: bool) -> str:
    return "yes" if value else "no"


def format_pct(value: Any) -> str:
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
        values = [str(row.get(header, "")).replace("|", "\\|") for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    if max_rows is not None and len(rows) > max_rows:
        lines.append("| " + " | ".join(["..."] + ["" for _ in headers[1:]]) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def universe_tickers() -> list[str]:
    rows = read_csv_rows(UNIVERSE_PATH)
    return [str(row.get("ticker", "")).strip().upper() for row in rows if str(row.get("ticker", "")).strip()]


def expected_eval_end(readiness_manifest: dict[str, Any]) -> str:
    common = str(readiness_manifest.get("common_latest_usable_data_timestamp", "")).strip()
    return common or str(readiness_manifest.get("actual_latest_data_timestamp", "")).strip()


def add_check(rows: list[dict[str, Any]], name: str, passed: bool, details: str, *, severity: str = "fail") -> None:
    rows.append(
        {
            "check": name,
            "status": "PASS" if passed else ("WARN" if severity == "warn" else "FAIL"),
            "severity": "info" if passed else severity,
            "details": details,
        }
    )


def generated_paths(output_dir: Path) -> dict[str, Path]:
    hourly = output_dir / "hourly"
    return {
        "run_config": output_dir / "run_config.json",
        "manifest": output_dir / "manifest.json",
        "benchmark_summary": hourly / "benchmark_summary.json",
        "predictions": hourly / "predicted_vs_actual.csv",
        "accuracy": hourly / "accuracy_summary.csv",
        "classification_accuracy": hourly / "classification_accuracy_summary.csv",
        "baseline_summary": hourly / "baseline_summary.csv",
        "baseline_delta": hourly / "baseline_delta_summary.csv",
        "significance": hourly / "significance_summary.csv",
        "regime_accuracy": hourly / "regime_accuracy_summary.csv",
        "model_error": hourly / "model_error_summary.csv",
        "source_health": hourly / "source_health_summary.csv",
        "run_log": hourly / "benchmark_run_log.md",
    }


def prediction_frequency_ok(predictions: pd.DataFrame) -> bool:
    if predictions.empty or "frequency" not in predictions.columns:
        return False
    values = set(predictions["frequency"].dropna().astype(str).str.lower().unique())
    return values == {"hourly"}


def no_vn100_model_evidence(source_health: pd.DataFrame, run_config: dict[str, Any], manifest: dict[str, Any]) -> bool:
    if bool(run_config.get("vn100_evidence_reused", True)) or bool(manifest.get("vn100_evidence_reused", True)):
        return False
    if source_health.empty or "symbol" not in source_health.columns:
        return True
    used = source_health.copy()
    if "used_in_model" in used.columns:
        used = used[used["used_in_model"].astype(str).str.lower().isin(["true", "1", "yes"])]
    else:
        used = used[used.get("asset_type", "").astype(str).str.lower().eq("stock")]
    symbol_has_vn100 = used["symbol"].astype(str).str.upper().str.contains("VN100", regex=False).any()
    path_has_vn100 = (
        used["cache_path"].astype(str).str.lower().str.contains("vn100", regex=False).any()
        if "cache_path" in used.columns
        else False
    )
    return not bool(symbol_has_vn100 or path_has_vn100)


def no_daily_or_resampled_markers(
    run_config: dict[str, Any],
    manifest: dict[str, Any],
    predictions: pd.DataFrame,
    source_health: pd.DataFrame,
) -> bool:
    flags = [
        bool(run_config.get("daily_data_used", True)),
        bool(run_config.get("resampling_used", True)),
        bool(run_config.get("daily_to_hourly_resampling_used", True)),
        bool(manifest.get("daily_data_used", True)),
        bool(manifest.get("resampling_used", True)),
        bool(manifest.get("daily_to_hourly_resampling_used", True)),
    ]
    if any(flags):
        return False
    if not prediction_frequency_ok(predictions):
        return False
    if source_health.empty:
        return False
    if "frequency_1h" in source_health.columns:
        if not source_health["frequency_1h"].astype(str).str.lower().isin(["true", "1", "yes"]).all():
            return False
    return True


def predictions_by_model_horizon_ok(summary: dict[str, Any], predictions: pd.DataFrame) -> bool:
    rows = summary.get("predictions_by_model_horizon", [])
    if not isinstance(rows, list) or not rows:
        return False
    reported = sum(int(row.get("n_predictions", 0) or 0) for row in rows if isinstance(row, dict))
    return reported == int(len(predictions))


def audit_outputs(output_dir: Path = OUTPUT_DIR) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = generated_paths(output_dir)
    run_config = read_json(paths["run_config"])
    manifest = read_json(paths["manifest"])
    readiness_manifest = read_json(READINESS_MANIFEST_PATH)
    benchmark_summary = read_json(paths["benchmark_summary"])
    predictions = read_frame(paths["predictions"])
    classification_accuracy = read_frame(paths["classification_accuracy"])
    baseline_summary = read_frame(paths["baseline_summary"])
    baseline_delta = read_frame(paths["baseline_delta"])
    source_health = read_frame(paths["source_health"])
    tickers = universe_tickers()
    expected_end = expected_eval_end(readiness_manifest)

    rows: list[dict[str, Any]] = []
    add_check(rows, "output_directory_exists", output_dir.exists(), rel(output_dir))
    add_check(rows, "run_config_exists", paths["run_config"].exists(), rel(paths["run_config"]))
    add_check(rows, "manifest_exists", paths["manifest"].exists(), rel(paths["manifest"]))
    add_check(rows, "predicted_vs_actual_exists_non_empty", paths["predictions"].exists() and not predictions.empty, rel(paths["predictions"]))
    represented = set(predictions.get("ticker", pd.Series(dtype=str)).dropna().astype(str).str.upper())
    missing_tickers = sorted(set(tickers) - represented)
    add_check(rows, "all_30_active_tickers_represented", len(tickers) == 30 and not missing_tickers, f"missing={missing_tickers}")
    add_check(rows, "frequency_hourly_only", prediction_frequency_ok(predictions), "prediction frequency must be hourly")
    add_check(rows, "train_cutoff_matches", str(run_config.get("train_cutoff", "")) == TRAIN_CUTOFF_TEXT, str(run_config.get("train_cutoff", "")))
    add_check(rows, "eval_start_matches", str(run_config.get("eval_start", "")) == EVAL_START_TEXT, str(run_config.get("eval_start", "")))
    add_check(rows, "eval_end_matches_readiness", str(run_config.get("eval_end", "")) == expected_end, f"run_config={run_config.get('eval_end', '')}; expected={expected_end}")
    add_check(rows, "no_daily_or_resampled_markers", no_daily_or_resampled_markers(run_config, manifest, predictions, source_health), "daily/resampled flags must remain false and frequency must remain 1H/hourly")
    add_check(rows, "no_vn100_model_evidence_reused", no_vn100_model_evidence(source_health, run_config, manifest), "VN100 index rows, if present, must be readiness-only and not model input")
    no_old_output = (
        not bool(run_config.get("old_2005_2006_output_reused", True))
        and not bool(manifest.get("old_2005_2006_output_reused", True))
        and "2005_2026" not in rel(output_dir)
        and "2006" not in rel(output_dir)
    )
    add_check(rows, "no_old_2005_2006_output_reused", no_old_output, rel(output_dir))
    prediction_columns = set(predictions.columns)
    has_time_column = "timestamp" in prediction_columns or "datetime" in prediction_columns
    missing_columns = sorted(REQUIRED_PREDICTION_COLUMNS - prediction_columns)
    confidence_supported = "confidence" in prediction_columns
    add_check(
        rows,
        "predicted_vs_actual_required_columns",
        has_time_column and not missing_columns and confidence_supported,
        f"missing={missing_columns}; has_time_column={has_time_column}; confidence_supported={confidence_supported}",
    )
    add_check(
        rows,
        "benchmark_summary_prediction_counts",
        paths["benchmark_summary"].exists()
        and int(benchmark_summary.get("n_predictions", -1) or -1) == int(len(predictions))
        and predictions_by_model_horizon_ok(benchmark_summary, predictions),
        f"summary_n={benchmark_summary.get('n_predictions')}; actual_n={len(predictions)}",
    )
    add_check(
        rows,
        "baseline_comparison_exists",
        paths["baseline_summary"].exists() and not baseline_summary.empty and paths["baseline_delta"].exists() and not baseline_delta.empty,
        "baseline_summary and baseline_delta_summary must be non-empty",
    )
    for label in ("significance", "regime_accuracy", "model_error", "source_health", "run_log"):
        path = paths[label]
        add_check(rows, f"{label}_generated", path.exists(), rel(path), severity="warn")
    add_check(
        rows,
        "claim_boundary_generated",
        True,
        "No trading-readiness, profitability, cost/slippage, paper, or DOCX claim is made by this audit.",
        severity="warn",
    )

    audit_passed = not any(row["status"] == "FAIL" for row in rows)
    context = {
        "run_config": run_config,
        "manifest": manifest,
        "benchmark_summary": benchmark_summary,
        "predictions": predictions,
        "classification_accuracy": classification_accuracy,
        "baseline_summary": baseline_summary,
        "baseline_delta": baseline_delta,
        "source_health": source_health,
        "active_tickers": tickers,
        "expected_eval_end": expected_end,
        "audit_passed": audit_passed,
    }
    return rows, context


def write_audit_report(rows: list[dict[str, Any]], context: dict[str, Any]) -> None:
    write_csv(AUDIT_CSV_PATH, rows, ["check", "status", "severity", "details"])
    pass_count = sum(1 for row in rows if row["status"] == "PASS")
    fail_count = sum(1 for row in rows if row["status"] == "FAIL")
    warn_count = sum(1 for row in rows if row["status"] == "WARN")
    lines = [
        "# VN30 Hourly 2015 Jan-2025 Benchmark Audit",
        "",
        f"- Created at UTC: `{now_utc()}`.",
        f"- Output directory: `{rel(OUTPUT_DIR)}`.",
        f"- Audit passed: {bool_text(bool(context['audit_passed']))}.",
        f"- Checks passed: {pass_count}.",
        f"- Warnings: {warn_count}.",
        f"- Failures: {fail_count}.",
        "",
        "## Claim Boundary",
        "",
        "- No trading-readiness claim.",
        "- No profitability claim.",
        "- No cost/slippage claim.",
        "- No paper or DOCX claim.",
        "",
        "## Checks",
        "",
        markdown_table(["check", "status", "severity", "details"], rows),
        "",
    ]
    AUDIT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def aggregate_accuracy(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["model", "horizon", "n_obs", "accuracy"])
    working = predictions.copy()
    working["is_correct"] = pd.to_numeric(working["is_correct"], errors="coerce")
    working = working[working["is_correct"].isin([0, 1])].copy()
    if working.empty:
        return pd.DataFrame(columns=["model", "horizon", "n_obs", "accuracy"])
    return (
        working.groupby(["model", "horizon"], sort=True)["is_correct"]
        .agg(n_obs="count", accuracy="mean")
        .reset_index()
        .sort_values(["accuracy", "n_obs"], ascending=[False, False])
        .reset_index(drop=True)
    )


def accuracy_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        rows.append(
            {
                "model": getattr(row, "model", ""),
                "horizon": int(getattr(row, "horizon", 0)),
                "n_obs": int(getattr(row, "n_obs", 0)),
                "accuracy": format_pct(getattr(row, "accuracy", "")),
            }
        )
    return rows


def baseline_delta_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    working = frame.copy()
    working["accuracy_delta"] = pd.to_numeric(working["accuracy_delta"], errors="coerce")
    working["model_accuracy"] = pd.to_numeric(working["model_accuracy"], errors="coerce")
    working["baseline_accuracy"] = pd.to_numeric(working["baseline_accuracy"], errors="coerce")
    working = working.sort_values("accuracy_delta", ascending=False).head(12)
    rows: list[dict[str, Any]] = []
    for row in working.itertuples(index=False):
        rows.append(
            {
                "model": getattr(row, "model", ""),
                "horizon": int(getattr(row, "horizon", 0)),
                "baseline": getattr(row, "baseline", ""),
                "model_accuracy": format_pct(getattr(row, "model_accuracy", "")),
                "baseline_accuracy": format_pct(getattr(row, "baseline_accuracy", "")),
                "delta": format_pct(getattr(row, "accuracy_delta", "")),
            }
        )
    return rows


def write_result_summary(context: dict[str, Any]) -> None:
    run_config = context["run_config"]
    manifest = context["manifest"]
    benchmark_summary = context["benchmark_summary"]
    predictions = context["predictions"]
    baseline_summary = context["baseline_summary"]
    baseline_delta = context["baseline_delta"]
    tickers = context["active_tickers"]
    accuracy = aggregate_accuracy(predictions)
    best = accuracy.iloc[0].to_dict() if not accuracy.empty else {}
    models_run = sorted(predictions["model"].dropna().astype(str).unique().tolist()) if not predictions.empty else []
    baselines_run = sorted(baseline_summary["baseline"].dropna().astype(str).unique().tolist()) if not baseline_summary.empty else []
    threshold = benchmark_summary.get("threshold", run_config.get("threshold", ""))
    global_accuracy = benchmark_summary.get("overall_accuracy", "")
    passed = bool(benchmark_summary.get("passed", False))
    weak_note = (
        "The global benchmark did not meet the configured threshold."
        if threshold != "" and not passed
        else "The global benchmark met the configured threshold, but this is not a trading-readiness claim."
    )
    lines = [
        "# VN30 Hourly 2015 Jan-2025 Benchmark Result Summary",
        "",
        f"- Active universe source: {ACTIVE_UNIVERSE_SOURCE}.",
        f"- Active ticker count: {len(tickers)}.",
        f"- Active ticker list: {', '.join(tickers)}.",
        f"- Train period: {run_config.get('train_start', '2015-01-01 00:00:00')} to {run_config.get('train_cutoff', TRAIN_CUTOFF_TEXT)}.",
        f"- Evaluation period: {run_config.get('eval_start', EVAL_START_TEXT)} to {run_config.get('eval_end', '')}.",
        f"- Models run: {', '.join(models_run) if models_run else 'none'}.",
        f"- Baselines run: {', '.join(baselines_run) if baselines_run else 'none'}.",
        f"- Total predictions: {int(len(predictions))}.",
        f"- Global directional accuracy: {format_pct(global_accuracy)}.",
        f"- Global benchmark pass: {bool_text(passed) if threshold != '' else 'no threshold configured'}.",
        f"- Audit passed: {bool_text(bool(context['audit_passed']))}.",
        f"- Benchmark run: {bool_text(bool(manifest.get('benchmark_run', False)))}.",
        f"- Model training run: {bool_text(bool(manifest.get('model_training_run', False)))}, only inside benchmark workflow.",
        "- Data fetch run: no.",
        "- Daily data used: no.",
        "- Resampling used: no.",
        "- Paper/DOCX generated: no.",
        "",
        "## Accuracy By Model And Horizon",
        "",
        markdown_table(["model", "horizon", "n_obs", "accuracy"], accuracy_rows(accuracy), max_rows=20),
        "",
        "## Best Model/Horizon",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- Model: {best.get('model', '')}.",
                f"- Horizon: {int(best.get('horizon', 0))}.",
                f"- Accuracy: {format_pct(best.get('accuracy', ''))}.",
                f"- Observations: {int(best.get('n_obs', 0))}.",
                "",
            ]
        )
    else:
        lines.extend(["- No model/horizon result was available.", ""])
    lines.extend(
        [
            "## Baseline Deltas",
            "",
            markdown_table(
                ["model", "horizon", "baseline", "model_accuracy", "baseline_accuracy", "delta"],
                baseline_delta_rows(baseline_delta),
                max_rows=12,
            ),
            "",
            "## Limitations",
            "",
            f"- {weak_note}",
            "- The run is a directional classification benchmark, not a trading system.",
            "- No transaction cost, slippage, capital allocation, or execution diagnostics were run.",
            "- Index caches were inspected for readiness only and were not model feature inputs.",
            "- The evaluation uses the validated gateway cache with the Jan-2025 frozen VN30 universe only.",
            "",
            "## Claim Boundary",
            "",
            "- No trading-readiness claim.",
            "- No profitability claim.",
            "- No cost/slippage claim.",
            "- No paper or DOCX generated.",
            "",
        ]
    )
    SUMMARY_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rows, context = audit_outputs()
    write_audit_report(rows, context)
    write_result_summary(context)
    if context["audit_passed"]:
        print(f"VN30 hourly 2015 benchmark audit passed: {rel(AUDIT_MD_PATH)}")
        return 0
    print(f"VN30 hourly 2015 benchmark audit failed: {rel(AUDIT_MD_PATH)}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
