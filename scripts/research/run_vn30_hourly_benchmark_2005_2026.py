"""Run or gate the frozen VN30 hourly benchmark for the 2005-2026 NCKH design."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.vn30_hourly_common import (  # noqa: E402
    BENCHMARK_OUTPUT_DIR,
    EVAL_END_TEXT,
    EVAL_START_TEXT,
    REPORT_ROOT,
    TRAIN_CUTOFF_TEXT,
    TRAIN_START_TEXT,
    VN30_TICKERS,
    audit_hourly_coverage,
    failed_tickers_from_audit,
    load_hourly_universe_frame,
    read_universe,
    rel,
    usable_tickers_from_audit,
    write_csv,
    write_json,
    write_missing_evidence_report,
)
from scripts.run_vn100_hybrid_frequency_accuracy_benchmark import (  # noqa: E402
    ACCURACY_COLUMNS,
    BASELINE_DELTA_COLUMNS,
    BASELINE_SUMMARY_COLUMNS,
    BenchmarkConfig,
    CLASSIFICATION_ACCURACY_COLUMNS,
    MODEL_ERROR_COLUMNS,
    PREDICTED_COLUMNS,
    REGIME_ACCURACY_COLUMNS,
    SIGNIFICANCE_COLUMNS,
    build_baseline_delta_summary,
    build_classification_accuracy_summary,
    build_regime_accuracy_summary,
    build_significance_summary,
    run_frequency_benchmark,
)


DEFAULT_REPORT_PATH = REPORT_ROOT / "vn30_hourly_benchmark_missing_evidence.md"
DEFAULT_MODELS = ["xgboost", "lightgbm", "random_forest", "stacking"]
DEFAULT_HORIZONS = [1, 4, 8, 20]
SOURCE_HEALTH_COLUMNS = [
    "ticker",
    "raw_hourly_sources",
    "raw_hourly_files",
    "first_available_hourly_timestamp",
    "last_available_hourly_timestamp",
    "hourly_rows",
    "benchmark_usable",
    "missing_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the official frozen VN30 hourly benchmark gate.")
    parser.add_argument("--output-dir", type=Path, default=BENCHMARK_OUTPUT_DIR)
    parser.add_argument("--missing-report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--allow-missing-exit-zero", action="store_true")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--horizons", default=",".join(str(item) for item in DEFAULT_HORIZONS))
    return parser.parse_args()


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in parse_csv_list(value)]


def run_config_payload(output_dir: Path, *, models: list[str], horizons: list[int], status: str) -> dict[str, Any]:
    return {
        "universe": "VN30",
        "tickers": VN30_TICKERS,
        "frequency": "hourly",
        "train_start": TRAIN_START_TEXT,
        "train_cutoff": TRAIN_CUTOFF_TEXT,
        "eval_start": EVAL_START_TEXT,
        "eval_end": EVAL_END_TEXT,
        "target_mode": "classification",
        "training_label_cutoff_rule": "target_timestamp <= train_cutoff",
        "models": models,
        "horizons": horizons,
        "threshold": 0.60,
        "output_dir": rel(output_dir),
        "daily_data_used": False,
        "daily_to_hourly_resampling_used": False,
        "vn100_evidence_reused": False,
        "status": status,
    }


def manifest_payload(audit_rows: list[dict[str, Any]], *, status: str) -> dict[str, Any]:
    usable = usable_tickers_from_audit(audit_rows)
    return {
        "universe": "VN30",
        "frequency": "hourly",
        "frozen_ticker_count": len(VN30_TICKERS),
        "benchmark_usable_ticker_count": len(usable),
        "benchmark_usable_tickers": usable,
        "failed_ticker_count": len(failed_tickers_from_audit(audit_rows)),
        "train_start": TRAIN_START_TEXT,
        "train_cutoff": TRAIN_CUTOFF_TEXT,
        "eval_start": EVAL_START_TEXT,
        "eval_end": EVAL_END_TEXT,
        "target_mode": "classification",
        "training_label_cutoff_rule": "target_timestamp <= train_cutoff",
        "all_30_benchmark_usable": len(usable) == 30,
        "daily_data_used": False,
        "daily_to_hourly_resampling_used": False,
        "vn100_seven_ticker_evidence_reused": False,
        "status": status,
    }


def source_health_rows(audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{column: row.get(column, "") for column in SOURCE_HEALTH_COLUMNS} for row in audit_rows]


def empty_benchmark_summary(audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = usable_tickers_from_audit(audit_rows)
    return {
        "universe": "VN30",
        "frequency": "hourly",
        "benchmark_run": False,
        "status": "stopped_missing_evidence",
        "n_predictions": 0,
        "overall_accuracy": None,
        "passed": False,
        "evaluated_tickers": [],
        "benchmark_usable_ticker_count": len(usable),
        "benchmark_usable_tickers": usable,
        "failed_ticker_count": 30 - len(usable),
        "train_start": TRAIN_START_TEXT,
        "train_cutoff": TRAIN_CUTOFF_TEXT,
        "eval_start": EVAL_START_TEXT,
        "eval_end": EVAL_END_TEXT,
        "target_mode": "classification",
        "training_label_cutoff_rule": "target_timestamp <= train_cutoff",
        "missing_evidence_report": rel(DEFAULT_REPORT_PATH),
    }


def write_empty_hourly_outputs(output_dir: Path, audit_rows: list[dict[str, Any]]) -> None:
    hourly_dir = output_dir / "hourly"
    hourly_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=PREDICTED_COLUMNS).to_csv(hourly_dir / "predicted_vs_actual.csv", index=False)
    pd.DataFrame(columns=ACCURACY_COLUMNS).to_csv(hourly_dir / "accuracy_summary.csv", index=False)
    pd.DataFrame(columns=CLASSIFICATION_ACCURACY_COLUMNS).to_csv(
        hourly_dir / "classification_accuracy_summary.csv",
        index=False,
    )
    pd.DataFrame(columns=BASELINE_SUMMARY_COLUMNS).to_csv(hourly_dir / "baseline_summary.csv", index=False)
    pd.DataFrame(columns=BASELINE_DELTA_COLUMNS).to_csv(hourly_dir / "baseline_delta_summary.csv", index=False)
    pd.DataFrame(columns=SIGNIFICANCE_COLUMNS).to_csv(hourly_dir / "significance_summary.csv", index=False)
    pd.DataFrame(columns=REGIME_ACCURACY_COLUMNS).to_csv(hourly_dir / "regime_accuracy_summary.csv", index=False)
    pd.DataFrame(columns=MODEL_ERROR_COLUMNS).to_csv(hourly_dir / "model_error_summary.csv", index=False)
    write_csv(hourly_dir / "source_health_summary.csv", source_health_rows(audit_rows), fieldnames=SOURCE_HEALTH_COLUMNS)
    write_json(hourly_dir / "benchmark_summary.json", empty_benchmark_summary(audit_rows))


def build_config(output_dir: Path, *, models: list[str], horizons: list[int]) -> BenchmarkConfig:
    return BenchmarkConfig(
        universe="VN30",
        daily_start=TRAIN_START_TEXT,
        daily_end=TRAIN_CUTOFF_TEXT,
        hourly_start=TRAIN_START_TEXT,
        hourly_end=EVAL_END_TEXT,
        train_cutoff=TRAIN_CUTOFF_TEXT,
        eval_start=EVAL_START_TEXT,
        eval_end=EVAL_END_TEXT,
        models=models,
        daily_horizons=[],
        hourly_horizons=horizons,
        threshold=0.60,
        provider="local_hourly_csv",
        pull_missing=False,
        cache_dir="",
        output_dir=str(output_dir),
        cache_only=True,
        fetch_only=False,
        resume_fetch=False,
        provider_calls_allowed=False,
        checkpointing_enabled=False,
        max_tickers=None,
        min_history_days=None,
        min_obs_per_group=50,
        max_daily_gap_days=30,
        rate_limit_per_minute=0,
        request_sleep_seconds=None,
        max_fetch_retries=0,
        fetch_batch_size=0,
        fetch_batch_cooldown_seconds=0.0,
        source_failure_threshold=0,
        source_empty_threshold=0,
        source_cooldown_seconds=0.0,
        min_provider_daily_rows=0,
        min_provider_hourly_rows=0,
        coverage_start_tolerance_days=0,
        coverage_end_tolerance_days=0,
        min_coverage_ratio=1.0,
        allow_partial_cache_for_benchmark=False,
        min_pre_eval_rows_daily=0,
        min_pre_eval_rows_hourly=1,
        min_eval_rows_daily=0,
        min_eval_rows_hourly=1,
        bootstrap_samples=1000,
        bootstrap_seed=42,
        enable_regime_evaluation=True,
        regime_return_window=20,
        regime_vol_window=20,
        regime_bull_threshold=0.03,
        regime_bear_threshold=-0.03,
        regime_vol_quantile=0.70,
        target_mode="classification",
        enable_confidence_filter=False,
        confidence_threshold=0.55,
        enable_confidence_threshold_sweep=False,
        confidence_threshold_grid=[0.50],
        min_sweep_coverage=0.30,
        no_trade_band=0.0,
        min_coverage_after_filter=0.30,
        enable_horizon_tuning=False,
        tuning_models=[],
        tuning_trials=0,
        tuning_seed=42,
        tuning_metric="directional_accuracy",
        tuning_time_budget_seconds=None,
        tuning_output_dir=None,
        provider_timeout_seconds=0.0,
        model_fit_timeout_seconds=300.0,
        retrain_frequency="monthly",
        seed=42,
    )


def write_success_outputs(
    output_dir: Path,
    config: BenchmarkConfig,
    audit_rows: list[dict[str, Any]],
    outputs: tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    hourly_dir = output_dir / "hourly"
    hourly_dir.mkdir(parents=True, exist_ok=True)
    predictions, accuracy, summary, baseline, model_errors, _baseline_predictions, _tuning = outputs
    baseline_delta = build_baseline_delta_summary(accuracy, baseline)
    significance = build_significance_summary(
        predictions,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
    )
    regime_accuracy = build_regime_accuracy_summary(predictions, config.threshold, config.min_obs_per_group)
    classification_accuracy = build_classification_accuracy_summary(
        predictions,
        threshold=config.threshold,
        min_obs_per_group=config.min_obs_per_group,
    )
    summary.update(
        {
            "benchmark_run": True,
            "status": "completed",
            "all_30_benchmark_usable": True,
            "daily_data_used": False,
            "daily_to_hourly_resampling_used": False,
            "vn100_seven_ticker_evidence_reused": False,
        }
    )
    predictions.to_csv(hourly_dir / "predicted_vs_actual.csv", index=False)
    accuracy.to_csv(hourly_dir / "accuracy_summary.csv", index=False)
    classification_accuracy.to_csv(hourly_dir / "classification_accuracy_summary.csv", index=False)
    baseline.to_csv(hourly_dir / "baseline_summary.csv", index=False)
    baseline_delta.to_csv(hourly_dir / "baseline_delta_summary.csv", index=False)
    significance.to_csv(hourly_dir / "significance_summary.csv", index=False)
    regime_accuracy.to_csv(hourly_dir / "regime_accuracy_summary.csv", index=False)
    model_errors.to_csv(hourly_dir / "model_error_summary.csv", index=False)
    write_csv(hourly_dir / "source_health_summary.csv", source_health_rows(audit_rows), fieldnames=SOURCE_HEALTH_COLUMNS)
    write_json(hourly_dir / "benchmark_summary.json", summary)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    models = parse_csv_list(args.models)
    horizons = parse_int_list(args.horizons)
    tickers = read_universe()
    if tickers != VN30_TICKERS:
        raise ValueError("Frozen VN30 ticker list does not match the mandatory list.")

    audit_rows = audit_hourly_coverage()
    usable = usable_tickers_from_audit(audit_rows)
    status = "ready" if len(usable) == 30 else "stopped_missing_evidence"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", run_config_payload(output_dir, models=models, horizons=horizons, status=status))
    write_json(output_dir / "manifest.json", manifest_payload(audit_rows, status=status))

    if len(usable) < 30:
        write_empty_hourly_outputs(output_dir, audit_rows)
        write_missing_evidence_report(args.missing_report, audit_rows, source=Path(__file__).name)
        print(
            "VN30 hourly benchmark stopped before training/evaluation: "
            f"usable={len(usable)}/30 missing_report={rel(args.missing_report)} output_dir={rel(output_dir)}"
        )
        if not args.allow_missing_exit_zero:
            return 2
        return 0

    hourly_raw = load_hourly_universe_frame(tickers)
    config = build_config(output_dir, models=models, horizons=horizons)
    outputs = run_frequency_benchmark(
        raw_df=hourly_raw,
        frequency="hourly",
        horizons=horizons,
        models=models,
        initial_train_start=TRAIN_START_TEXT,
        initial_train_end=TRAIN_CUTOFF_TEXT,
        eval_start=EVAL_START_TEXT,
        eval_end=EVAL_END_TEXT,
        threshold=0.60,
        provider="local_hourly_csv",
        universe="VN30",
        retrain_frequency=config.retrain_frequency,
        seed=config.seed,
        min_history_days=config.min_history_days,
        min_obs_per_group=config.min_obs_per_group,
        max_daily_gap_days=config.max_daily_gap_days,
        config=config,
    )
    write_success_outputs(output_dir, config, audit_rows, outputs)
    summary = outputs[2]
    print(
        "VN30 hourly benchmark complete: "
        f"accuracy={summary.get('overall_accuracy')} n={summary.get('n_predictions')} output_dir={rel(output_dir)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
