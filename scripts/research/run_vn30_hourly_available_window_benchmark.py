"""Run the VN30 hourly available-window benchmark from the design decision."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.vn30_hourly_available_window_common import (  # noqa: E402
    BENCHMARK_OUTPUT_DIR,
    DEFAULT_HORIZONS,
    DEFAULT_MODELS,
    DESIGN_DECISION_JSON,
    REPORT_ROOT,
    TARGET_MODE,
    final_paper_can_proceed,
    load_design_decision,
    load_selected_hourly_frame,
    rel,
    selected_tickers,
    write_csv,
    write_json,
)
from scripts.run_vn100_hybrid_frequency_accuracy_benchmark import (  # noqa: E402
    ACCURACY_COLUMNS,
    BASELINE_DELTA_COLUMNS,
    BASELINE_SUMMARY_COLUMNS,
    CLASSIFICATION_ACCURACY_COLUMNS,
    MODEL_ERROR_COLUMNS,
    PREDICTED_COLUMNS,
    REGIME_ACCURACY_COLUMNS,
    SIGNIFICANCE_COLUMNS,
    BenchmarkConfig,
    build_baseline_delta_summary,
    build_classification_accuracy_summary,
    build_regime_accuracy_summary,
    build_significance_summary,
    run_frequency_benchmark,
)


NOT_RUN_REPORT = REPORT_ROOT / "vn30_available_window_benchmark_not_run.md"
SOURCE_HEALTH_COLUMNS = [
    "ticker",
    "first_available_hourly_timestamp",
    "last_available_hourly_timestamp",
    "rows_used",
    "selected_for_available_window",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VN30 hourly available-window benchmark.")
    parser.add_argument("--design-json", type=Path, default=DESIGN_DECISION_JSON)
    parser.add_argument("--output-dir", type=Path, default=BENCHMARK_OUTPUT_DIR)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--horizons", default=",".join(str(item) for item in DEFAULT_HORIZONS))
    return parser.parse_args()


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in parse_csv_list(value)]


def run_config_payload(output_dir: Path, decision: dict[str, Any], models: list[str], horizons: list[int], status: str) -> dict[str, Any]:
    return {
        "universe": "VN30",
        "study": "VN30 hourly available-window",
        "selected_tickers": selected_tickers(decision),
        "excluded_tickers": decision.get("excluded_tickers", []),
        "frequency": "hourly",
        "train_start": decision.get("training_start", ""),
        "train_cutoff": decision.get("training_cutoff", ""),
        "eval_start": decision.get("evaluation_start", ""),
        "eval_end": decision.get("evaluation_end", ""),
        "target_mode": TARGET_MODE,
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


def manifest_payload(decision: dict[str, Any], status: str) -> dict[str, Any]:
    tickers = selected_tickers(decision)
    return {
        "universe": "VN30",
        "study": "VN30 hourly available-window",
        "frequency": "hourly",
        "frozen_ticker_count": 30,
        "selected_ticker_count": len(tickers),
        "selected_tickers": tickers,
        "excluded_ticker_count": 30 - len(tickers),
        "excluded_tickers": decision.get("excluded_tickers", []),
        "train_start": decision.get("training_start", ""),
        "train_cutoff": decision.get("training_cutoff", ""),
        "eval_start": decision.get("evaluation_start", ""),
        "eval_end": decision.get("evaluation_end", ""),
        "target_mode": TARGET_MODE,
        "training_label_cutoff_rule": "target_timestamp <= train_cutoff",
        "full_vn30_representativeness": bool(decision.get("full_vn30_representativeness")),
        "final_paper_can_proceed": bool(decision.get("final_paper_can_proceed")),
        "daily_data_used": False,
        "daily_to_hourly_resampling_used": False,
        "vn100_evidence_reused": False,
        "status": status,
    }


def build_config(output_dir: Path, decision: dict[str, Any], models: list[str], horizons: list[int]) -> BenchmarkConfig:
    return BenchmarkConfig(
        universe="VN30_AVAILABLE_WINDOW",
        daily_start=decision["training_start"],
        daily_end=decision["training_cutoff"],
        hourly_start=decision["training_start"],
        hourly_end=decision["evaluation_end"],
        train_cutoff=decision["training_cutoff"],
        eval_start=decision["evaluation_start"],
        eval_end=decision["evaluation_end"],
        models=models,
        daily_horizons=[],
        hourly_horizons=horizons,
        threshold=0.60,
        provider="local_hourly_available_window",
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
        allow_partial_cache_for_benchmark=True,
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
        target_mode=TARGET_MODE,
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


def source_health_rows(raw: pd.DataFrame, decision: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ticker in selected_tickers(decision):
        group = raw[raw["ticker"].astype(str).str.upper().eq(ticker)].copy() if not raw.empty else pd.DataFrame()
        timestamps = pd.to_datetime(group["datetime"], errors="coerce").dropna() if not group.empty else pd.Series(dtype="datetime64[ns]")
        rows.append(
            {
                "ticker": ticker,
                "first_available_hourly_timestamp": timestamps.min().strftime("%Y-%m-%d %H:%M:%S") if not timestamps.empty else "",
                "last_available_hourly_timestamp": timestamps.max().strftime("%Y-%m-%d %H:%M:%S") if not timestamps.empty else "",
                "rows_used": int(len(group)),
                "selected_for_available_window": True,
            }
        )
    return rows


def write_not_run_report(path: Path, decision: dict[str, Any]) -> None:
    content = [
        "# VN30 Available-Window Benchmark Not Run",
        "",
        "The available-window benchmark was not run because the design gate did not allow a final paper.",
        "",
        f"- Selected ticker count: {len(selected_tickers(decision))}.",
        f"- Final paper can proceed: {str(final_paper_can_proceed(decision)).lower()}.",
        f"- Claim boundary: {decision.get('claim_boundary', '')}.",
        "",
        "No fake benchmark outputs were generated.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def write_outputs(
    output_dir: Path,
    config: BenchmarkConfig,
    decision: dict[str, Any],
    raw: pd.DataFrame,
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
            "study": "VN30 hourly available-window",
            "selected_ticker_count": len(selected_tickers(decision)),
            "selected_tickers": selected_tickers(decision),
            "excluded_tickers": decision.get("excluded_tickers", []),
            "full_vn30_representativeness": bool(decision.get("full_vn30_representativeness")),
            "daily_data_used": False,
            "daily_to_hourly_resampling_used": False,
            "vn100_evidence_reused": False,
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
    write_csv(hourly_dir / "source_health_summary.csv", source_health_rows(raw, decision), fieldnames=SOURCE_HEALTH_COLUMNS)
    write_json(hourly_dir / "benchmark_summary.json", summary)


def main() -> int:
    args = parse_args()
    decision = load_design_decision(args.design_json)
    models = parse_csv_list(args.models)
    horizons = parse_int_list(args.horizons)
    if not final_paper_can_proceed(decision):
        write_not_run_report(NOT_RUN_REPORT, decision)
        print(f"VN30 available-window benchmark not run: report={rel(NOT_RUN_REPORT)}")
        return 0

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", run_config_payload(output_dir, decision, models, horizons, "running"))
    write_json(output_dir / "manifest.json", manifest_payload(decision, "running"))
    raw = load_selected_hourly_frame(decision)
    config = build_config(output_dir, decision, models, horizons)
    outputs = run_frequency_benchmark(
        raw_df=raw,
        frequency="hourly",
        horizons=horizons,
        models=models,
        initial_train_start=decision["training_start"],
        initial_train_end=decision["training_cutoff"],
        eval_start=decision["evaluation_start"],
        eval_end=decision["evaluation_end"],
        threshold=0.60,
        provider="local_hourly_available_window",
        universe="VN30_AVAILABLE_WINDOW",
        retrain_frequency=config.retrain_frequency,
        seed=config.seed,
        min_history_days=config.min_history_days,
        min_obs_per_group=config.min_obs_per_group,
        max_daily_gap_days=config.max_daily_gap_days,
        config=config,
    )
    write_outputs(output_dir, config, decision, raw, outputs)
    write_json(output_dir / "run_config.json", run_config_payload(output_dir, decision, models, horizons, "completed"))
    write_json(output_dir / "manifest.json", manifest_payload(decision, "completed"))
    summary = outputs[2]
    print(
        "VN30 hourly available-window benchmark complete: "
        f"predictions={len(outputs[0])} accuracy={summary.get('overall_accuracy')} output_dir={rel(output_dir)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
