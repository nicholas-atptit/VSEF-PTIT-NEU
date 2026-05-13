"""VN100 hybrid-frequency walk-forward directional-accuracy benchmark.

The benchmark keeps daily and hourly market data in separate raw caches. Daily
evaluation combines true daily 2006-2015 OHLCV with hourly 2016-2024 OHLCV
resampled to daily bars. Hourly evaluation uses the hourly cache directly.

Official VN100 benchmark design:
- historical/training label cutoff: 2024-12-31
- daily data range: 2006-01-01 to 2015-12-31
- hourly raw/actual data may extend to 2025-12-31 for held-out labels
- out-of-sample evaluation range: 2025-01-01 to 2025-12-31

The 2025 window is held out for walk-forward out-of-sample evaluation only.
Models must not train on 2025 data before predicting 2025 targets. Later data,
such as 2026-05-11 snapshots, belongs to explicitly labeled extended
monitoring runs and is not part of the official benchmark period.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.adapters.vnstock_adapter import VnstockAdapter
from src.data.universe import VN100_BACKUP_TICKERS
from src.utils.logging import get_logger


logger = get_logger(__name__)


PREDICTED_COLUMNS = [
    "timestamp",
    "date",
    "ticker",
    "frequency",
    "horizon",
    "model",
    "target_mode",
    "actual_close",
    "predicted_close",
    "actual_return",
    "predicted_return",
    "actual_direction",
    "predicted_direction",
    "confidence",
    "filtered_out",
    "regime",
    "volatility_regime",
    "is_correct",
]

ACCURACY_COLUMNS = [
    "frequency",
    "model",
    "horizon",
    "ticker",
    "n_obs",
    "accuracy",
    "reliable",
    "passed_60pct",
]

BASELINE_SUMMARY_COLUMNS = [
    "frequency",
    "baseline",
    "horizon",
    "ticker",
    "n_obs",
    "accuracy",
    "reliable",
    "passed_60pct",
]

BASELINE_DELTA_COLUMNS = [
    "frequency",
    "model",
    "horizon",
    "baseline",
    "model_accuracy",
    "baseline_accuracy",
    "accuracy_delta",
    "model_n_obs",
    "baseline_n_obs",
    "model_better_than_baseline",
]

SIGNIFICANCE_COLUMNS = [
    "frequency",
    "model",
    "horizon",
    "n_obs",
    "accuracy",
    "null_accuracy",
    "binomial_p_value",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "significant_at_5pct",
    "significant_at_10pct",
]

MCNEMAR_SUMMARY_COLUMNS = [
    "frequency",
    "model",
    "horizon",
    "baseline",
    "matched_n_obs",
    "model_correct_only",
    "baseline_correct_only",
    "both_correct",
    "both_wrong",
    "mcnemar_p_value",
]

BASELINE_PREDICTION_COLUMNS = [
    "timestamp",
    "date",
    "ticker",
    "frequency",
    "horizon",
    "baseline",
    "actual_direction",
    "predicted_direction",
    "is_correct",
    "regime",
    "volatility_regime",
]

REGIME_ACCURACY_COLUMNS = [
    "frequency",
    "regime",
    "model",
    "horizon",
    "n_obs",
    "accuracy",
    "passed_60pct",
    "reliable",
]

REGIME_BASELINE_DELTA_COLUMNS = [
    "frequency",
    "regime",
    "model",
    "horizon",
    "baseline",
    "model_accuracy",
    "baseline_accuracy",
    "accuracy_delta",
    "model_better_than_baseline",
]

CLASSIFICATION_ACCURACY_COLUMNS = [
    "frequency",
    "model",
    "horizon",
    "target_mode",
    "n_obs",
    "accuracy",
    "passed_60pct",
    "reliable",
]

CONFIDENCE_FILTER_COLUMNS = [
    "frequency",
    "model",
    "horizon",
    "confidence_threshold",
    "total_rows",
    "evaluated_rows",
    "coverage_ratio",
    "unfiltered_accuracy",
    "filtered_accuracy",
    "filtered_passed_60pct",
    "min_coverage_after_filter",
    "coverage_ok",
]

CONFIDENCE_THRESHOLD_SWEEP_COLUMNS = [
    "frequency",
    "model",
    "horizon",
    "threshold",
    "total_rows",
    "evaluated_rows",
    "coverage_ratio",
    "filtered_accuracy",
    "passed_60pct",
    "coverage_ok",
    "selected_candidate",
]

STRATEGY_SELECTION_COLUMNS = [
    "frequency",
    "model",
    "horizon",
    "target_mode",
    "regime",
    "confidence_threshold",
    "candidate_type",
    "total_eligible_rows",
    "n_obs",
    "evaluated_rows",
    "coverage_ratio",
    "accuracy",
    "pass_60",
    "pass_63",
    "pass_level",
    "selected_candidate",
    "selection_reason",
]

TUNING_SUMMARY_COLUMNS = [
    "frequency",
    "horizon",
    "model",
    "tuning_trials",
    "best_params",
    "best_validation_score",
    "tuning_metric",
    "tuning_status",
    "error_message",
    "tuning_backend",
]

MODEL_ERROR_COLUMNS = [
    "frequency",
    "ticker",
    "horizon",
    "model",
    "error_type",
    "error_message",
    "stage",
]

FETCH_SUMMARY_COLUMNS = [
    "ticker",
    "frequency",
    "requested_start",
    "requested_end",
    "actual_start",
    "actual_end",
    "rows",
    "status",
    "data_source",
    "request_attempts",
    "throttled_seconds",
    "provider_error_count",
    "provider_empty_count",
    "coverage_start_ok",
    "coverage_end_ok",
    "coverage_ratio",
    "coverage_start_tolerance_days",
    "coverage_end_tolerance_days",
    "benchmark_usable",
    "benchmark_usable_reason",
    "effective_start",
    "effective_end",
    "pre_eval_rows",
    "eval_rows",
    "min_pre_eval_rows",
    "min_eval_rows",
    "invalid_reason",
    "error_message",
]

USABLE_CACHE_SUMMARY_COLUMNS = [
    "ticker",
    "frequency",
    "status",
    "data_source",
    "actual_start",
    "actual_end",
    "benchmark_usable",
    "benchmark_usable_reason",
    "pre_eval_rows",
    "eval_rows",
    "effective_start",
    "effective_end",
]

SOURCE_HEALTH_COLUMNS = [
    "source",
    "frequency",
    "calls",
    "successes",
    "empty_responses",
    "failures",
    "skipped_by_cooldown",
]

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
DAILY_COLUMNS = ["date", "ticker", *OHLCV_COLUMNS]
HOURLY_COLUMNS = ["datetime", "ticker", *OHLCV_COLUMNS]
DEFAULT_MODELS = "xgboost,lightgbm,random_forest,stacking"
DEFAULT_DAILY_HORIZONS = "1,5,10,20"
DEFAULT_HOURLY_HORIZONS = "1,4,8,20"
EVALUATION_TYPE = "walk_forward_out_of_sample"
DEFAULT_MIN_OBS_PER_GROUP = 50
DEFAULT_MAX_DAILY_GAP_DAYS = 30
DEFAULT_RATE_LIMIT_PER_MINUTE = 120
DEFAULT_MAX_FETCH_RETRIES = 1
DEFAULT_FETCH_BATCH_SIZE = 10
DEFAULT_FETCH_BATCH_COOLDOWN_SECONDS = 30.0
DEFAULT_SOURCE_FAILURE_THRESHOLD = 3
DEFAULT_SOURCE_EMPTY_THRESHOLD = 5
DEFAULT_SOURCE_COOLDOWN_SECONDS = 300.0
DEFAULT_MIN_PROVIDER_DAILY_ROWS = 150
DEFAULT_MIN_PROVIDER_HOURLY_ROWS = 500
DEFAULT_COVERAGE_START_TOLERANCE_DAYS = 7
DEFAULT_COVERAGE_END_TOLERANCE_DAYS = 3
DEFAULT_MIN_COVERAGE_RATIO = 0.80
DEFAULT_MIN_PRE_EVAL_ROWS_DAILY = 120
DEFAULT_MIN_PRE_EVAL_ROWS_HOURLY = 500
DEFAULT_MIN_EVAL_ROWS_DAILY = 60
DEFAULT_MIN_EVAL_ROWS_HOURLY = 500
DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_REGIME_RETURN_WINDOW = 20
DEFAULT_REGIME_VOL_WINDOW = 20
DEFAULT_REGIME_BULL_THRESHOLD = 0.03
DEFAULT_REGIME_BEAR_THRESHOLD = -0.03
DEFAULT_REGIME_VOL_QUANTILE = 0.70
DEFAULT_CONFIDENCE_THRESHOLD = 0.55
DEFAULT_CONFIDENCE_THRESHOLD_GRID = "0.50,0.525,0.55,0.575,0.60,0.625,0.65,0.675,0.70"
DEFAULT_NO_TRADE_BAND = 0.00
DEFAULT_MIN_COVERAGE_AFTER_FILTER = 0.30
DEFAULT_MIN_SWEEP_COVERAGE = 0.30
DEFAULT_TUNING_MODELS = "xgboost,lightgbm"
DEFAULT_TUNING_TRIALS = 20
DEFAULT_TUNING_SEED = 42
DEFAULT_TUNING_METRIC = "directional_accuracy"
DEFAULT_PROVIDER_TIMEOUT_SECONDS = 30.0
DEFAULT_MODEL_FIT_TIMEOUT_SECONDS = 300.0
DEFAULT_CACHE_DIR = "data/market_cache/vnstock_data/vn100"
UNIVERSE_CACHE_PATH = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "universe" / "VN100.csv"
QUOTE_SOURCE_PRIORITY = ("VCI", "KBS", "VND", "MAS")
BENCHMARK_SUPPORTED_MODELS = ("lightgbm", "random_forest", "stacking", "xgboost")
BENCHMARK_STACKING_BASE_MODELS = ("xgboost", "lightgbm", "random_forest")
_BINOMIAL_WARNING_EMITTED = False


@dataclass(frozen=True)
class BenchmarkConfig:
    universe: str
    daily_start: str
    daily_end: str
    hourly_start: str
    hourly_end: str
    train_cutoff: str | None
    eval_start: str
    eval_end: str
    models: list[str]
    daily_horizons: list[int]
    hourly_horizons: list[int]
    threshold: float
    provider: str
    pull_missing: bool
    cache_dir: str
    output_dir: str
    cache_only: bool
    fetch_only: bool
    resume_fetch: bool
    provider_calls_allowed: bool
    checkpointing_enabled: bool
    max_tickers: int | None
    min_history_days: int | None
    min_obs_per_group: int
    max_daily_gap_days: int
    rate_limit_per_minute: int
    request_sleep_seconds: float | None
    max_fetch_retries: int
    fetch_batch_size: int
    fetch_batch_cooldown_seconds: float
    source_failure_threshold: int
    source_empty_threshold: int
    source_cooldown_seconds: float
    min_provider_daily_rows: int
    min_provider_hourly_rows: int
    coverage_start_tolerance_days: int
    coverage_end_tolerance_days: int
    min_coverage_ratio: float
    allow_partial_cache_for_benchmark: bool
    min_pre_eval_rows_daily: int
    min_pre_eval_rows_hourly: int
    min_eval_rows_daily: int
    min_eval_rows_hourly: int
    bootstrap_samples: int
    bootstrap_seed: int
    enable_regime_evaluation: bool
    regime_return_window: int
    regime_vol_window: int
    regime_bull_threshold: float
    regime_bear_threshold: float
    regime_vol_quantile: float
    target_mode: str
    enable_confidence_filter: bool
    confidence_threshold: float
    enable_confidence_threshold_sweep: bool
    confidence_threshold_grid: list[float]
    min_sweep_coverage: float
    no_trade_band: float
    min_coverage_after_filter: float
    enable_horizon_tuning: bool
    tuning_models: list[str]
    tuning_trials: int
    tuning_seed: int
    tuning_metric: str
    tuning_time_budget_seconds: float | None
    tuning_output_dir: str | None
    provider_timeout_seconds: float
    model_fit_timeout_seconds: float
    retrain_frequency: str
    seed: int


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="VN100 hybrid-frequency walk-forward accuracy benchmark")
    parser.add_argument("--universe", default="VN100")
    parser.add_argument("--daily-start", default="2006-01-01")
    parser.add_argument("--daily-end", default="2015-12-31")
    parser.add_argument("--hourly-start", default="2016-01-01")
    parser.add_argument("--hourly-end", default="2024-12-31")
    parser.add_argument("--train-cutoff", default=None)
    parser.add_argument("--eval-start", default="2025-01-01")
    parser.add_argument("--eval-end", default="2025-12-31")
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--daily-horizons", default=DEFAULT_DAILY_HORIZONS)
    parser.add_argument("--hourly-horizons", default=DEFAULT_HOURLY_HORIZONS)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--provider", default="vnstock_data")
    parser.add_argument("--pull-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--resume-fetch", action="store_true")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", default="outputs/vn100_hybrid_accuracy_benchmark")
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--min-history-days", type=int, default=None)
    parser.add_argument("--min-obs-per-group", type=int, default=DEFAULT_MIN_OBS_PER_GROUP)
    parser.add_argument("--max-daily-gap-days", type=int, default=DEFAULT_MAX_DAILY_GAP_DAYS)
    parser.add_argument("--rate-limit-per-minute", type=int, default=DEFAULT_RATE_LIMIT_PER_MINUTE)
    parser.add_argument("--request-sleep-seconds", type=float, default=None)
    parser.add_argument("--max-fetch-retries", type=int, default=DEFAULT_MAX_FETCH_RETRIES)
    parser.add_argument("--fetch-batch-size", type=int, default=DEFAULT_FETCH_BATCH_SIZE)
    parser.add_argument("--fetch-batch-cooldown-seconds", type=float, default=DEFAULT_FETCH_BATCH_COOLDOWN_SECONDS)
    parser.add_argument("--source-failure-threshold", type=int, default=DEFAULT_SOURCE_FAILURE_THRESHOLD)
    parser.add_argument("--source-empty-threshold", type=int, default=DEFAULT_SOURCE_EMPTY_THRESHOLD)
    parser.add_argument("--source-cooldown-seconds", type=float, default=DEFAULT_SOURCE_COOLDOWN_SECONDS)
    parser.add_argument("--min-provider-daily-rows", type=int, default=DEFAULT_MIN_PROVIDER_DAILY_ROWS)
    parser.add_argument("--min-provider-hourly-rows", type=int, default=DEFAULT_MIN_PROVIDER_HOURLY_ROWS)
    parser.add_argument("--coverage-start-tolerance-days", type=int, default=DEFAULT_COVERAGE_START_TOLERANCE_DAYS)
    parser.add_argument("--coverage-end-tolerance-days", type=int, default=DEFAULT_COVERAGE_END_TOLERANCE_DAYS)
    parser.add_argument("--min-coverage-ratio", type=float, default=DEFAULT_MIN_COVERAGE_RATIO)
    parser.add_argument("--allow-partial-cache-for-benchmark", action="store_true")
    parser.add_argument("--min-pre-eval-rows-daily", type=int, default=DEFAULT_MIN_PRE_EVAL_ROWS_DAILY)
    parser.add_argument("--min-pre-eval-rows-hourly", type=int, default=DEFAULT_MIN_PRE_EVAL_ROWS_HOURLY)
    parser.add_argument("--min-eval-rows-daily", type=int, default=DEFAULT_MIN_EVAL_ROWS_DAILY)
    parser.add_argument("--min-eval-rows-hourly", type=int, default=DEFAULT_MIN_EVAL_ROWS_HOURLY)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--enable-regime-evaluation", action="store_true")
    parser.add_argument("--regime-return-window", type=int, default=DEFAULT_REGIME_RETURN_WINDOW)
    parser.add_argument("--regime-vol-window", type=int, default=DEFAULT_REGIME_VOL_WINDOW)
    parser.add_argument("--regime-bull-threshold", type=float, default=DEFAULT_REGIME_BULL_THRESHOLD)
    parser.add_argument("--regime-bear-threshold", type=float, default=DEFAULT_REGIME_BEAR_THRESHOLD)
    parser.add_argument("--regime-vol-quantile", type=float, default=DEFAULT_REGIME_VOL_QUANTILE)
    parser.add_argument("--target-mode", choices=["regression", "classification"], default="regression")
    parser.add_argument("--enable-confidence-filter", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--enable-confidence-threshold-sweep", action="store_true")
    parser.add_argument("--confidence-threshold-grid", default=DEFAULT_CONFIDENCE_THRESHOLD_GRID)
    parser.add_argument("--min-sweep-coverage", type=float, default=DEFAULT_MIN_SWEEP_COVERAGE)
    parser.add_argument("--no-trade-band", type=float, default=DEFAULT_NO_TRADE_BAND)
    parser.add_argument("--min-coverage-after-filter", type=float, default=DEFAULT_MIN_COVERAGE_AFTER_FILTER)
    parser.add_argument("--enable-horizon-tuning", action="store_true")
    parser.add_argument("--tuning-models", default=DEFAULT_TUNING_MODELS)
    parser.add_argument("--tuning-trials", type=int, default=DEFAULT_TUNING_TRIALS)
    parser.add_argument("--tuning-seed", type=int, default=DEFAULT_TUNING_SEED)
    parser.add_argument("--tuning-metric", choices=["directional_accuracy"], default=DEFAULT_TUNING_METRIC)
    parser.add_argument("--tuning-time-budget-seconds", type=float, default=None)
    parser.add_argument("--tuning-output-dir", default=None)
    parser.add_argument("--provider-timeout-seconds", type=float, default=DEFAULT_PROVIDER_TIMEOUT_SECONDS)
    parser.add_argument("--model-fit-timeout-seconds", type=float, default=DEFAULT_MODEL_FIT_TIMEOUT_SECONDS)
    parser.add_argument("--retrain-frequency", default="monthly", choices=["daily", "weekly", "monthly", "quarterly", "never"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.cache_only and args.fetch_only:
        raise ValueError("--cache-only and --fetch-only are mutually exclusive")
    train_cutoff = str(args.train_cutoff).strip() if args.train_cutoff is not None and str(args.train_cutoff).strip() else None
    if train_cutoff is not None and _to_date(train_cutoff) >= _to_date(args.eval_start):
        raise ValueError("--train-cutoff must be earlier than --eval-start")
    cache_dir = str(args.cache_dir)
    if args.cache_only:
        required_cache_dir = (REPO_ROOT / DEFAULT_CACHE_DIR).resolve()
        supplied_cache_dir = (REPO_ROOT / cache_dir).resolve() if not Path(cache_dir).is_absolute() else Path(cache_dir).resolve()
        if supplied_cache_dir != required_cache_dir:
            raise ValueError(
                "--cache-only must read OHLCV from "
                f"{DEFAULT_CACHE_DIR}; got {args.cache_dir!r}"
            )
        cache_dir = DEFAULT_CACHE_DIR
    models = _parse_str_list(args.models)
    if not models:
        raise ValueError("--models must contain at least one model")
    _validate_benchmark_models(models)
    tuning_models = _parse_str_list(args.tuning_models)
    unsupported_tuning_models = [model for model in tuning_models if model not in BENCHMARK_SUPPORTED_MODELS]
    if unsupported_tuning_models:
        raise ValueError(
            f"Unsupported --tuning-models value(s): {sorted(dict.fromkeys(unsupported_tuning_models))}. "
            f"Supported benchmark models: {list(BENCHMARK_SUPPORTED_MODELS)}."
        )
    pull_missing = False if args.cache_only else bool(args.pull_missing)
    provider_calls_allowed = bool(not args.cache_only and pull_missing)
    return BenchmarkConfig(
        universe=str(args.universe).upper(),
        daily_start=args.daily_start,
        daily_end=args.daily_end,
        hourly_start=args.hourly_start,
        hourly_end=args.hourly_end,
        train_cutoff=train_cutoff,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        models=models,
        daily_horizons=_parse_int_list(args.daily_horizons, "--daily-horizons"),
        hourly_horizons=_parse_int_list(args.hourly_horizons, "--hourly-horizons"),
        threshold=float(args.threshold),
        provider=str(args.provider),
        pull_missing=pull_missing,
        cache_dir=cache_dir,
        output_dir=str(args.output_dir),
        cache_only=bool(args.cache_only),
        fetch_only=bool(args.fetch_only),
        resume_fetch=bool(args.resume_fetch),
        provider_calls_allowed=provider_calls_allowed,
        checkpointing_enabled=True,
        max_tickers=args.max_tickers,
        min_history_days=args.min_history_days,
        min_obs_per_group=max(int(args.min_obs_per_group), 1),
        max_daily_gap_days=max(int(args.max_daily_gap_days), 1),
        rate_limit_per_minute=max(int(args.rate_limit_per_minute), 1),
        request_sleep_seconds=float(args.request_sleep_seconds) if args.request_sleep_seconds is not None else None,
        max_fetch_retries=max(int(args.max_fetch_retries), 1),
        fetch_batch_size=max(int(args.fetch_batch_size), 1),
        fetch_batch_cooldown_seconds=max(float(args.fetch_batch_cooldown_seconds), 0.0),
        source_failure_threshold=max(int(args.source_failure_threshold), 1),
        source_empty_threshold=max(int(args.source_empty_threshold), 1),
        source_cooldown_seconds=max(float(args.source_cooldown_seconds), 0.0),
        min_provider_daily_rows=max(int(args.min_provider_daily_rows), 1),
        min_provider_hourly_rows=max(int(args.min_provider_hourly_rows), 1),
        coverage_start_tolerance_days=max(int(args.coverage_start_tolerance_days), 0),
        coverage_end_tolerance_days=max(int(args.coverage_end_tolerance_days), 0),
        min_coverage_ratio=min(max(float(args.min_coverage_ratio), 0.0), 1.0),
        allow_partial_cache_for_benchmark=bool(args.allow_partial_cache_for_benchmark),
        min_pre_eval_rows_daily=max(int(args.min_pre_eval_rows_daily), 1),
        min_pre_eval_rows_hourly=max(int(args.min_pre_eval_rows_hourly), 1),
        min_eval_rows_daily=max(int(args.min_eval_rows_daily), 1),
        min_eval_rows_hourly=max(int(args.min_eval_rows_hourly), 1),
        bootstrap_samples=max(int(args.bootstrap_samples), 0),
        bootstrap_seed=int(args.bootstrap_seed),
        enable_regime_evaluation=bool(args.enable_regime_evaluation),
        regime_return_window=max(int(args.regime_return_window), 2),
        regime_vol_window=max(int(args.regime_vol_window), 2),
        regime_bull_threshold=float(args.regime_bull_threshold),
        regime_bear_threshold=float(args.regime_bear_threshold),
        regime_vol_quantile=min(max(float(args.regime_vol_quantile), 0.01), 0.99),
        target_mode=str(args.target_mode).lower(),
        enable_confidence_filter=bool(args.enable_confidence_filter),
        confidence_threshold=min(max(float(args.confidence_threshold), 0.0), 1.0),
        enable_confidence_threshold_sweep=bool(args.enable_confidence_threshold_sweep),
        confidence_threshold_grid=_parse_float_list(args.confidence_threshold_grid, "--confidence-threshold-grid"),
        min_sweep_coverage=min(max(float(args.min_sweep_coverage), 0.0), 1.0),
        no_trade_band=max(float(args.no_trade_band), 0.0),
        min_coverage_after_filter=min(max(float(args.min_coverage_after_filter), 0.0), 1.0),
        enable_horizon_tuning=bool(args.enable_horizon_tuning),
        tuning_models=tuning_models,
        tuning_trials=max(int(args.tuning_trials), 1),
        tuning_seed=int(args.tuning_seed),
        tuning_metric=str(args.tuning_metric).lower(),
        tuning_time_budget_seconds=max(float(args.tuning_time_budget_seconds), 0.0)
        if args.tuning_time_budget_seconds is not None
        else None,
        tuning_output_dir=str(args.tuning_output_dir) if args.tuning_output_dir else None,
        provider_timeout_seconds=max(float(args.provider_timeout_seconds), 0.0),
        model_fit_timeout_seconds=max(float(args.model_fit_timeout_seconds), 0.0),
        retrain_frequency=str(args.retrain_frequency).lower(),
        seed=int(args.seed),
    )


def _parse_str_list(raw: str) -> list[str]:
    return [value.strip().lower() for value in str(raw).split(",") if value.strip()]


def _validate_benchmark_models(models: list[str]) -> None:
    unsupported = [model for model in models if model not in BENCHMARK_SUPPORTED_MODELS]
    if unsupported:
        raise ValueError(
            "Unsupported --models value(s) for VN100 hybrid-frequency benchmark: "
            f"{sorted(dict.fromkeys(unsupported))}. "
            f"Supported benchmark models: {list(BENCHMARK_SUPPORTED_MODELS)}. "
            "Other repository model implementations exist, but they are not wired into this benchmark runner."
        )


def _parse_int_list(raw: str, name: str) -> list[int]:
    values = []
    for value in str(raw).split(","):
        if not value.strip():
            continue
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"{name} values must be positive integers")
        if parsed not in values:
            values.append(parsed)
    if not values:
        raise ValueError(f"{name} must contain at least one horizon")
    return values


def _parse_float_list(raw: str, name: str) -> list[float]:
    values: list[float] = []
    for value in str(raw).split(","):
        if not value.strip():
            continue
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"{name} values must be finite")
        parsed = min(max(parsed, 0.0), 1.0)
        if parsed not in values:
            values.append(parsed)
    if not values:
        raise ValueError(f"{name} must contain at least one threshold")
    return values


def _to_date(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _to_end_of_day(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


def _finite_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


class ProviderRateLimiter:
    """Simple monotonic-clock spacing for provider requests."""

    def __init__(self, *, rate_limit_per_minute: int, request_sleep_seconds: float | None) -> None:
        self.interval_seconds = (
            max(float(request_sleep_seconds), 0.0)
            if request_sleep_seconds is not None
            else 60.0 / max(int(rate_limit_per_minute), 1)
        )
        self._last_request_time: float | None = None
        self.total_throttled_seconds = 0.0

    def wait(self) -> float:
        if self.interval_seconds <= 0.0:
            self._last_request_time = time.monotonic()
            return 0.0

        now = time.monotonic()
        sleep_seconds = self.interval_seconds
        if self._last_request_time is not None:
            elapsed = now - self._last_request_time
            sleep_seconds = max(0.0, self.interval_seconds - elapsed)
        if sleep_seconds > 0.0:
            time.sleep(sleep_seconds)
        self._last_request_time = time.monotonic()
        self.total_throttled_seconds += sleep_seconds
        return float(sleep_seconds)


def _provider_result_row_count(result: Any) -> int | None:
    if isinstance(result, (pd.DataFrame, pd.Series)):
        return int(len(result))
    if isinstance(result, (list, tuple)):
        return int(len(result))
    return None


def provider_call(
    *,
    rate_limiter: ProviderRateLimiter,
    call_type: str,
    func: Any,
    stats: dict[str, Any] | None = None,
    symbol: str | None = None,
    frequency: str | None = None,
    source: str | None = None,
) -> Any:
    # TODO: Enforce provider_timeout_seconds with a Windows-safe cooperative
    # boundary. Threads cannot safely kill provider calls, so the CLI option is
    # recorded in run metadata without changing current provider behavior.
    throttled_seconds = rate_limiter.wait()
    if stats is not None:
        stats["throttled_seconds"] = float(stats.get("throttled_seconds", 0.0)) + float(throttled_seconds)
        stats["request_attempts"] = int(stats.get("request_attempts", 0)) + 1

    log_context: dict[str, Any] = {
        "call_type": call_type,
        "throttled_seconds": float(throttled_seconds),
    }
    if symbol is not None:
        log_context["symbol"] = symbol
    if frequency is not None:
        log_context["frequency"] = frequency
    if source is not None:
        log_context["source"] = source

    logger.info("provider_call_started", **log_context)
    start_ns = time.perf_counter_ns()
    try:
        result = func()
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
        completed_context = {**log_context, "elapsed_ms": f"{elapsed_ms:.2f}"}
        rows = _provider_result_row_count(result)
        if rows is not None:
            completed_context["rows"] = rows
        logger.info("provider_call_completed", **completed_context)
        return result
    except Exception as exc:
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
        logger.warning(
            "provider_call_failed",
            **log_context,
            elapsed_ms=f"{elapsed_ms:.2f}",
            error=str(exc),
        )
        raise


@dataclass
class SourceHealthRecord:
    source: str
    frequency: str
    calls: int = 0
    successes: int = 0
    empty_responses: int = 0
    failures: int = 0
    skipped_by_cooldown: int = 0
    consecutive_failures: int = 0
    consecutive_empty: int = 0
    cooldown_until: float = 0.0


class SourceHealthManager:
    def __init__(
        self,
        *,
        failure_threshold: int,
        empty_threshold: int,
        cooldown_seconds: float,
    ) -> None:
        self.failure_threshold = max(int(failure_threshold), 1)
        self.empty_threshold = max(int(empty_threshold), 1)
        self.cooldown_seconds = max(float(cooldown_seconds), 0.0)
        self._records: dict[tuple[str, str], SourceHealthRecord] = {}
        for frequency in ("daily", "hourly"):
            for source in QUOTE_SOURCE_PRIORITY:
                self._record(source, frequency)

    def _record(self, source: str, frequency: str) -> SourceHealthRecord:
        key = (source.upper(), frequency.lower())
        if key not in self._records:
            self._records[key] = SourceHealthRecord(source=key[0], frequency=key[1])
        return self._records[key]

    def should_skip(self, source: str, frequency: str) -> bool:
        record = self._record(source, frequency)
        if record.cooldown_until > time.monotonic():
            record.skipped_by_cooldown += 1
            logger.info(
                "provider_source_skipped_by_cooldown",
                source=record.source,
                frequency=record.frequency,
                cooldown_remaining_seconds=max(record.cooldown_until - time.monotonic(), 0.0),
            )
            return True
        return False

    def record_call(self, source: str, frequency: str) -> None:
        self._record(source, frequency).calls += 1

    def record_success(self, source: str, frequency: str) -> None:
        record = self._record(source, frequency)
        record.successes += 1
        record.consecutive_failures = 0
        record.consecutive_empty = 0
        record.cooldown_until = 0.0

    def record_empty(self, source: str, frequency: str) -> None:
        record = self._record(source, frequency)
        record.empty_responses += 1
        record.consecutive_empty += 1
        record.consecutive_failures = 0
        if record.consecutive_empty >= self.empty_threshold:
            self._start_cooldown(record, reason="empty_threshold")

    def record_failure(self, source: str, frequency: str) -> None:
        record = self._record(source, frequency)
        record.failures += 1
        record.consecutive_failures += 1
        record.consecutive_empty = 0
        if record.consecutive_failures >= self.failure_threshold:
            self._start_cooldown(record, reason="failure_threshold")

    def _start_cooldown(self, record: SourceHealthRecord, *, reason: str) -> None:
        if self.cooldown_seconds <= 0.0:
            record.consecutive_failures = 0
            record.consecutive_empty = 0
            return
        record.cooldown_until = time.monotonic() + self.cooldown_seconds
        logger.warning(
            "provider_source_cooldown_started",
            source=record.source,
            frequency=record.frequency,
            reason=reason,
            cooldown_seconds=self.cooldown_seconds,
        )
        record.consecutive_failures = 0
        record.consecutive_empty = 0

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                "source": record.source,
                "frequency": record.frequency,
                "calls": int(record.calls),
                "successes": int(record.successes),
                "empty_responses": int(record.empty_responses),
                "failures": int(record.failures),
                "skipped_by_cooldown": int(record.skipped_by_cooldown),
            }
            for record in self._records.values()
        ]
        return pd.DataFrame(rows, columns=SOURCE_HEALTH_COLUMNS).sort_values(
            ["frequency", "source"]
        ).reset_index(drop=True)


class SourcePreferenceCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._preferences: dict[str, dict[str, str]] = {"daily": {}, "hourly": {}}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("source_preference_load_failed", path=str(self.path), error=str(exc))
            return
        if not isinstance(payload, dict):
            return
        for frequency in ("daily", "hourly"):
            values = payload.get(frequency, {})
            if not isinstance(values, dict):
                continue
            self._preferences[frequency] = {
                str(ticker).upper().strip(): str(source).upper().strip()
                for ticker, source in values.items()
                if str(ticker).strip() and str(source).upper().strip() in QUOTE_SOURCE_PRIORITY
            }

    def ordered_sources(self, ticker: str, frequency: str) -> list[str]:
        base = list(QUOTE_SOURCE_PRIORITY)
        preferred = self._preferences.get(frequency.lower(), {}).get(ticker.upper().strip())
        if preferred in base:
            return [preferred, *[source for source in base if source != preferred]]
        return base

    def remember(self, ticker: str, frequency: str, source: str) -> None:
        frequency = frequency.lower()
        ticker = ticker.upper().strip()
        source = source.upper().strip()
        if frequency not in self._preferences or source not in QUOTE_SOURCE_PRIORITY or not ticker:
            return
        if self._preferences[frequency].get(ticker) == source:
            return
        self._preferences[frequency][ticker] = source
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(_json_safe(self._preferences), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self._dirty = False


def _find_time_column(frame: pd.DataFrame) -> str | None:
    for candidate in ("datetime", "timestamp", "time", "date", "trading_date"):
        if candidate in frame.columns:
            return candidate
    if isinstance(frame.index, pd.DatetimeIndex):
        return "__index__"
    return None


def _safe_iso_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(pd.Timestamp(value).date())


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _empty_frequency_frame(frequency: str) -> pd.DataFrame:
    return pd.DataFrame(columns=HOURLY_COLUMNS if frequency == "hourly" else DAILY_COLUMNS)


def _apply_max_tickers(tickers: list[str], config: BenchmarkConfig) -> list[str]:
    normalized = [str(ticker).upper().strip() for ticker in tickers if str(ticker).strip()]
    normalized = sorted(dict.fromkeys(normalized))
    if config.max_tickers is not None:
        if int(config.max_tickers) <= 0:
            raise ValueError("--max-tickers must be positive when supplied")
        normalized = normalized[: int(config.max_tickers)]
    return normalized


def _read_universe_cache(config: BenchmarkConfig, *, required: bool) -> list[str]:
    if not UNIVERSE_CACHE_PATH.exists():
        if required:
            raise FileNotFoundError(
                "Cache-only mode requires VN100 universe cache at "
                f"{UNIVERSE_CACHE_PATH}"
            )
        return []
    try:
        frame = pd.read_csv(UNIVERSE_CACHE_PATH, low_memory=False)
    except Exception as exc:
        if required:
            raise ValueError(f"Unable to read VN100 universe cache {UNIVERSE_CACHE_PATH}: {exc}") from exc
        logger.warning("vn100_universe_cache_read_failed", path=str(UNIVERSE_CACHE_PATH), error=str(exc))
        return []
    if frame.empty:
        if required:
            raise ValueError(f"VN100 universe cache is empty: {UNIVERSE_CACHE_PATH}")
        return []
    ticker_column = next(
        (column for column in ("ticker", "symbol", "code") if column in frame.columns),
        frame.columns[0],
    )
    tickers = _apply_max_tickers(frame[ticker_column].dropna().astype(str).tolist(), config)
    if required and not tickers:
        raise ValueError(f"VN100 universe cache resolved to zero tickers: {UNIVERSE_CACHE_PATH}")
    if tickers:
        logger.info("vn100_resolved_from_universe_cache", count=len(tickers), path=str(UNIVERSE_CACHE_PATH))
    return tickers


def _write_universe_cache(tickers: list[str], config: BenchmarkConfig) -> None:
    normalized = _apply_max_tickers(tickers, config)
    if not normalized:
        return
    UNIVERSE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": normalized}).to_csv(UNIVERSE_CACHE_PATH, index=False)


def _resolve_tickers_from_complete_cache(config: BenchmarkConfig) -> list[str]:
    cache_dir = Path(config.cache_dir)
    tickers = _apply_max_tickers(VN100_BACKUP_TICKERS.copy(), config)
    if not tickers:
        return []
    for ticker in tickers:
        daily = load_cache(cache_dir, "daily", ticker, config.daily_start, config.daily_end)
        hourly = load_cache(cache_dir, "hourly", ticker, config.hourly_start, config.hourly_end)
        if not _coverage_ok(daily, "daily", config.daily_start, config.daily_end, config):
            return []
        if not _coverage_ok(hourly, "hourly", config.hourly_start, config.hourly_end, config):
            return []
    logger.info("vn100_resolved_from_complete_cache", count=len(tickers), cache_dir=str(cache_dir))
    return tickers


def resolve_tickers(config: BenchmarkConfig) -> list[str]:
    if config.universe != "VN100":
        raise ValueError(f"Unsupported universe: {config.universe}. Only VN100 is implemented.")
    if config.cache_only:
        return _read_universe_cache(config, required=True)

    cached_universe = _read_universe_cache(config, required=False)
    if cached_universe:
        return cached_universe

    cached_tickers = _resolve_tickers_from_complete_cache(config)
    if cached_tickers:
        return cached_tickers

    tickers: list[str] = []
    if config.provider_calls_allowed:
        limiter = ProviderRateLimiter(
            rate_limit_per_minute=config.rate_limit_per_minute,
            request_sleep_seconds=config.request_sleep_seconds,
        )
        adapter = VnstockAdapter(rate_limiter=limiter)
        Listing = adapter._get_class("Listing")
        if Listing is not None:
            for source in ("VCI", "KBS"):
                try:
                    result = provider_call(
                        rate_limiter=limiter,
                        call_type="Listing.symbols_by_group",
                        symbol="VN100",
                        source=source,
                        func=lambda source=source: Listing(source=source).symbols_by_group("VN100"),
                    )
                    frame = VnstockAdapter._as_dataframe(result)
                    if frame is not None and not frame.empty:
                        first_col = frame.columns[0]
                        tickers = frame[first_col].dropna().astype(str).str.upper().str.strip().tolist()
                        if tickers:
                            break
                except Exception as exc:
                    logger.warning("vn100_universe_source_failed", source=source, error=str(exc))
                    continue
    if not tickers:
        tickers = VN100_BACKUP_TICKERS.copy()
    tickers = _apply_max_tickers(tickers, config)
    if not tickers:
        raise ValueError("VN100 universe resolved to zero tickers")
    return tickers


def _assign_ticker(prepared: pd.DataFrame, ticker: str) -> pd.Series:
    if ticker == "__KEEP__" and "ticker" in prepared.columns:
        return prepared["ticker"].astype(str).str.upper().str.strip()
    return pd.Series(ticker.upper().strip(), index=prepared.index)


def standardize_daily(frame: pd.DataFrame, ticker: str, start: str, end: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    prepared = frame.copy()
    time_column = _find_time_column(prepared)
    if time_column == "__index__":
        prepared = prepared.reset_index().rename(columns={prepared.index.name or "index": "date"})
        time_column = "date"
    if time_column is None:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    if time_column != "date":
        prepared = prepared.rename(columns={time_column: "date"})
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce").dt.normalize()
    prepared["ticker"] = _assign_ticker(prepared, ticker)
    for column in OHLCV_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = np.nan
        prepared[column] = _finite_numeric(prepared[column])
    start_ts = _to_date(start)
    end_ts = _to_date(end)
    prepared = prepared[(prepared["date"] >= start_ts) & (prepared["date"] <= end_ts)].copy()
    prepared = prepared.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    prepared = prepared[(prepared["close"] > 0.0) & (prepared["volume"] >= 0.0)].copy()
    prepared = prepared.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    return prepared[DAILY_COLUMNS].reset_index(drop=True)


def standardize_hourly(frame: pd.DataFrame, ticker: str, start: str, end: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=HOURLY_COLUMNS)
    prepared = frame.copy()
    time_column = _find_time_column(prepared)
    if time_column == "__index__":
        prepared = prepared.reset_index().rename(columns={prepared.index.name or "index": "datetime"})
        time_column = "datetime"
    if time_column is None:
        return pd.DataFrame(columns=HOURLY_COLUMNS)
    if time_column != "datetime":
        prepared = prepared.rename(columns={time_column: "datetime"})
    prepared["datetime"] = pd.to_datetime(prepared["datetime"], errors="coerce")
    prepared["ticker"] = _assign_ticker(prepared, ticker)
    for column in OHLCV_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = np.nan
        prepared[column] = _finite_numeric(prepared[column])
    start_ts = _to_date(start)
    end_ts = _to_end_of_day(end)
    prepared = prepared[(prepared["datetime"] >= start_ts) & (prepared["datetime"] <= end_ts)].copy()
    prepared = prepared.dropna(subset=["datetime", "open", "high", "low", "close", "volume"])
    prepared = prepared[(prepared["close"] > 0.0) & (prepared["volume"] >= 0.0)].copy()
    prepared = prepared.sort_values(["ticker", "datetime"]).drop_duplicates(["ticker", "datetime"], keep="last")
    return prepared[HOURLY_COLUMNS].reset_index(drop=True)


def _date_bounds(frame: pd.DataFrame, frequency: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    column = "datetime" if frequency == "hourly" else "date"
    if frame.empty or column not in frame.columns:
        return None, None
    values = pd.to_datetime(frame[column], errors="coerce").dropna()
    if values.empty:
        return None, None
    return pd.Timestamp(values.min()).normalize(), pd.Timestamp(values.max()).normalize()


def _requested_span_days(start: str, end: str) -> int:
    return max(int((_to_date(end) - _to_date(start)).days) + 1, 1)


def _coverage_ratio(
    frame: pd.DataFrame,
    *,
    frequency: str,
    start: str,
    end: str,
) -> float:
    if frame is None or frame.empty:
        return 0.0
    requested_start = _to_date(start)
    requested_end = _to_date(end)
    if frequency == "daily":
        expected_rows = len(pd.bdate_range(requested_start, requested_end))
        if expected_rows <= 0:
            expected_rows = _requested_span_days(start, end)
        return min(float(len(frame)) / max(float(expected_rows), 1.0), 1.0)

    actual_start, actual_end = _date_bounds(frame, frequency)
    if actual_start is None or actual_end is None:
        return 0.0
    overlap_start = max(requested_start, actual_start)
    overlap_end = min(requested_end, actual_end)
    overlap_days = max(int((overlap_end - overlap_start).days) + 1, 0)
    return min(float(overlap_days) / float(_requested_span_days(start, end)), 1.0)


def _coverage_metrics(
    frame: pd.DataFrame,
    frequency: str,
    start: str,
    end: str,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    actual_start, actual_end = _date_bounds(frame, frequency)
    requested_start = _to_date(start)
    requested_end = _to_date(end)
    start_tolerance = int(config.coverage_start_tolerance_days)
    end_tolerance = int(config.coverage_end_tolerance_days)
    start_ok = bool(actual_start is not None and actual_start <= requested_start + pd.Timedelta(days=start_tolerance))
    end_ok = bool(actual_end is not None and actual_end >= requested_end - pd.Timedelta(days=end_tolerance))
    ratio = _coverage_ratio(frame, frequency=frequency, start=start, end=end)
    ratio_ok = bool(ratio >= float(config.min_coverage_ratio))
    return {
        "coverage_start_ok": start_ok,
        "coverage_end_ok": end_ok,
        "coverage_ratio": float(ratio),
        "coverage_start_tolerance_days": start_tolerance,
        "coverage_end_tolerance_days": end_tolerance,
        "coverage_ok": bool(start_ok and end_ok and ratio_ok),
    }


def _benchmark_row_thresholds(frequency: str, config: BenchmarkConfig) -> tuple[int, int]:
    if frequency == "hourly":
        return int(config.min_pre_eval_rows_hourly), int(config.min_eval_rows_hourly)
    return int(config.min_pre_eval_rows_daily), int(config.min_eval_rows_daily)


def _training_label_cutoff(config: BenchmarkConfig, fallback_end: str | None = None) -> pd.Timestamp | None:
    cutoff = getattr(config, "train_cutoff", None) or fallback_end
    if cutoff is None or str(cutoff).strip() == "":
        return None
    return _to_end_of_day(str(cutoff))


def _training_label_cutoff_rule(config: BenchmarkConfig) -> str:
    if getattr(config, "train_cutoff", None):
        return "target_timestamp <= train_cutoff"
    return "target_timestamp < forecast_chunk_start"


def _config_data_end(config: BenchmarkConfig) -> str:
    candidates = []
    for value in (config.daily_end, config.hourly_end, config.eval_end):
        try:
            candidates.append(_to_date(value))
        except Exception:
            continue
    if not candidates:
        return ""
    return str(max(candidates).date())


def _benchmark_metadata_fields(config: BenchmarkConfig, *, data_end: str | None = None) -> dict[str, Any]:
    return {
        "train_cutoff": getattr(config, "train_cutoff", None) or "",
        "data_end": data_end or _config_data_end(config),
        "eval_start": config.eval_start,
        "eval_end": config.eval_end,
        "training_label_cutoff_rule": _training_label_cutoff_rule(config),
        "actual_rows_allowed_after_train_cutoff": True,
    }


def _run_config_payload(config: BenchmarkConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload.update(_benchmark_metadata_fields(config, data_end=_config_data_end(config)))
    return payload


def _training_label_mask(
    labeled: pd.DataFrame,
    *,
    initial_train_start: str,
    forecast_chunk_start: pd.Timestamp,
    config: BenchmarkConfig,
) -> pd.Series:
    timestamps = pd.to_datetime(labeled["timestamp"], errors="coerce")
    target_timestamps = pd.to_datetime(labeled["target_timestamp"], errors="coerce")
    mask = (
        (timestamps >= _to_date(initial_train_start))
        & target_timestamps.notna()
        & labeled["actual_return"].notna()
    )
    cutoff = _training_label_cutoff(config)
    if cutoff is not None:
        return mask & (target_timestamps <= cutoff)
    return mask & (target_timestamps < pd.Timestamp(forecast_chunk_start))


def _benchmark_usability_metrics(
    frame: pd.DataFrame,
    frequency: str,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    actual_start, actual_end = _date_bounds(frame, frequency)
    min_pre_eval_rows, min_eval_rows = _benchmark_row_thresholds(frequency, config)
    pre_eval_rows = 0
    eval_rows = 0
    if frame is not None and not frame.empty:
        time_column = "datetime" if frequency == "hourly" else "date"
        if time_column in frame.columns:
            timestamps = pd.to_datetime(frame[time_column], errors="coerce").dropna()
            eval_start = _to_date(config.eval_start)
            eval_end = _to_end_of_day(config.eval_end)
            train_cutoff = _training_label_cutoff(config)
            pre_eval_boundary = train_cutoff if train_cutoff is not None else eval_start
            pre_eval_rows = int((timestamps <= pre_eval_boundary).sum()) if train_cutoff is not None else int((timestamps < pre_eval_boundary).sum())
            eval_rows = int(((timestamps >= eval_start) & (timestamps <= eval_end)).sum())

    eval_start_day = _to_date(config.eval_start)
    eval_end_floor = _to_date(config.eval_end) - pd.Timedelta(days=int(config.coverage_end_tolerance_days))
    reasons: list[str] = []
    if frame is None or frame.empty or actual_start is None or actual_end is None:
        reasons.append("cache_missing")
    else:
        if not actual_start < eval_start_day:
            reasons.append("starts_at_or_after_eval_start")
        if not actual_end >= eval_end_floor:
            reasons.append("ends_before_eval_end_tolerance")
    if pre_eval_rows < min_pre_eval_rows:
        reasons.append(f"pre_eval_rows_below_min:{pre_eval_rows}<{min_pre_eval_rows}")
    if eval_rows < min_eval_rows:
        reasons.append(f"eval_rows_below_min:{eval_rows}<{min_eval_rows}")

    usable = not reasons
    return {
        "benchmark_usable": bool(usable),
        "benchmark_usable_reason": "usable" if usable else "; ".join(dict.fromkeys(reasons)),
        "effective_start": _safe_iso_date(actual_start),
        "effective_end": _safe_iso_date(actual_end),
        "pre_eval_rows": int(pre_eval_rows),
        "eval_rows": int(eval_rows),
        "min_pre_eval_rows": int(min_pre_eval_rows),
        "min_eval_rows": int(min_eval_rows),
    }


def _minimum_provider_rows(
    *,
    frequency: str,
    start: str,
    end: str,
    config: BenchmarkConfig,
) -> int:
    span_days = _requested_span_days(start, end)
    if span_days <= 7:
        return 1
    if frequency == "daily":
        return min(int(config.min_provider_daily_rows), max(2, int(span_days * 0.55)))
    return min(int(config.min_provider_hourly_rows), max(10, int(span_days * 2)))


def _validate_provider_frame(
    frame: pd.DataFrame,
    *,
    frequency: str,
    start: str,
    end: str,
    config: BenchmarkConfig,
) -> tuple[bool, str]:
    if frame is None or frame.empty:
        return False, "empty_response"
    minimum_rows = _minimum_provider_rows(frequency=frequency, start=start, end=end, config=config)
    if len(frame) < minimum_rows:
        return False, "insufficient_rows"
    if not _coverage_metrics(frame, frequency, start, end, config)["coverage_ok"]:
        return False, "coverage_gap"
    return True, ""


def _validate_local_cache_frame(
    frame: pd.DataFrame,
    *,
    frequency: str,
    start: str,
    end: str,
    config: BenchmarkConfig,
) -> tuple[bool, str]:
    if frame is None or frame.empty:
        return False, "cache_missing"
    minimum_rows = _minimum_provider_rows(frequency=frequency, start=start, end=end, config=config)
    if len(frame) < minimum_rows:
        return False, "insufficient_rows"
    if not _coverage_metrics(frame, frequency, start, end, config)["coverage_ok"]:
        return False, "coverage_gap"
    return True, ""


def _coverage_ok(frame: pd.DataFrame, frequency: str, start: str, end: str, config: BenchmarkConfig) -> bool:
    return bool(_validate_local_cache_frame(frame, frequency=frequency, start=start, end=end, config=config)[0])


def _missing_ranges(frame: pd.DataFrame, frequency: str, start: str, end: str, config: BenchmarkConfig) -> list[tuple[str, str]]:
    start_ts = _to_date(start)
    end_ts = _to_date(end)
    actual_start, actual_end = _date_bounds(frame, frequency)
    if actual_start is None or actual_end is None:
        return [(str(start_ts.date()), str(end_ts.date()))]
    ranges: list[tuple[str, str]] = []
    if actual_start > start_ts + pd.Timedelta(days=int(config.coverage_start_tolerance_days)):
        ranges.append((str(start_ts.date()), str(min(end_ts, actual_start - pd.Timedelta(days=1)).date())))
    if actual_end < end_ts - pd.Timedelta(days=int(config.coverage_end_tolerance_days)):
        ranges.append((str(max(start_ts, actual_end + pd.Timedelta(days=1)).date()), str(end_ts.date())))
    return [(range_start, range_end) for range_start, range_end in ranges if pd.Timestamp(range_start) <= pd.Timestamp(range_end)]


def _cache_path(cache_dir: Path, frequency: str, ticker: str) -> Path:
    return cache_dir / frequency / f"{ticker.upper()}.csv"


def load_cache(cache_dir: Path, frequency: str, ticker: str, start: str, end: str) -> pd.DataFrame:
    path = _cache_path(cache_dir, frequency, ticker)
    if not path.exists():
        return _empty_frequency_frame(frequency)
    raw = pd.read_csv(path, low_memory=False)
    if frequency == "hourly":
        return standardize_hourly(raw, ticker, start, end)
    return standardize_daily(raw, ticker, start, end)


def _fetch_summary_row(
    *,
    ticker: str,
    frequency: str,
    start: str,
    end: str,
    frame: pd.DataFrame,
    status: str,
    data_source: str,
    request_attempts: int = 0,
    throttled_seconds: float = 0.0,
    provider_error_count: int = 0,
    provider_empty_count: int = 0,
    invalid_reasons: list[str] | None = None,
    errors: list[str] | None = None,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    invalid_reasons = invalid_reasons or []
    errors = errors or []
    actual_start, actual_end = _date_bounds(frame, frequency)
    coverage = _coverage_metrics(frame, frequency, start, end, config)
    usability = _benchmark_usability_metrics(frame, frequency, config)
    return {
        "ticker": ticker.upper(),
        "frequency": frequency,
        "requested_start": start,
        "requested_end": end,
        "actual_start": _safe_iso_date(actual_start),
        "actual_end": _safe_iso_date(actual_end),
        "rows": int(len(frame)),
        "status": status,
        "data_source": data_source,
        "request_attempts": int(request_attempts),
        "throttled_seconds": float(throttled_seconds),
        "provider_error_count": int(provider_error_count),
        "provider_empty_count": int(provider_empty_count),
        "coverage_start_ok": bool(coverage["coverage_start_ok"]),
        "coverage_end_ok": bool(coverage["coverage_end_ok"]),
        "coverage_ratio": float(coverage["coverage_ratio"]),
        "coverage_start_tolerance_days": int(coverage["coverage_start_tolerance_days"]),
        "coverage_end_tolerance_days": int(coverage["coverage_end_tolerance_days"]),
        "benchmark_usable": bool(usability["benchmark_usable"]),
        "benchmark_usable_reason": str(usability["benchmark_usable_reason"]),
        "effective_start": str(usability["effective_start"]),
        "effective_end": str(usability["effective_end"]),
        "pre_eval_rows": int(usability["pre_eval_rows"]),
        "eval_rows": int(usability["eval_rows"]),
        "min_pre_eval_rows": int(usability["min_pre_eval_rows"]),
        "min_eval_rows": int(usability["min_eval_rows"]),
        "invalid_reason": "; ".join(dict.fromkeys(reason for reason in invalid_reasons if reason)),
        "error_message": "; ".join(errors),
    }


def build_usable_cache_summary(fetch_summary: pd.DataFrame) -> pd.DataFrame:
    if fetch_summary.empty:
        return pd.DataFrame(columns=USABLE_CACHE_SUMMARY_COLUMNS)
    summary = fetch_summary.copy()
    for column in USABLE_CACHE_SUMMARY_COLUMNS:
        if column not in summary.columns:
            summary[column] = ""
    return (
        summary[USABLE_CACHE_SUMMARY_COLUMNS]
        .sort_values(["ticker", "frequency"], kind="mergesort")
        .reset_index(drop=True)
    )


def fetch_provider_data(
    adapter: VnstockAdapter,
    ticker: str,
    frequency: str,
    start: str,
    end: str,
    *,
    config: BenchmarkConfig,
    rate_limiter: ProviderRateLimiter,
    max_fetch_retries: int,
    source_health: SourceHealthManager,
    source_preferences: SourcePreferenceCache,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    interval = "1H" if frequency == "hourly" else "1D"
    stats: dict[str, Any] = {
        "request_attempts": 0,
        "throttled_seconds": 0.0,
        "provider_error_count": 0,
        "provider_empty_count": 0,
        "errors": [],
        "invalid_reasons": [],
    }

    if not adapter._is_available():
        stats["provider_error_count"] += 1
        stats["errors"].append("vnstock_data_not_installed")
        stats["invalid_reasons"].append("provider_error")
        empty = _empty_frequency_frame(frequency)
        return empty, stats

    ticker = ticker.upper().strip()
    Quote = adapter._get_class("Quote")
    QuoteHistory = adapter._get_class("QuoteHistory")
    if Quote is None and QuoteHistory is None:
        stats["provider_error_count"] += 1
        stats["errors"].append("vnstock_quote_classes_unavailable")
        stats["invalid_reasons"].append("provider_error")
        empty = _empty_frequency_frame(frequency)
        return empty, stats

    standardized = _empty_frequency_frame(frequency)
    for retry_idx in range(1, max(int(max_fetch_retries), 1) + 1):
        for source in source_preferences.ordered_sources(ticker, frequency):
            if source_health.should_skip(source, frequency):
                stats["errors"].append(f"source_skipped_by_cooldown source={source} retry={retry_idx}")
                continue
            try:
                source_health.record_call(source, frequency)
                if Quote is not None:
                    result = provider_call(
                        rate_limiter=rate_limiter,
                        call_type="Quote.history",
                        symbol=ticker,
                        frequency=interval,
                        source=source,
                        stats=stats,
                        func=lambda source=source: Quote(source=source, symbol=ticker).history(
                            start=start,
                            end=end,
                            interval=interval,
                            get_all=True,
                        ),
                    )
                else:
                    result = provider_call(
                        rate_limiter=rate_limiter,
                        call_type="QuoteHistory.history",
                        symbol=ticker,
                        frequency=interval,
                        source=source,
                        stats=stats,
                        func=lambda source=source: QuoteHistory(source=source, symbol=ticker).history(
                            start_date=start,
                            end_date=end,
                            timeframe=interval,
                        ),
                    )
                candidate = VnstockAdapter._as_dataframe(result)
                if candidate is None or candidate.empty:
                    stats["provider_empty_count"] += 1
                    source_health.record_empty(source, frequency)
                    stats["invalid_reasons"].append("empty_response")
                    stats["errors"].append(f"empty_response source={source} retry={retry_idx}")
                    continue

                candidate_standardized = (
                    standardize_hourly(candidate, ticker, start, end)
                    if frequency == "hourly"
                    else standardize_daily(candidate, ticker, start, end)
                )
                is_valid, invalid_reason = _validate_provider_frame(
                    candidate_standardized,
                    frequency=frequency,
                    start=start,
                    end=end,
                    config=config,
                )
                if is_valid:
                    standardized = candidate_standardized
                    source_health.record_success(source, frequency)
                    source_preferences.remember(ticker, frequency, source)
                    break

                if invalid_reason == "empty_response":
                    stats["provider_empty_count"] += 1
                    source_health.record_empty(source, frequency)
                else:
                    stats["provider_error_count"] += 1
                    source_health.record_failure(source, frequency)
                stats["invalid_reasons"].append(invalid_reason)
                actual_start, actual_end = _date_bounds(candidate_standardized, frequency)
                stats["errors"].append(
                    f"{invalid_reason} source={source} retry={retry_idx} "
                    f"rows={len(candidate_standardized)} "
                    f"actual_start={_safe_iso_date(actual_start)} actual_end={_safe_iso_date(actual_end)}"
                )
            except Exception as exc:
                stats["provider_error_count"] += 1
                source_health.record_failure(source, frequency)
                stats["invalid_reasons"].append("provider_error")
                stats["errors"].append(f"source={source} retry={retry_idx}: {exc}")
        if not standardized.empty:
            break

    return standardized, stats


def load_or_fetch_ticker(
    *,
    adapter: VnstockAdapter | None,
    cache_dir: Path,
    ticker: str,
    frequency: str,
    start: str,
    end: str,
    pull_missing: bool,
    rate_limiter: ProviderRateLimiter,
    max_fetch_retries: int,
    config: BenchmarkConfig,
    source_health: SourceHealthManager,
    source_preferences: SourcePreferenceCache | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    errors: list[str] = []
    invalid_reasons: list[str] = []
    status = "failed"
    data_source = "failed"
    request_attempts = 0
    throttled_seconds = 0.0
    provider_error_count = 0
    provider_empty_count = 0
    try:
        cached = load_cache(cache_dir, frequency, ticker, start, end)
        cache_valid, cache_invalid_reason = _validate_local_cache_frame(
            cached,
            frequency=frequency,
            start=start,
            end=end,
            config=config,
        )
        cache_usability = _benchmark_usability_metrics(cached, frequency, config)
        if config.cache_only:
            if cache_valid:
                final = cached
                return final, _fetch_summary_row(
                    ticker=ticker,
                    frequency=frequency,
                    start=start,
                    end=end,
                    frame=final,
                    status="ok",
                    data_source="cache",
                    config=config,
                )
            if not cached.empty:
                invalid_detail = cache_invalid_reason or "coverage_gap"
                benchmark_usable = bool(cache_usability["benchmark_usable"])
                if config.allow_partial_cache_for_benchmark and benchmark_usable:
                    return cached, _fetch_summary_row(
                        ticker=ticker,
                        frequency=frequency,
                        start=start,
                        end=end,
                        frame=cached,
                        status="partial",
                        data_source="cache_partial_usable",
                        invalid_reasons=[invalid_detail],
                        config=config,
                    )
                errors_for_row = [f"cache_invalid:{invalid_detail}"]
                if benchmark_usable and not config.allow_partial_cache_for_benchmark:
                    errors_for_row.append("partial_cache_not_allowed")
                elif not benchmark_usable:
                    errors_for_row.append(f"benchmark_unusable:{cache_usability['benchmark_usable_reason']}")
                row = _fetch_summary_row(
                    ticker=ticker,
                    frequency=frequency,
                    start=start,
                    end=end,
                    frame=cached,
                    status="partial",
                    data_source="cache_partial",
                    request_attempts=0,
                    throttled_seconds=0.0,
                    provider_error_count=0,
                    provider_empty_count=0,
                    invalid_reasons=[invalid_detail],
                    errors=errors_for_row,
                    config=config,
                )
                return _empty_frequency_frame(frequency), row
            invalid_detail = cache_invalid_reason or "cache_missing"
            row = _fetch_summary_row(
                ticker=ticker,
                frequency=frequency,
                start=start,
                end=end,
                frame=cached,
                status="failed",
                data_source="cache_missing",
                request_attempts=0,
                throttled_seconds=0.0,
                provider_error_count=0,
                provider_empty_count=0,
                invalid_reasons=["cache_missing"],
                errors=[] if invalid_detail == "cache_missing" else [f"cache_invalid:{invalid_detail}"],
                config=config,
            )
            return _empty_frequency_frame(frequency), row

        if cache_valid:
            status = "ok"
            data_source = "cache"
            final = cached
        elif not pull_missing:
            final = cached
            status = "partial" if not cached.empty else "failed"
            data_source = "cache_partial" if not cached.empty else "cache_missing"
            invalid_reasons.append(cache_invalid_reason or ("coverage_gap" if not cached.empty else "cache_missing"))
        elif adapter is None or source_preferences is None or not config.provider_calls_allowed:
            final = cached
            status = "partial" if not cached.empty else "failed"
            data_source = "cache_partial" if not cached.empty else "cache_missing"
            invalid_reasons.append(cache_invalid_reason or "cache_missing")
        else:
            fetched_parts = []
            missing_ranges = _missing_ranges(cached, frequency, start, end, config)
            if not missing_ranges and not cache_valid:
                missing_ranges = [(start, end)]
            for range_start, range_end in missing_ranges:
                try:
                    fetched, fetch_stats = fetch_provider_data(
                        adapter,
                        ticker,
                        frequency,
                        range_start,
                        range_end,
                        config=config,
                        rate_limiter=rate_limiter,
                        max_fetch_retries=max_fetch_retries,
                        source_health=source_health,
                        source_preferences=source_preferences,
                    )
                    request_attempts += int(fetch_stats.get("request_attempts", 0))
                    throttled_seconds += float(fetch_stats.get("throttled_seconds", 0.0))
                    provider_error_count += int(fetch_stats.get("provider_error_count", 0))
                    provider_empty_count += int(fetch_stats.get("provider_empty_count", 0))
                    errors.extend(str(error) for error in fetch_stats.get("errors", []))
                    invalid_reasons.extend(str(reason) for reason in fetch_stats.get("invalid_reasons", []))
                    fetched_parts.append(fetched)
                except Exception as exc:
                    provider_error_count += 1
                    invalid_reasons.append("provider_error")
                    errors.append(f"provider_fetch_{range_start}_{range_end}: {exc}")
            pieces = [cached, *[part for part in fetched_parts if part is not None and not part.empty]]
            if pieces:
                merged = pd.concat(pieces, ignore_index=True)
                final = standardize_hourly(merged, ticker, start, end) if frequency == "hourly" else standardize_daily(merged, ticker, start, end)
            else:
                final = _empty_frequency_frame(frequency)
            if not final.empty:
                path = _cache_path(cache_dir, frequency, ticker)
                path.parent.mkdir(parents=True, exist_ok=True)
                final.to_csv(path, index=False)
            final_valid, final_invalid_reason = _validate_local_cache_frame(
                final,
                frequency=frequency,
                start=start,
                end=end,
                config=config,
            )
            if final_valid:
                status = "ok"
            elif final.empty:
                status = "failed"
                if not invalid_reasons:
                    invalid_reasons.append(final_invalid_reason or "provider_error")
            else:
                status = "partial"
                invalid_reasons.append(final_invalid_reason or "coverage_gap")
            has_cache = not cached.empty
            has_provider = any(part is not None and not part.empty for part in fetched_parts)
            if final.empty:
                data_source = "failed"
            elif has_cache and has_provider:
                data_source = "mixed"
            elif has_provider:
                data_source = "provider"
            elif has_cache:
                data_source = "cache_partial"
            else:
                data_source = "failed"
    except Exception as exc:
        final = _empty_frequency_frame(frequency)
        status = "failed"
        data_source = "cache_missing" if config.cache_only else "failed"
        invalid_reasons.append("cache_missing" if config.cache_only else "provider_error")
        errors.append(str(exc))

    row = _fetch_summary_row(
        ticker=ticker,
        frequency=frequency,
        start=start,
        end=end,
        frame=final,
        status=status,
        data_source=data_source,
        request_attempts=request_attempts,
        throttled_seconds=throttled_seconds,
        provider_error_count=provider_error_count,
        provider_empty_count=provider_empty_count,
        invalid_reasons=invalid_reasons,
        errors=errors,
        config=config,
    )
    return final, row


class FetchCheckpointWriter:
    def __init__(self, output_dir: Path, config: BenchmarkConfig) -> None:
        self.output_dir = output_dir
        self.config = config
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.output_dir / "run_config.json", _run_config_payload(config))

    def write(self, fetch_rows: list[dict[str, Any]], source_health_summary: pd.DataFrame) -> None:
        fetch_summary = pd.DataFrame(fetch_rows, columns=FETCH_SUMMARY_COLUMNS)
        fetch_summary.to_csv(
            self.output_dir / "fetch_summary.csv",
            index=False,
        )
        build_usable_cache_summary(fetch_summary).to_csv(self.output_dir / "usable_cache_summary.csv", index=False)
        source_health_summary.to_csv(self.output_dir / "source_health_summary.csv", index=False)


def load_market_data(
    *,
    config: BenchmarkConfig,
    tickers: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    if config.provider != "vnstock_data":
        raise ValueError("Only provider='vnstock_data' is supported by this benchmark")
    cache_dir = Path(config.cache_dir)
    daily_frames: dict[str, pd.DataFrame] = {}
    hourly_frames: dict[str, pd.DataFrame] = {}
    fetch_rows: list[dict[str, Any]] = []
    checkpoint_writer = (
        FetchCheckpointWriter(Path(config.output_dir), config)
        if config.checkpointing_enabled
        else None
    )
    rate_limiter = ProviderRateLimiter(
        rate_limit_per_minute=config.rate_limit_per_minute,
        request_sleep_seconds=config.request_sleep_seconds,
    )
    adapter = (
        VnstockAdapter(symbol_list=tickers, rate_limiter=rate_limiter)
        if config.provider_calls_allowed
        else None
    )
    source_health = SourceHealthManager(
        failure_threshold=config.source_failure_threshold,
        empty_threshold=config.source_empty_threshold,
        cooldown_seconds=config.source_cooldown_seconds,
    )
    source_preferences = SourcePreferenceCache(cache_dir / "source_preference.json") if config.provider_calls_allowed else None
    batch_provider_attempts = 0

    for idx, ticker in enumerate(tickers, start=1):
        for frequency, start, end, frame_store in (
            ("daily", config.daily_start, config.daily_end, daily_frames),
            ("hourly", config.hourly_start, config.hourly_end, hourly_frames),
        ):
            frame, row = load_or_fetch_ticker(
                adapter=adapter,
                cache_dir=cache_dir,
                ticker=ticker,
                frequency=frequency,
                start=start,
                end=end,
                pull_missing=config.pull_missing,
                rate_limiter=rate_limiter,
                max_fetch_retries=config.max_fetch_retries,
                config=config,
                source_health=source_health,
                source_preferences=source_preferences,
            )
            fetch_rows.append(row)
            batch_provider_attempts += int(row.get("request_attempts", 0))
            if not frame.empty:
                frame_store[ticker] = frame
            if checkpoint_writer is not None:
                checkpoint_writer.write(fetch_rows, source_health.to_frame())

        if (
            config.pull_missing
            and idx < len(tickers)
            and idx % int(config.fetch_batch_size) == 0
            and batch_provider_attempts > 0
            and config.fetch_batch_cooldown_seconds > 0.0
        ):
            time.sleep(float(config.fetch_batch_cooldown_seconds))
            batch_provider_attempts = 0

    fetch_summary = pd.DataFrame(fetch_rows, columns=FETCH_SUMMARY_COLUMNS)
    if source_preferences is not None:
        source_preferences.save()
    return daily_frames, hourly_frames, fetch_summary, source_health.to_frame()


def resample_hourly_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    if hourly.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    prepared = hourly.copy()
    prepared["datetime"] = pd.to_datetime(prepared["datetime"], errors="coerce")
    prepared = prepared.dropna(subset=["datetime"]).sort_values(["ticker", "datetime"])
    prepared["date"] = prepared["datetime"].dt.normalize()
    rows = []
    for (ticker, date), group in prepared.groupby(["ticker", "date"], sort=True):
        ordered = group.sort_values("datetime")
        rows.append(
            {
                "date": date,
                "ticker": ticker,
                "open": float(ordered["open"].iloc[0]),
                "high": float(ordered["high"].max()),
                "low": float(ordered["low"].min()),
                "close": float(ordered["close"].iloc[-1]),
                "volume": float(ordered["volume"].sum()),
            }
        )
    if not rows:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    return standardize_daily(pd.DataFrame(rows), "__KEEP__", str(prepared["date"].min().date()), str(prepared["date"].max().date()))


def combine_daily_inputs(
    daily_frames: dict[str, pd.DataFrame],
    hourly_frames: dict[str, pd.DataFrame],
    config: BenchmarkConfig,
) -> pd.DataFrame:
    daily_parts = []
    if daily_frames:
        daily_parts.append(pd.concat(daily_frames.values(), ignore_index=True))
    if hourly_frames:
        hourly_all = pd.concat(hourly_frames.values(), ignore_index=True)
        daily_parts.append(resample_hourly_to_daily(hourly_all))
    if not daily_parts:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    combined = pd.concat(daily_parts, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.normalize()
    combined = combined[
        (combined["date"] >= _to_date(config.daily_start))
        & (combined["date"] <= _to_date(config.hourly_end))
    ].copy()
    for column in OHLCV_COLUMNS:
        combined[column] = _finite_numeric(combined[column])
    combined = combined.dropna(subset=["date", "ticker", *OHLCV_COLUMNS])
    combined = combined[(combined["close"] > 0.0) & (combined["volume"] >= 0.0)].copy()
    combined["ticker"] = combined["ticker"].astype(str).str.upper().str.strip()
    return combined.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")[DAILY_COLUMNS].reset_index(drop=True)


def combine_hourly_inputs(hourly_frames: dict[str, pd.DataFrame], config: BenchmarkConfig) -> pd.DataFrame:
    if not hourly_frames:
        return pd.DataFrame(columns=HOURLY_COLUMNS)
    combined = pd.concat(hourly_frames.values(), ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["datetime"], errors="coerce")
    combined = combined[
        (combined["datetime"] >= _to_date(config.hourly_start))
        & (combined["datetime"] <= _to_end_of_day(config.hourly_end))
    ].copy()
    return standardize_hourly(combined, "__KEEP__", config.hourly_start, config.hourly_end)


def assign_continuity_segments(
    raw: pd.DataFrame,
    *,
    frequency: str,
    time_column: str,
    max_daily_gap_days: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if raw.empty:
        return raw.copy(), []
    prepared = raw.copy()
    prepared[time_column] = pd.to_datetime(prepared[time_column], errors="coerce")
    prepared = prepared.dropna(subset=[time_column]).sort_values(["ticker", time_column]).reset_index(drop=True)
    if frequency != "daily":
        prepared["segment_id"] = 0
        return prepared, []

    warnings: list[dict[str, Any]] = []
    parts: list[pd.DataFrame] = []
    for ticker, group in prepared.groupby("ticker", sort=True):
        ordered = group.sort_values(time_column).reset_index(drop=True)
        gaps = ordered[time_column].diff().dt.days
        breaks = gaps > int(max_daily_gap_days)
        ordered["segment_id"] = breaks.fillna(False).astype(int).cumsum().astype(int)
        for gap_idx in ordered.index[breaks.fillna(False)]:
            previous_timestamp = ordered.loc[gap_idx - 1, time_column] if gap_idx > 0 else pd.NaT
            current_timestamp = ordered.loc[gap_idx, time_column]
            warnings.append(
                {
                    "ticker": str(ticker).upper(),
                    "frequency": frequency,
                    "previous_timestamp": pd.Timestamp(previous_timestamp).isoformat()
                    if pd.notna(previous_timestamp)
                    else "",
                    "current_timestamp": pd.Timestamp(current_timestamp).isoformat(),
                    "gap_days": int(gaps.loc[gap_idx]),
                    "max_allowed_gap_days": int(max_daily_gap_days),
                    "action": "split_segment_prevent_rolling_and_label_crossing",
                }
            )
        parts.append(ordered)

    if not parts:
        prepared["segment_id"] = 0
        return prepared, warnings
    return pd.concat(parts, ignore_index=True).sort_values(["ticker", time_column]).reset_index(drop=True), warnings


def build_feature_frame(raw: pd.DataFrame, frequency: str) -> tuple[pd.DataFrame, list[str]]:
    if "segment_id" not in raw.columns:
        return _build_feature_frame_single_segment(raw, frequency)

    frames: list[pd.DataFrame] = []
    feature_columns: list[str] = []
    for _, segment in raw.groupby("segment_id", sort=True):
        segment_frame, segment_features = _build_feature_frame_single_segment(segment, frequency)
        if segment_features:
            feature_columns = segment_features
        if not segment_frame.empty:
            frames.append(segment_frame)
    if not frames:
        return pd.DataFrame(), feature_columns
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("timestamp").reset_index(drop=True), feature_columns


def _build_feature_frame_single_segment(raw: pd.DataFrame, frequency: str) -> tuple[pd.DataFrame, list[str]]:
    time_column = "datetime" if frequency == "hourly" else "date"
    if raw.empty:
        return pd.DataFrame(), []
    prepared = raw.copy().sort_values(time_column).reset_index(drop=True)
    prepared["timestamp"] = pd.to_datetime(prepared[time_column], errors="coerce")
    for column in OHLCV_COLUMNS:
        prepared[column] = _finite_numeric(prepared[column])
    prepared = prepared.dropna(subset=["timestamp", *OHLCV_COLUMNS])
    close = prepared["close"].astype(float)
    volume = prepared["volume"].astype(float)

    feature_columns: list[str] = []

    prepared["return_1"] = close.pct_change(periods=1, fill_method=None)
    feature_columns.append("return_1")
    for lag in (2, 3, 5, 10, 20):
        column = f"return_{lag}"
        prepared[column] = close.pct_change(periods=lag, fill_method=None)
        feature_columns.append(column)
    for lag in (1, 2, 3, 5, 10, 20):
        column = f"return_1_lag_{lag}"
        prepared[column] = prepared["return_1"].shift(lag)
        feature_columns.append(column)
    for window in (5, 10, 20, 60):
        min_periods = max(3, min(window, window // 2))
        mean_col = f"rolling_return_mean_{window}"
        vol_col = f"rolling_return_vol_{window}"
        sma_col = f"close_sma_ratio_{window}"
        momentum_col = f"momentum_{window}"
        prepared[mean_col] = prepared["return_1"].rolling(window, min_periods=min_periods).mean()
        prepared[vol_col] = prepared["return_1"].rolling(window, min_periods=min_periods).std()
        sma = close.rolling(window, min_periods=min_periods).mean()
        prepared[sma_col] = close / sma - 1.0
        prepared[momentum_col] = close / close.shift(window) - 1.0
        feature_columns.extend([mean_col, vol_col, sma_col, momentum_col])

    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(14, min_periods=7).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=7).mean()
    rs = gain / loss.replace(0.0, np.nan)
    prepared["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    prepared.loc[(loss == 0.0) & (gain > 0.0), "rsi_14"] = 100.0
    prepared.loc[(loss == 0.0) & (gain == 0.0), "rsi_14"] = 50.0
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    prepared["macd"] = ema_12 - ema_26
    prepared["macd_signal"] = prepared["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    prepared["macd_hist"] = prepared["macd"] - prepared["macd_signal"]
    feature_columns.extend(["rsi_14", "macd", "macd_signal", "macd_hist"])

    prepared["volume_change_1"] = volume.pct_change(periods=1, fill_method=None)
    volume_ma_20 = volume.rolling(20, min_periods=5).mean()
    prepared["volume_shock_20"] = volume / volume_ma_20 - 1.0
    prepared["high_low_range"] = (prepared["high"] - prepared["low"]) / close
    prepared["open_close_spread"] = (close - prepared["open"]) / prepared["open"].replace(0.0, np.nan)
    prepared["close_position_in_range"] = (close - prepared["low"]) / (prepared["high"] - prepared["low"]).replace(0.0, np.nan)
    feature_columns.extend(["volume_change_1", "volume_shock_20", "high_low_range", "open_close_spread", "close_position_in_range"])

    prepared["day_of_week"] = prepared["timestamp"].dt.dayofweek.astype(float)
    prepared["day_of_month"] = prepared["timestamp"].dt.day.astype(float)
    prepared["month"] = prepared["timestamp"].dt.month.astype(float)
    prepared["quarter"] = prepared["timestamp"].dt.quarter.astype(float)
    feature_columns.extend(["day_of_week", "day_of_month", "month", "quarter"])
    if frequency == "hourly":
        prepared["hour"] = prepared["timestamp"].dt.hour.astype(float)
        prepared["minute"] = prepared["timestamp"].dt.minute.astype(float)
        feature_columns.extend(["hour", "minute"])

    prepared[feature_columns] = prepared[feature_columns].replace([np.inf, -np.inf], np.nan)
    return prepared.reset_index(drop=True), feature_columns


def add_horizon_labels(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if "segment_id" in frame.columns:
        frames = [_add_horizon_labels_single(segment, horizon) for _, segment in frame.groupby("segment_id", sort=True)]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return _add_horizon_labels_single(frame, horizon)


def _add_horizon_labels_single(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["actual_close"] = prepared["close"].shift(-int(horizon))
    prepared["target_timestamp"] = prepared["timestamp"].shift(-int(horizon))
    prepared["actual_return"] = prepared["actual_close"] / prepared["close"] - 1.0
    prepared["actual_direction"] = np.where(prepared["actual_return"] > 0.0, 1, 0)
    return prepared


def _merge_model_params(defaults: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    params = dict(defaults)
    if overrides:
        params.update(overrides)
    return params


def make_regressor(model_name: str, seed: int, params: dict[str, Any] | None = None) -> Any:
    name = model_name.lower()
    if name == "random_forest":
        return RandomForestRegressor(
            **_merge_model_params(
                {
                    "n_estimators": 160,
                    "max_depth": 8,
                    "min_samples_leaf": 5,
                    "random_state": seed,
                    "n_jobs": 1,
                },
                params,
            )
        )
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("xgboost is not installed") from exc
        return XGBRegressor(
            **_merge_model_params(
                {
                    "n_estimators": 140,
                    "max_depth": 3,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 1.0,
                    "objective": "reg:squarederror",
                    "random_state": seed,
                    "n_jobs": 1,
                    "tree_method": "hist",
                    "verbosity": 0,
                },
                params,
            )
        )
    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("lightgbm is not installed") from exc
        return LGBMRegressor(
            **_merge_model_params(
                {
                    "n_estimators": 140,
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "num_leaves": 31,
                    "random_state": seed,
                    "n_jobs": 1,
                    "verbosity": -1,
                    "deterministic": True,
                    "force_col_wise": True,
                },
                params,
            )
        )
    raise ValueError(
        f"Unsupported model '{model_name}' for VN100 hybrid-frequency benchmark. "
        f"Supported benchmark models: {list(BENCHMARK_SUPPORTED_MODELS)}."
    )


def make_classifier(model_name: str, seed: int, params: dict[str, Any] | None = None) -> Any:
    name = model_name.lower()
    if name == "random_forest":
        return RandomForestClassifier(
            **_merge_model_params(
                {
                    "n_estimators": 160,
                    "max_depth": 8,
                    "min_samples_leaf": 5,
                    "random_state": seed,
                    "n_jobs": 1,
                },
                params,
            )
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is not installed") from exc
        return XGBClassifier(
            **_merge_model_params(
                {
                    "n_estimators": 140,
                    "max_depth": 3,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 1.0,
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "random_state": seed,
                    "n_jobs": 1,
                    "tree_method": "hist",
                    "verbosity": 0,
                },
                params,
            )
        )
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("lightgbm is not installed") from exc
        return LGBMClassifier(
            **_merge_model_params(
                {
                    "n_estimators": 140,
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "num_leaves": 31,
                    "random_state": seed,
                    "n_jobs": 1,
                    "verbosity": -1,
                    "deterministic": True,
                    "force_col_wise": True,
                },
                params,
            )
        )
    raise ValueError(
        f"Unsupported classifier '{model_name}' for VN100 hybrid-frequency benchmark. "
        f"Supported benchmark models: {list(BENCHMARK_SUPPORTED_MODELS)}."
    )


def _prepare_xy(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    *,
    fill_values: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    x_raw = frame[feature_columns].apply(_finite_numeric)
    y = _finite_numeric(frame[target_column])
    mask = y.notna()
    if fill_values is None:
        fill_values = x_raw.loc[mask].median(axis=0, skipna=True).fillna(0.0).astype(float)
    x = x_raw.fillna(fill_values).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x, y, mask, fill_values


def fit_predict_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    seed: int,
    model_params: dict[str, Any] | None = None,
) -> np.ndarray:
    # TODO: Enforce model_fit_timeout_seconds with a Windows-safe process or
    # cooperative model wrapper. The option is metadata-only for now.
    model = make_regressor(model_name, seed, model_params)
    train_x, train_y, train_mask, fill_values = _prepare_xy(train_df, feature_columns, target_column)
    if int(train_mask.sum()) < 60:
        raise ValueError(f"insufficient training rows for {model_name}: {int(train_mask.sum())}")
    model.fit(train_x.loc[train_mask], train_y.loc[train_mask])
    test_x, _, _, _ = _prepare_xy(test_df, feature_columns, target_column, fill_values=fill_values)
    return np.asarray(model.predict(test_x), dtype=float)


def _prepare_classifier_xy(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    fill_values: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    x_raw = frame[feature_columns].apply(_finite_numeric)
    actual_return = _finite_numeric(frame["actual_return"])
    y = (actual_return > 0.0).astype(int)
    mask = actual_return.notna() & (actual_return != 0.0)
    if fill_values is None:
        fill_values = x_raw.loc[mask].median(axis=0, skipna=True).fillna(0.0).astype(float)
    x = x_raw.fillna(fill_values).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x, y, mask, fill_values


def _classifier_probability_up(model: Any, test_x: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"{model.__class__.__name__} does not support predict_proba")
    probabilities = np.asarray(model.predict_proba(test_x), dtype=float)
    classes = getattr(model, "classes_", np.array([0, 1]))
    if probabilities.ndim != 2 or probabilities.shape[1] == 0:
        raise ValueError("predict_proba returned no class probabilities")
    class_values = [int(value) for value in classes]
    if 1 in class_values:
        up_idx = class_values.index(1)
    else:
        up_idx = min(probabilities.shape[1] - 1, 1)
    return probabilities[:, up_idx]


def fit_predict_classifier(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
    model_params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model = make_classifier(model_name, seed, model_params)
    train_x, train_y, train_mask, fill_values = _prepare_classifier_xy(train_df, feature_columns)
    if int(train_mask.sum()) < 60:
        raise ValueError(f"insufficient classification training rows for {model_name}: {int(train_mask.sum())}")
    clean_y = train_y.loc[train_mask].astype(int)
    if clean_y.nunique(dropna=True) < 2:
        raise ValueError(f"classification target has one class for {model_name}")
    model.fit(train_x.loc[train_mask], clean_y)
    test_x, _, _, _ = _prepare_classifier_xy(test_df, feature_columns, fill_values=fill_values)
    probability_up = _classifier_probability_up(model, test_x)
    predicted_direction = (probability_up >= 0.5).astype(int)
    confidence = np.maximum(probability_up, 1.0 - probability_up)
    return predicted_direction, confidence


def _fit_model_on_frame(
    model_name: str,
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    seed: int,
    model_params: dict[str, Any] | None = None,
) -> tuple[Any, pd.Series]:
    model = make_regressor(model_name, seed, model_params)
    train_x, train_y, train_mask, fill_values = _prepare_xy(train_df, feature_columns, target_column)
    if int(train_mask.sum()) < 30:
        raise ValueError(f"insufficient training rows for {model_name}: {int(train_mask.sum())}")
    model.fit(train_x.loc[train_mask], train_y.loc[train_mask])
    return model, fill_values


def _predict_with_fitted(model: Any, frame: pd.DataFrame, feature_columns: list[str], target_column: str, fill_values: pd.Series) -> np.ndarray:
    x, _, _, _ = _prepare_xy(frame, feature_columns, target_column, fill_values=fill_values)
    return np.asarray(model.predict(x), dtype=float)


def fit_predict_stacking(
    base_model_names: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    seed: int,
    model_params_by_model: dict[str, dict[str, Any]] | None = None,
) -> np.ndarray:
    available_base_models: list[str] = []
    for base_name in base_model_names:
        try:
            make_regressor(base_name, seed, (model_params_by_model or {}).get(base_name))
            available_base_models.append(base_name)
        except Exception:
            continue
    base_model_names = available_base_models
    if len(base_model_names) < 2:
        raise ValueError("stacking requires at least two available base models")
    train_y = _finite_numeric(train_df[target_column])
    train_mask = train_y.notna()
    clean_train = train_df.loc[train_mask].reset_index(drop=True)
    clean_y = train_y.loc[train_mask].reset_index(drop=True)
    if len(clean_train) < 90:
        raise ValueError(f"insufficient training rows for stacking: {len(clean_train)}")

    oof = np.full((len(clean_train), len(base_model_names)), np.nan, dtype=float)
    n_splits = min(3, max(2, len(clean_train) // 120))
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for fold_id, (fit_idx, valid_idx) in enumerate(splitter.split(clean_train)):
        fold_train = clean_train.iloc[fit_idx].copy()
        fold_valid = clean_train.iloc[valid_idx].copy()
        if len(fold_train) < 30:
            continue
        for col_idx, base_name in enumerate(base_model_names):
            try:
                model, fill_values = _fit_model_on_frame(
                    base_name,
                    fold_train,
                    feature_columns,
                    target_column,
                    seed + fold_id + col_idx,
                    (model_params_by_model or {}).get(base_name),
                )
                oof[valid_idx, col_idx] = _predict_with_fitted(model, fold_valid, feature_columns, target_column, fill_values)
            except Exception:
                continue

    meta_mask = np.isfinite(oof).all(axis=1) & clean_y.notna().to_numpy()
    if int(meta_mask.sum()) < 30:
        full_train_predictions = []
        for col_idx, base_name in enumerate(base_model_names):
            model, fill_values = _fit_model_on_frame(
                base_name,
                clean_train,
                feature_columns,
                target_column,
                seed + col_idx,
                (model_params_by_model or {}).get(base_name),
            )
            full_train_predictions.append(_predict_with_fitted(model, clean_train, feature_columns, target_column, fill_values))
        oof = np.column_stack(full_train_predictions)
        meta_mask = np.isfinite(oof).all(axis=1) & clean_y.notna().to_numpy()
    if int(meta_mask.sum()) < 30:
        raise ValueError("stacking meta learner could not build enough finite meta samples")

    meta_model = Ridge(alpha=1.0)
    meta_model.fit(oof[meta_mask], clean_y.loc[meta_mask].to_numpy(dtype=float))

    test_predictions = []
    for col_idx, base_name in enumerate(base_model_names):
        model, fill_values = _fit_model_on_frame(
            base_name,
            clean_train,
            feature_columns,
            target_column,
            seed + 100 + col_idx,
            (model_params_by_model or {}).get(base_name),
        )
        test_predictions.append(_predict_with_fitted(model, test_df, feature_columns, target_column, fill_values))
    meta_test = np.column_stack(test_predictions)
    return np.asarray(meta_model.predict(meta_test), dtype=float)


def _fit_classifier_on_frame(
    model_name: str,
    train_df: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
    model_params: dict[str, Any] | None = None,
) -> tuple[Any, pd.Series]:
    model = make_classifier(model_name, seed, model_params)
    train_x, train_y, train_mask, fill_values = _prepare_classifier_xy(train_df, feature_columns)
    if int(train_mask.sum()) < 30:
        raise ValueError(f"insufficient classification training rows for {model_name}: {int(train_mask.sum())}")
    clean_y = train_y.loc[train_mask].astype(int)
    if clean_y.nunique(dropna=True) < 2:
        raise ValueError(f"classification target has one class for {model_name}")
    model.fit(train_x.loc[train_mask], clean_y)
    return model, fill_values


def _predict_classifier_probability_up(
    model: Any,
    frame: pd.DataFrame,
    feature_columns: list[str],
    fill_values: pd.Series,
) -> np.ndarray:
    x, _, _, _ = _prepare_classifier_xy(frame, feature_columns, fill_values=fill_values)
    return _classifier_probability_up(model, x)


def fit_predict_stacking_classifier(
    base_model_names: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
    model_params_by_model: dict[str, dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    available_base_models: list[str] = []
    for base_name in base_model_names:
        try:
            make_classifier(base_name, seed, (model_params_by_model or {}).get(base_name))
            available_base_models.append(base_name)
        except Exception:
            continue
    base_model_names = available_base_models
    if len(base_model_names) < 2:
        raise ValueError("classification stacking requires at least two available base models")

    _, train_y, train_mask, _ = _prepare_classifier_xy(train_df, feature_columns)
    clean_train = train_df.loc[train_mask].reset_index(drop=True)
    clean_y = train_y.loc[train_mask].astype(int).reset_index(drop=True)
    if len(clean_train) < 90:
        raise ValueError(f"insufficient classification training rows for stacking: {len(clean_train)}")
    if clean_y.nunique(dropna=True) < 2:
        raise ValueError("classification stacking target has one class")

    oof = np.full((len(clean_train), len(base_model_names)), np.nan, dtype=float)
    n_splits = min(3, max(2, len(clean_train) // 120))
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for fold_id, (fit_idx, valid_idx) in enumerate(splitter.split(clean_train)):
        fold_train = clean_train.iloc[fit_idx].copy()
        fold_valid = clean_train.iloc[valid_idx].copy()
        if len(fold_train) < 30:
            continue
        for col_idx, base_name in enumerate(base_model_names):
            try:
                model, fill_values = _fit_classifier_on_frame(
                    base_name,
                    fold_train,
                    feature_columns,
                    seed + fold_id + col_idx,
                    (model_params_by_model or {}).get(base_name),
                )
                oof[valid_idx, col_idx] = _predict_classifier_probability_up(model, fold_valid, feature_columns, fill_values)
            except Exception:
                continue

    meta_mask = np.isfinite(oof).all(axis=1)
    if int(meta_mask.sum()) < 30:
        full_train_probabilities = []
        for col_idx, base_name in enumerate(base_model_names):
            model, fill_values = _fit_classifier_on_frame(
                base_name,
                clean_train,
                feature_columns,
                seed + col_idx,
                (model_params_by_model or {}).get(base_name),
            )
            full_train_probabilities.append(_predict_classifier_probability_up(model, clean_train, feature_columns, fill_values))
        oof = np.column_stack(full_train_probabilities)
        meta_mask = np.isfinite(oof).all(axis=1)
    if int(meta_mask.sum()) < 30:
        raise ValueError("classification stacking meta learner could not build enough finite meta samples")

    meta_model = LogisticRegression(max_iter=1000, random_state=seed)
    meta_model.fit(oof[meta_mask], clean_y.loc[meta_mask].to_numpy(dtype=int))

    test_probabilities = []
    for col_idx, base_name in enumerate(base_model_names):
        model, fill_values = _fit_classifier_on_frame(
            base_name,
            clean_train,
            feature_columns,
            seed + 100 + col_idx,
            (model_params_by_model or {}).get(base_name),
        )
        test_probabilities.append(_predict_classifier_probability_up(model, test_df, feature_columns, fill_values))
    meta_test = np.column_stack(test_probabilities)
    probability_up = _classifier_probability_up(meta_model, pd.DataFrame(meta_test))
    predicted_direction = (probability_up >= 0.5).astype(int)
    confidence = np.maximum(probability_up, 1.0 - probability_up)
    return predicted_direction, confidence


def _model_execution_summary(
    *,
    requested_models: list[str],
    predictions: pd.DataFrame,
    skipped: list[dict[str, Any]],
) -> dict[str, Any]:
    requested = list(dict.fromkeys(str(model).strip().lower() for model in requested_models if str(model).strip()))
    executed = (
        sorted(predictions["model"].dropna().astype(str).str.lower().unique().tolist())
        if not predictions.empty and "model" in predictions.columns
        else []
    )
    executed_set = set(executed)
    skipped_models = [model for model in requested if model not in executed_set]
    skipped_reasons: dict[str, list[str]] = {model: [] for model in skipped_models}

    for item in skipped:
        model = item.get("model")
        if model is None:
            continue
        model_name = str(model).strip().lower()
        if model_name not in skipped_reasons:
            continue
        reason = str(item.get("reason") or "no_predictions_generated")
        if reason not in skipped_reasons[model_name]:
            skipped_reasons[model_name].append(reason)

    for model in skipped_models:
        if not skipped_reasons[model]:
            skipped_reasons[model].append("no_predictions_generated")

    return {
        "available_models": list(BENCHMARK_SUPPORTED_MODELS),
        "requested_models": requested,
        "executed_models": executed,
        "skipped_models": skipped_models,
        "skipped_model_reasons": skipped_reasons,
    }


def _retrain_group_key(timestamps: pd.Series, retrain_frequency: str) -> pd.Series:
    ts = pd.to_datetime(timestamps, errors="coerce")
    if retrain_frequency == "daily":
        return ts.dt.strftime("%Y-%m-%d")
    if retrain_frequency == "weekly":
        return ts.dt.to_period("W").astype(str)
    if retrain_frequency == "quarterly":
        return ts.dt.to_period("Q").astype(str)
    if retrain_frequency == "never":
        return pd.Series("all", index=timestamps.index)
    return ts.dt.to_period("M").astype(str)


def _valid_prediction_rows(test_df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    result = test_df.copy()
    result["predicted_return"] = pd.Series(predictions, index=result.index)
    result["actual_return"] = _finite_numeric(result["actual_return"])
    result["predicted_return"] = _finite_numeric(result["predicted_return"])
    result["actual_close"] = _finite_numeric(result["actual_close"])
    result["close"] = _finite_numeric(result["close"])
    mask = (
        result["actual_return"].notna()
        & result["predicted_return"].notna()
        & result["actual_close"].notna()
        & result["close"].notna()
        & (result["actual_return"] != 0.0)
    )
    return result.loc[mask].copy()


def _valid_classification_prediction_rows(
    test_df: pd.DataFrame,
    predicted_direction: np.ndarray,
    confidence: np.ndarray,
) -> pd.DataFrame:
    result = test_df.copy()
    result["predicted_direction"] = pd.Series(predicted_direction, index=result.index)
    result["confidence"] = pd.Series(confidence, index=result.index)
    result["actual_return"] = _finite_numeric(result["actual_return"])
    result["actual_close"] = _finite_numeric(result["actual_close"])
    result["close"] = _finite_numeric(result["close"])
    result["confidence"] = _finite_numeric(result["confidence"])
    result["predicted_direction"] = pd.to_numeric(result["predicted_direction"], errors="coerce")
    mask = (
        result["actual_return"].notna()
        & result["actual_close"].notna()
        & result["close"].notna()
        & result["predicted_direction"].isin([0, 1])
        & (result["actual_return"] != 0.0)
    )
    result = result.loc[mask].copy()
    result["predicted_direction"] = result["predicted_direction"].astype(int)
    probability_edge = result["confidence"].where(result["predicted_direction"].eq(1), 1.0 - result["confidence"]) - 0.5
    result["predicted_return"] = probability_edge.fillna(0.0)
    return result


def add_regime_labels(feature_frame: pd.DataFrame, config: BenchmarkConfig) -> pd.DataFrame:
    if feature_frame.empty:
        return feature_frame.copy()
    prepared = feature_frame.copy().sort_values("timestamp").reset_index(drop=True)
    close = _finite_numeric(prepared["close"])
    return_window = int(config.regime_return_window)
    vol_window = int(config.regime_vol_window)
    returns = _finite_numeric(prepared.get("return_1", close.pct_change(periods=1, fill_method=None)))
    rolling_return = close / close.shift(return_window) - 1.0
    rolling_volatility = returns.rolling(vol_window, min_periods=max(3, min(vol_window, vol_window // 2))).std()
    vol_quantile_window = max(vol_window * 5, vol_window + 1)
    high_vol_threshold = rolling_volatility.rolling(
        vol_quantile_window,
        min_periods=vol_window,
    ).quantile(float(config.regime_vol_quantile))
    low_vol_threshold = rolling_volatility.rolling(
        vol_quantile_window,
        min_periods=vol_window,
    ).quantile(1.0 - float(config.regime_vol_quantile))

    trend_regime = pd.Series("unknown", index=prepared.index, dtype=object)
    trend_regime.loc[rolling_return > float(config.regime_bull_threshold)] = "bull"
    trend_regime.loc[rolling_return < float(config.regime_bear_threshold)] = "bear"
    sideways_mask = rolling_return.notna() & trend_regime.eq("unknown")
    trend_regime.loc[sideways_mask] = "sideways"

    volatility_regime = pd.Series("unknown", index=prepared.index, dtype=object)
    volatility_regime.loc[rolling_volatility.notna() & high_vol_threshold.notna() & (rolling_volatility >= high_vol_threshold)] = "high_volatility"
    volatility_regime.loc[rolling_volatility.notna() & low_vol_threshold.notna() & (rolling_volatility <= low_vol_threshold)] = "low_volatility"
    prepared["regime"] = trend_regime
    prepared["volatility_regime"] = volatility_regime
    return prepared


def _filter_prediction_flag(
    *,
    enable_confidence_filter: bool,
    predicted_return: float,
    confidence: float | None,
    confidence_threshold: float,
    no_trade_band: float,
) -> bool:
    if not enable_confidence_filter:
        return False
    if abs(float(predicted_return)) <= float(no_trade_band):
        return True
    if confidence is None or not math.isfinite(float(confidence)):
        return True
    return bool(float(confidence) < float(confidence_threshold))


def _prediction_records(
    rows: pd.DataFrame,
    *,
    ticker: str,
    frequency: str,
    horizon: int,
    model_name: str,
    target_mode: str,
    enable_confidence_filter: bool,
    confidence_threshold: float,
    no_trade_band: float,
) -> list[dict[str, Any]]:
    records = []
    for row in rows.itertuples(index=False):
        actual_return = float(row.actual_return)
        predicted_return = float(row.predicted_return)
        actual_direction = 1 if actual_return > 0.0 else 0
        predicted_direction = int(getattr(row, "predicted_direction", 1 if predicted_return > 0.0 else 0))
        confidence_value = getattr(row, "confidence", np.nan)
        confidence = _finite_float_or_none(confidence_value)
        filtered_out = _filter_prediction_flag(
            enable_confidence_filter=enable_confidence_filter,
            predicted_return=predicted_return,
            confidence=confidence,
            confidence_threshold=confidence_threshold,
            no_trade_band=no_trade_band,
        )
        predicted_close = (
            float(row.close) * (1.0 + predicted_return)
            if str(target_mode) == "regression"
            else float("nan")
        )
        records.append(
            {
                "timestamp": pd.Timestamp(row.timestamp).isoformat(),
                "date": str(pd.Timestamp(row.timestamp).date()),
                "ticker": ticker,
                "frequency": frequency,
                "horizon": int(horizon),
                "model": model_name,
                "target_mode": target_mode,
                "actual_close": float(row.actual_close),
                "predicted_close": predicted_close,
                "actual_return": actual_return,
                "predicted_return": predicted_return,
                "actual_direction": actual_direction,
                "predicted_direction": predicted_direction,
                "confidence": confidence,
                "filtered_out": bool(filtered_out),
                "regime": str(getattr(row, "regime", "unknown") or "unknown"),
                "volatility_regime": str(getattr(row, "volatility_regime", "unknown") or "unknown"),
                "is_correct": int(actual_direction == predicted_direction),
            }
        )
    return records


def _stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts)
    return sum((idx + 1) * ord(char) for idx, char in enumerate(payload)) % (2**32 - 1)


def _baseline_summary_rows(
    eval_frame: pd.DataFrame,
    *,
    frequency: str,
    ticker: str,
    horizon: int,
    threshold: float,
    min_obs_per_group: int,
    seed: int,
) -> list[dict[str, Any]]:
    if eval_frame.empty:
        return []

    frame = eval_frame.copy().sort_values("timestamp").reset_index(drop=True)
    actual_return = _finite_numeric(frame["actual_return"])
    actual_direction = (actual_return > 0.0).astype(int)
    valid_actual = actual_return.notna() & (actual_return != 0.0)

    candidates: dict[str, pd.Series] = {
        "always_up": pd.Series(1, index=frame.index, dtype=int),
    }
    if "return_1" in frame.columns:
        previous_return = _finite_numeric(frame["return_1"])
        candidates["previous_direction"] = (previous_return > 0.0).astype("Int64").where(previous_return.notna())

    rng = np.random.default_rng(_stable_seed(seed, frequency, ticker, horizon, "random_seeded_direction"))
    candidates["random_seeded_direction"] = pd.Series(rng.integers(0, 2, size=len(frame)), index=frame.index, dtype=int)

    for column in ("rolling_return_mean_5", "rolling_return_mean_10", "rolling_return_mean_20"):
        if column in frame.columns:
            signal = _finite_numeric(frame[column])
            candidates["moving_average_signal"] = (signal > 0.0).astype("Int64").where(signal.notna())
            break

    rows: list[dict[str, Any]] = []
    for baseline_name, predicted_direction in candidates.items():
        predicted_numeric = pd.to_numeric(predicted_direction, errors="coerce")
        mask = valid_actual & predicted_numeric.isin([0, 1])
        n_obs = int(mask.sum())
        accuracy = float((predicted_numeric.loc[mask].astype(int) == actual_direction.loc[mask]).mean()) if n_obs else float("nan")
        reliable = n_obs >= int(min_obs_per_group)
        rows.append(
            {
                "frequency": frequency,
                "baseline": baseline_name,
                "horizon": int(horizon),
                "ticker": ticker,
                "n_obs": n_obs,
                "accuracy": accuracy,
                "reliable": bool(reliable),
                "passed_60pct": bool(reliable and n_obs > 0 and accuracy >= float(threshold)),
            }
        )
    return rows


def _baseline_prediction_records(
    eval_frame: pd.DataFrame,
    *,
    frequency: str,
    ticker: str,
    horizon: int,
    seed: int,
) -> list[dict[str, Any]]:
    if eval_frame.empty:
        return []

    frame = eval_frame.copy().sort_values("timestamp").reset_index(drop=True)
    actual_return = _finite_numeric(frame["actual_return"])
    actual_direction = (actual_return > 0.0).astype(int)
    valid_actual = actual_return.notna() & (actual_return != 0.0)
    candidates: dict[str, pd.Series] = {
        "always_up": pd.Series(1, index=frame.index, dtype=int),
    }
    if "return_1" in frame.columns:
        previous_return = _finite_numeric(frame["return_1"])
        candidates["previous_direction"] = (previous_return > 0.0).astype("Int64").where(previous_return.notna())
    rng = np.random.default_rng(_stable_seed(seed, frequency, ticker, horizon, "random_seeded_direction"))
    candidates["random_seeded_direction"] = pd.Series(rng.integers(0, 2, size=len(frame)), index=frame.index, dtype=int)
    for column in ("rolling_return_mean_5", "rolling_return_mean_10", "rolling_return_mean_20"):
        if column in frame.columns:
            signal = _finite_numeric(frame[column])
            candidates["moving_average_signal"] = (signal > 0.0).astype("Int64").where(signal.notna())
            break

    records: list[dict[str, Any]] = []
    for baseline_name, predicted_direction in candidates.items():
        predicted_numeric = pd.to_numeric(predicted_direction, errors="coerce")
        mask = valid_actual & predicted_numeric.isin([0, 1])
        for idx in frame.index[mask]:
            predicted = int(predicted_numeric.loc[idx])
            actual = int(actual_direction.loc[idx])
            timestamp = pd.Timestamp(frame.loc[idx, "timestamp"])
            records.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "date": str(timestamp.date()),
                    "ticker": ticker,
                    "frequency": frequency,
                    "horizon": int(horizon),
                    "baseline": baseline_name,
                    "actual_direction": actual,
                    "predicted_direction": predicted,
                    "is_correct": int(actual == predicted),
                    "regime": str(frame.loc[idx, "regime"]) if "regime" in frame.columns else "unknown",
                    "volatility_regime": str(frame.loc[idx, "volatility_regime"]) if "volatility_regime" in frame.columns else "unknown",
                }
            )
    return records


def _parameter_grid(model_name: str) -> list[dict[str, Any]]:
    if model_name == "lightgbm":
        return [
            {"num_leaves": 15, "learning_rate": 0.03, "n_estimators": 80, "max_depth": 3, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8},
            {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 120, "max_depth": 4, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8},
            {"num_leaves": 31, "learning_rate": 0.03, "n_estimators": 180, "max_depth": 5, "min_child_samples": 30, "subsample": 0.9, "colsample_bytree": 0.8},
            {"num_leaves": 63, "learning_rate": 0.02, "n_estimators": 220, "max_depth": 6, "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.9},
            {"num_leaves": 15, "learning_rate": 0.08, "n_estimators": 100, "max_depth": 3, "min_child_samples": 40, "subsample": 0.9, "colsample_bytree": 0.9},
        ]
    if model_name == "xgboost":
        return [
            {"max_depth": 2, "learning_rate": 0.03, "n_estimators": 80, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 1.0, "reg_alpha": 0.0},
            {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 120, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 1.0, "reg_alpha": 0.0},
            {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 180, "subsample": 0.9, "colsample_bytree": 0.8, "min_child_weight": 3, "reg_lambda": 1.5, "reg_alpha": 0.0},
            {"max_depth": 3, "learning_rate": 0.02, "n_estimators": 220, "subsample": 0.8, "colsample_bytree": 0.9, "min_child_weight": 5, "reg_lambda": 2.0, "reg_alpha": 0.1},
            {"max_depth": 2, "learning_rate": 0.08, "n_estimators": 100, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 1, "reg_lambda": 1.0, "reg_alpha": 0.1},
        ]
    return []


def _ordered_tuning_candidates(model_name: str, *, trials: int, seed: int, frequency: str, horizon: int) -> list[dict[str, Any]]:
    grid = _parameter_grid(model_name)
    if not grid:
        return []
    rng = np.random.default_rng(_stable_seed(seed, frequency, horizon, model_name, "tuning"))
    order = list(rng.permutation(len(grid)))
    ordered = [grid[idx] for idx in order]
    return ordered[: max(1, min(int(trials), len(ordered)))]


def _pre_eval_validation_split(
    labeled: pd.DataFrame,
    eval_start: str,
    *,
    config: BenchmarkConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = _training_label_cutoff(config)
    if cutoff is not None:
        cutoff_mask = pd.to_datetime(labeled["target_timestamp"], errors="coerce") <= cutoff
    else:
        cutoff_mask = pd.to_datetime(labeled["target_timestamp"], errors="coerce") < _to_date(eval_start)
    pre_eval = labeled[
        cutoff_mask
        & labeled["actual_return"].notna()
        & (labeled["actual_return"] != 0.0)
    ].copy().sort_values("timestamp")
    if len(pre_eval) < 120:
        return pd.DataFrame(), pd.DataFrame()
    valid_size = min(max(int(len(pre_eval) * 0.20), 60), max(len(pre_eval) - 60, 1))
    train = pre_eval.iloc[:-valid_size].copy()
    valid = pre_eval.iloc[-valid_size:].copy()
    if len(train) < 60 or valid.empty:
        return pd.DataFrame(), pd.DataFrame()
    return train, valid


def _directional_validation_score(
    *,
    model_name: str,
    target_mode: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
    params: dict[str, Any],
) -> float:
    if target_mode == "classification":
        predicted_direction, _ = fit_predict_classifier(model_name, train, valid, feature_columns, seed, params)
    else:
        predicted_return = fit_predict_model(model_name, train, valid, feature_columns, "actual_return", seed, params)
        predicted_direction = (np.asarray(predicted_return, dtype=float) > 0.0).astype(int)
    actual_return = _finite_numeric(valid["actual_return"])
    mask = actual_return.notna() & (actual_return != 0.0)
    if int(mask.sum()) <= 0:
        return float("nan")
    actual_direction = (actual_return.loc[mask] > 0.0).astype(int).to_numpy()
    predicted = np.asarray(predicted_direction, dtype=int)[np.where(mask.to_numpy())[0]]
    return float((actual_direction == predicted).mean())


def build_horizon_tuning_summary(
    *,
    prepared_raw: pd.DataFrame,
    frequency: str,
    horizons: list[int],
    models: list[str],
    eval_start: str,
    target_mode: str,
    feature_frames_by_ticker: dict[str, tuple[pd.DataFrame, list[str]]],
    config: BenchmarkConfig,
) -> tuple[pd.DataFrame, dict[tuple[str, int, str], dict[str, Any]]]:
    if not config.enable_horizon_tuning:
        return pd.DataFrame(columns=TUNING_SUMMARY_COLUMNS), {}
    tuning_models = [model for model in models if model in set(config.tuning_models) and model in {"xgboost", "lightgbm"}]
    rows: list[dict[str, Any]] = []
    best_params: dict[tuple[str, int, str], dict[str, Any]] = {}
    started = time.monotonic()

    for horizon in horizons:
        labeled_frames: list[pd.DataFrame] = []
        feature_columns: list[str] = []
        for feature_frame, ticker_features in feature_frames_by_ticker.values():
            if feature_frame.empty or not ticker_features:
                continue
            feature_columns = ticker_features
            labeled = add_horizon_labels(feature_frame, horizon)
            if not labeled.empty:
                labeled_frames.append(labeled)
        if not labeled_frames or not feature_columns:
            for model_name in tuning_models:
                rows.append(
                    {
                        "frequency": frequency,
                        "horizon": int(horizon),
                        "model": model_name,
                        "tuning_trials": int(config.tuning_trials),
                        "best_params": "{}",
                        "best_validation_score": None,
                        "tuning_metric": config.tuning_metric,
                        "tuning_status": "skipped",
                        "error_message": "no_pre_eval_labeled_rows",
                        "tuning_backend": "grid",
                    }
                )
            continue
        pooled = pd.concat(labeled_frames, ignore_index=True).sort_values(["timestamp", "ticker"]).reset_index(drop=True)
        train, valid = _pre_eval_validation_split(pooled, eval_start, config=config)
        for model_name in tuning_models:
            best_score: float | None = None
            selected_params: dict[str, Any] = {}
            status = "skipped"
            error_message = ""
            candidates = _ordered_tuning_candidates(
                model_name,
                trials=config.tuning_trials,
                seed=config.tuning_seed,
                frequency=frequency,
                horizon=int(horizon),
            )
            if train.empty or valid.empty:
                error_message = "insufficient_pre_eval_validation_rows"
            elif not candidates:
                error_message = "no_tuning_grid"
            else:
                for trial_idx, params in enumerate(candidates, start=1):
                    if config.tuning_time_budget_seconds is not None and time.monotonic() - started > float(config.tuning_time_budget_seconds):
                        error_message = "tuning_time_budget_exceeded"
                        break
                    try:
                        score = _directional_validation_score(
                            model_name=model_name,
                            target_mode=target_mode,
                            train=train,
                            valid=valid,
                            feature_columns=feature_columns,
                            seed=int(config.tuning_seed) + trial_idx,
                            params=params,
                        )
                        if math.isfinite(score) and (best_score is None or score > best_score):
                            best_score = float(score)
                            selected_params = dict(params)
                            status = "ok"
                    except Exception as exc:
                        error_message = str(exc)
                        continue
            if selected_params:
                best_params[(frequency, int(horizon), model_name)] = selected_params
            rows.append(
                {
                    "frequency": frequency,
                    "horizon": int(horizon),
                    "model": model_name,
                    "tuning_trials": int(min(config.tuning_trials, len(candidates))) if candidates else int(config.tuning_trials),
                    "best_params": json.dumps(_json_safe(selected_params), sort_keys=True),
                    "best_validation_score": best_score,
                    "tuning_metric": config.tuning_metric,
                    "tuning_status": status,
                    "error_message": error_message,
                    "tuning_backend": "grid",
                }
            )
    return pd.DataFrame(rows, columns=TUNING_SUMMARY_COLUMNS), best_params


def run_frequency_benchmark(
    *,
    raw_df: pd.DataFrame,
    frequency: str,
    horizons: list[int],
    models: list[str],
    initial_train_start: str,
    initial_train_end: str,
    eval_start: str,
    eval_end: str,
    threshold: float,
    provider: str,
    universe: str,
    retrain_frequency: str,
    seed: int,
    min_history_days: int | None,
    min_obs_per_group: int,
    max_daily_gap_days: int,
    config: BenchmarkConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    time_column = "datetime" if frequency == "hourly" else "date"
    prediction_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    baseline_prediction_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    model_error_rows: list[dict[str, Any]] = []
    base_models = [model for model in models if model != "stacking"]
    stacking_base_models = [model for model in base_models if model in BENCHMARK_STACKING_BASE_MODELS]

    if raw_df.empty:
        predictions = pd.DataFrame(columns=PREDICTED_COLUMNS)
        accuracy_summary = _build_accuracy_summary(predictions, threshold, min_obs_per_group)
        baseline_summary = pd.DataFrame(columns=BASELINE_SUMMARY_COLUMNS)
        benchmark_summary = _build_benchmark_summary(
            frequency=frequency,
            provider=provider,
            universe=universe,
            raw_data_start="",
            raw_data_end="",
            initial_train_start=initial_train_start,
            initial_train_end=initial_train_end,
            evaluation_start=eval_start,
            evaluation_end=eval_end,
            threshold=threshold,
            predictions=predictions,
            accuracy_summary=accuracy_summary,
            min_obs_per_group=min_obs_per_group,
            data_gap_warnings=[],
        )
        benchmark_summary.update(_model_execution_summary(requested_models=models, predictions=predictions, skipped=[]))
        benchmark_summary["target_mode"] = str(config.target_mode)
        benchmark_summary["regime_evaluation_enabled"] = bool(config.enable_regime_evaluation)
        benchmark_summary["confidence_filter_enabled"] = bool(config.enable_confidence_filter)
        benchmark_summary["confidence_sweep_enabled"] = bool(config.enable_confidence_threshold_sweep)
        benchmark_summary["horizon_tuning_enabled"] = bool(config.enable_horizon_tuning)
        benchmark_summary["tuned_models"] = list(config.tuning_models if config.enable_horizon_tuning else [])
        benchmark_summary["tuning_trials"] = int(config.tuning_trials if config.enable_horizon_tuning else 0)
        benchmark_summary["tuning_summary_path"] = "tuning_summary.csv" if config.enable_horizon_tuning else ""
        benchmark_summary.update(_benchmark_metadata_fields(config, data_end=""))
        model_error_summary = pd.DataFrame(columns=MODEL_ERROR_COLUMNS)
        baseline_predictions = pd.DataFrame(columns=BASELINE_PREDICTION_COLUMNS)
        tuning_summary = pd.DataFrame(columns=TUNING_SUMMARY_COLUMNS)
        return predictions, accuracy_summary, benchmark_summary, baseline_summary, model_error_summary, baseline_predictions, tuning_summary

    prepared_raw = raw_df.copy()
    prepared_raw[time_column] = pd.to_datetime(prepared_raw[time_column], errors="coerce")
    prepared_raw = prepared_raw.dropna(subset=[time_column])
    prepared_raw, data_gap_warnings = assign_continuity_segments(
        prepared_raw,
        frequency=frequency,
        time_column=time_column,
        max_daily_gap_days=max_daily_gap_days,
    )
    raw_data_start = _safe_iso_date(prepared_raw[time_column].min())
    raw_data_end = _safe_iso_date(prepared_raw[time_column].max())
    effective_train_start, effective_train_end = _effective_time_bounds(
        prepared_raw,
        time_column=time_column,
        start=initial_train_start,
        end=initial_train_end,
    )
    effective_eval_start, effective_eval_end = _effective_time_bounds(
        prepared_raw,
        time_column=time_column,
        start=eval_start,
        end=eval_end,
    )

    feature_frames_by_ticker: dict[str, tuple[pd.DataFrame, list[str]]] = {}
    for ticker, ticker_raw in prepared_raw.groupby("ticker", sort=True):
        ticker = str(ticker).upper()
        feature_frame, feature_columns = build_feature_frame(ticker_raw, frequency)
        if not feature_frame.empty:
            if config.enable_regime_evaluation:
                feature_frame = add_regime_labels(feature_frame, config)
            else:
                feature_frame["regime"] = "unknown"
                feature_frame["volatility_regime"] = "unknown"
        feature_frames_by_ticker[ticker] = (feature_frame, feature_columns)

    tuning_summary, tuned_params = build_horizon_tuning_summary(
        prepared_raw=prepared_raw,
        frequency=frequency,
        horizons=horizons,
        models=models,
        eval_start=eval_start,
        target_mode=config.target_mode,
        feature_frames_by_ticker=feature_frames_by_ticker,
        config=config,
    )

    for ticker, (feature_frame, feature_columns) in feature_frames_by_ticker.items():
        if not feature_columns or feature_frame.empty:
            skipped.append({"ticker": ticker, "reason": "empty_feature_frame"})
            continue
        history_count = int((feature_frame["timestamp"] <= _to_end_of_day(initial_train_end)).sum())
        if min_history_days is not None and history_count < int(min_history_days):
            skipped.append({"ticker": ticker, "reason": f"history_below_min_history_days:{history_count}"})
            continue

        for horizon in horizons:
            labeled = add_horizon_labels(feature_frame, horizon)
            eval_mask = (
                (labeled["timestamp"] >= _to_date(eval_start))
                & (labeled["timestamp"] <= _to_end_of_day(eval_end))
                & labeled["actual_return"].notna()
                & (labeled["actual_return"] != 0.0)
            )
            eval_frame = labeled.loc[eval_mask].copy()
            if eval_frame.empty:
                skipped.append({"ticker": ticker, "horizon": horizon, "reason": "no_evaluation_rows"})
                continue
            baseline_rows.extend(
                _baseline_summary_rows(
                    eval_frame,
                    frequency=frequency,
                    ticker=ticker,
                    horizon=horizon,
                    threshold=threshold,
                    min_obs_per_group=min_obs_per_group,
                    seed=seed,
                )
            )
            baseline_prediction_rows.extend(
                _baseline_prediction_records(
                    eval_frame,
                    frequency=frequency,
                    ticker=ticker,
                    horizon=horizon,
                    seed=seed,
                )
            )
            eval_frame["retrain_group"] = _retrain_group_key(eval_frame["timestamp"], retrain_frequency)
            for _, chunk in eval_frame.groupby("retrain_group", sort=True):
                chunk = chunk.sort_values("timestamp").copy()
                chunk_start = pd.Timestamp(chunk["timestamp"].min())
                train_mask = _training_label_mask(
                    labeled,
                    initial_train_start=initial_train_start,
                    forecast_chunk_start=chunk_start,
                    config=config,
                )
                train_frame = labeled.loc[train_mask].copy().sort_values("timestamp")
                if train_frame.empty:
                    skipped.append({"ticker": ticker, "horizon": horizon, "reason": "empty_training_window"})
                    continue
                for model_name in models:
                    try:
                        model_params = tuned_params.get((frequency, int(horizon), model_name))
                        stacking_params = {
                            base_name: tuned_params[(frequency, int(horizon), base_name)]
                            for base_name in stacking_base_models
                            if (frequency, int(horizon), base_name) in tuned_params
                        }
                        if config.target_mode == "classification":
                            if model_name == "stacking":
                                predicted_direction, confidence = fit_predict_stacking_classifier(
                                    stacking_base_models,
                                    train_frame,
                                    chunk,
                                    feature_columns,
                                    seed,
                                    stacking_params,
                                )
                            else:
                                predicted_direction, confidence = fit_predict_classifier(
                                    model_name,
                                    train_frame,
                                    chunk,
                                    feature_columns,
                                    seed,
                                    model_params,
                                )
                            valid_rows = _valid_classification_prediction_rows(chunk, predicted_direction, confidence)
                        elif model_name == "stacking":
                            predictions = fit_predict_stacking(
                                stacking_base_models,
                                train_frame,
                                chunk,
                                feature_columns,
                                "actual_return",
                                seed,
                                stacking_params,
                            )
                            valid_rows = _valid_prediction_rows(chunk, predictions)
                        else:
                            predictions = fit_predict_model(
                                model_name,
                                train_frame,
                                chunk,
                                feature_columns,
                                "actual_return",
                                seed,
                                model_params,
                            )
                            valid_rows = _valid_prediction_rows(chunk, predictions)
                        prediction_rows.extend(
                            _prediction_records(
                                valid_rows,
                                ticker=ticker,
                                frequency=frequency,
                                horizon=horizon,
                                model_name=model_name,
                                target_mode=config.target_mode,
                                enable_confidence_filter=config.enable_confidence_filter,
                                confidence_threshold=config.confidence_threshold,
                                no_trade_band=config.no_trade_band,
                            )
                        )
                    except Exception as exc:
                        skipped.append({"ticker": ticker, "horizon": horizon, "model": model_name, "reason": str(exc)})
                        model_error_rows.append(
                            {
                                "frequency": frequency,
                                "ticker": ticker,
                                "horizon": int(horizon),
                                "model": model_name,
                                "error_type": exc.__class__.__name__,
                                "error_message": str(exc),
                                "stage": "fit_predict",
                            }
                        )

    predictions = pd.DataFrame(prediction_rows, columns=PREDICTED_COLUMNS)
    accuracy_summary = _build_accuracy_summary(predictions, threshold, min_obs_per_group)
    baseline_summary = (
        pd.DataFrame(baseline_rows, columns=BASELINE_SUMMARY_COLUMNS)
        if baseline_rows
        else pd.DataFrame(columns=BASELINE_SUMMARY_COLUMNS)
    )
    baseline_predictions = (
        pd.DataFrame(baseline_prediction_rows, columns=BASELINE_PREDICTION_COLUMNS)
        if baseline_prediction_rows
        else pd.DataFrame(columns=BASELINE_PREDICTION_COLUMNS)
    )
    benchmark_summary = _build_benchmark_summary(
        frequency=frequency,
        provider=provider,
        universe=universe,
        raw_data_start=raw_data_start,
        raw_data_end=raw_data_end,
        initial_train_start=initial_train_start,
        initial_train_end=initial_train_end,
        evaluation_start=eval_start,
        evaluation_end=eval_end,
        threshold=threshold,
        predictions=predictions,
        accuracy_summary=accuracy_summary,
        min_obs_per_group=min_obs_per_group,
        data_gap_warnings=data_gap_warnings,
        effective_train_start=effective_train_start,
        effective_train_end=effective_train_end,
        effective_eval_start=effective_eval_start,
        effective_eval_end=effective_eval_end,
    )
    benchmark_summary["skipped_count"] = int(len(skipped))
    benchmark_summary["skipped_examples"] = skipped[:20]
    benchmark_summary.update(_model_execution_summary(requested_models=models, predictions=predictions, skipped=skipped))
    benchmark_summary["target_mode"] = str(config.target_mode)
    benchmark_summary["regime_evaluation_enabled"] = bool(config.enable_regime_evaluation)
    benchmark_summary["confidence_filter_enabled"] = bool(config.enable_confidence_filter)
    benchmark_summary["confidence_sweep_enabled"] = bool(config.enable_confidence_threshold_sweep)
    benchmark_summary["horizon_tuning_enabled"] = bool(config.enable_horizon_tuning)
    benchmark_summary["tuned_models"] = list(config.tuning_models if config.enable_horizon_tuning else [])
    benchmark_summary["tuning_trials"] = int(config.tuning_trials)
    benchmark_summary["tuning_summary_path"] = "tuning_summary.csv" if config.enable_horizon_tuning else ""
    benchmark_summary.update(_benchmark_metadata_fields(config, data_end=raw_data_end))
    model_error_summary = (
        pd.DataFrame(model_error_rows, columns=MODEL_ERROR_COLUMNS)
        if model_error_rows
        else pd.DataFrame(columns=MODEL_ERROR_COLUMNS)
    )
    benchmark_summary["model_error_count"] = int(len(model_error_summary))
    return predictions, accuracy_summary, benchmark_summary, baseline_summary, model_error_summary, baseline_predictions, tuning_summary


def _build_accuracy_summary(predictions: pd.DataFrame, threshold: float, min_obs_per_group: int) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=ACCURACY_COLUMNS)
    rows = []
    grouped = predictions.groupby(["frequency", "model", "horizon", "ticker"], dropna=False, sort=True)
    for keys, group in grouped:
        frequency, model, horizon, ticker = keys
        accuracy = float(pd.to_numeric(group["is_correct"], errors="coerce").mean())
        n_obs = int(len(group))
        reliable = n_obs >= int(min_obs_per_group)
        rows.append(
            {
                "frequency": frequency,
                "model": model,
                "horizon": int(horizon),
                "ticker": ticker,
                "n_obs": n_obs,
                "accuracy": accuracy,
                "reliable": bool(reliable),
                "passed_60pct": bool(reliable and accuracy >= float(threshold)),
            }
        )
    return pd.DataFrame(rows, columns=ACCURACY_COLUMNS).sort_values(["frequency", "model", "horizon", "ticker"]).reset_index(drop=True)


def _finite_float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _aggregate_accuracy_summary(frame: pd.DataFrame, *, group_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*group_columns, "n_obs", "accuracy"])

    working = frame.copy()
    working["n_obs"] = pd.to_numeric(working.get("n_obs"), errors="coerce").fillna(0).astype(int)
    working["accuracy"] = pd.to_numeric(working.get("accuracy"), errors="coerce")
    working = working[(working["n_obs"] > 0) & working["accuracy"].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=[*group_columns, "n_obs", "accuracy"])

    rows: list[dict[str, Any]] = []
    for keys, group in working.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n_obs = int(group["n_obs"].sum())
        correct = float((group["accuracy"] * group["n_obs"]).sum())
        accuracy = correct / n_obs if n_obs else float("nan")
        row = {column: value for column, value in zip(group_columns, keys, strict=False)}
        row["n_obs"] = n_obs
        row["accuracy"] = accuracy
        rows.append(row)

    return pd.DataFrame(rows, columns=[*group_columns, "n_obs", "accuracy"])


def build_baseline_delta_summary(accuracy_summary: pd.DataFrame, baseline_summary: pd.DataFrame) -> pd.DataFrame:
    if accuracy_summary.empty or baseline_summary.empty:
        return pd.DataFrame(columns=BASELINE_DELTA_COLUMNS)

    model_accuracy = _aggregate_accuracy_summary(
        accuracy_summary,
        group_columns=["frequency", "model", "horizon"],
    )
    baseline_accuracy = _aggregate_accuracy_summary(
        baseline_summary,
        group_columns=["frequency", "baseline", "horizon"],
    )
    if model_accuracy.empty or baseline_accuracy.empty:
        return pd.DataFrame(columns=BASELINE_DELTA_COLUMNS)

    rows: list[dict[str, Any]] = []
    for model_row in model_accuracy.itertuples(index=False):
        matching_baselines = baseline_accuracy[
            (baseline_accuracy["frequency"].astype(str) == str(model_row.frequency))
            & (pd.to_numeric(baseline_accuracy["horizon"], errors="coerce") == int(model_row.horizon))
        ]
        for baseline_row in matching_baselines.itertuples(index=False):
            model_acc = _finite_float_or_none(model_row.accuracy)
            baseline_acc = _finite_float_or_none(baseline_row.accuracy)
            delta = (
                model_acc - baseline_acc
                if model_acc is not None and baseline_acc is not None
                else None
            )
            rows.append(
                {
                    "frequency": str(model_row.frequency),
                    "model": str(model_row.model),
                    "horizon": int(model_row.horizon),
                    "baseline": str(baseline_row.baseline),
                    "model_accuracy": model_acc,
                    "baseline_accuracy": baseline_acc,
                    "accuracy_delta": delta,
                    "model_n_obs": int(model_row.n_obs),
                    "baseline_n_obs": int(baseline_row.n_obs),
                    "model_better_than_baseline": bool(delta is not None and delta > 0.0),
                }
            )

    return (
        pd.DataFrame(rows, columns=BASELINE_DELTA_COLUMNS)
        .sort_values(["frequency", "model", "horizon", "baseline"])
        .reset_index(drop=True)
    )


def _binomial_p_value_greater(successes: int, n_obs: int, null_accuracy: float = 0.50) -> float | None:
    global _BINOMIAL_WARNING_EMITTED
    if n_obs <= 0:
        return None
    try:
        from scipy.stats import binomtest

        return float(binomtest(int(successes), int(n_obs), p=float(null_accuracy), alternative="greater").pvalue)
    except Exception as exc:
        if not _BINOMIAL_WARNING_EMITTED:
            logger.warning("binomial_p_value_unavailable", reason=str(exc))
            _BINOMIAL_WARNING_EMITTED = True
        return None


def _bootstrap_accuracy_ci(is_correct: pd.Series, *, samples: int, seed: int) -> tuple[float | None, float | None]:
    values = pd.to_numeric(is_correct, errors="coerce").dropna()
    values = values[values.isin([0, 1])]
    n_obs = int(len(values))
    if n_obs == 0 or int(samples) <= 0:
        return None, None

    p_hat = float(values.mean())
    rng = np.random.default_rng(int(seed))
    boot_means = rng.binomial(n_obs, p_hat, size=int(samples)) / float(n_obs)
    ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])
    return float(ci_low), float(ci_high)


def build_significance_summary(
    predictions: pd.DataFrame,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=SIGNIFICANCE_COLUMNS)

    working = predictions.copy()
    working["is_correct"] = pd.to_numeric(working["is_correct"], errors="coerce")
    working = working[working["is_correct"].isin([0, 1])].copy()
    if working.empty:
        return pd.DataFrame(columns=SIGNIFICANCE_COLUMNS)

    rows: list[dict[str, Any]] = []
    grouped = working.groupby(["frequency", "model", "horizon"], dropna=False, sort=True)
    for group_index, (keys, group) in enumerate(grouped):
        frequency, model, horizon = keys
        n_obs = int(len(group))
        successes = int(group["is_correct"].sum())
        accuracy = successes / n_obs if n_obs else float("nan")
        p_value = _binomial_p_value_greater(successes, n_obs, 0.50)
        ci_low, ci_high = _bootstrap_accuracy_ci(
            group["is_correct"],
            samples=bootstrap_samples,
            seed=int(bootstrap_seed) + group_index,
        )
        rows.append(
            {
                "frequency": str(frequency),
                "model": str(model),
                "horizon": int(horizon),
                "n_obs": n_obs,
                "accuracy": _finite_float_or_none(accuracy),
                "null_accuracy": 0.50,
                "binomial_p_value": p_value,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "significant_at_5pct": bool(p_value is not None and p_value < 0.05),
                "significant_at_10pct": bool(p_value is not None and p_value < 0.10),
            }
        )

    return (
        pd.DataFrame(rows, columns=SIGNIFICANCE_COLUMNS)
        .sort_values(["frequency", "model", "horizon"])
        .reset_index(drop=True)
    )


def _mcnemar_p_value(model_correct_only: int, baseline_correct_only: int) -> float | None:
    discordant = int(model_correct_only) + int(baseline_correct_only)
    if discordant <= 0:
        return None
    try:
        from scipy.stats import binomtest

        return float(binomtest(int(model_correct_only), int(discordant), p=0.5, alternative="two-sided").pvalue)
    except Exception:
        return None


def build_mcnemar_summary(predictions: pd.DataFrame, baseline_predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or baseline_predictions.empty:
        return pd.DataFrame(columns=MCNEMAR_SUMMARY_COLUMNS)
    model = predictions.copy()
    baseline = baseline_predictions.copy()
    join_cols = ["timestamp", "ticker", "frequency", "horizon"]
    required_model = [*join_cols, "model", "is_correct"]
    required_baseline = [*join_cols, "baseline", "is_correct"]
    if any(column not in model.columns for column in required_model) or any(column not in baseline.columns for column in required_baseline):
        return pd.DataFrame(columns=MCNEMAR_SUMMARY_COLUMNS)
    merged = model[required_model].merge(
        baseline[required_baseline],
        on=join_cols,
        how="inner",
        suffixes=("_model", "_baseline"),
    )
    if merged.empty:
        return pd.DataFrame(columns=MCNEMAR_SUMMARY_COLUMNS)
    merged["is_correct_model"] = pd.to_numeric(merged["is_correct_model"], errors="coerce")
    merged["is_correct_baseline"] = pd.to_numeric(merged["is_correct_baseline"], errors="coerce")
    merged = merged[merged["is_correct_model"].isin([0, 1]) & merged["is_correct_baseline"].isin([0, 1])].copy()
    if merged.empty:
        return pd.DataFrame(columns=MCNEMAR_SUMMARY_COLUMNS)

    rows: list[dict[str, Any]] = []
    for keys, group in merged.groupby(["frequency", "model", "horizon", "baseline"], sort=True):
        frequency, model_name, horizon, baseline_name = keys
        model_correct = group["is_correct_model"].astype(int).eq(1)
        baseline_correct = group["is_correct_baseline"].astype(int).eq(1)
        model_only = int((model_correct & ~baseline_correct).sum())
        baseline_only = int((~model_correct & baseline_correct).sum())
        both_correct = int((model_correct & baseline_correct).sum())
        both_wrong = int((~model_correct & ~baseline_correct).sum())
        rows.append(
            {
                "frequency": str(frequency),
                "model": str(model_name),
                "horizon": int(horizon),
                "baseline": str(baseline_name),
                "matched_n_obs": int(len(group)),
                "model_correct_only": model_only,
                "baseline_correct_only": baseline_only,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "mcnemar_p_value": _mcnemar_p_value(model_only, baseline_only),
            }
        )
    return pd.DataFrame(rows, columns=MCNEMAR_SUMMARY_COLUMNS).sort_values(["frequency", "model", "horizon", "baseline"]).reset_index(drop=True)


def _expanded_regime_frame(frame: pd.DataFrame, *, entity_column: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        base = row._asdict()
        for column in ("regime", "volatility_regime"):
            regime = str(base.get(column, "unknown") or "unknown")
            if regime == "unknown":
                continue
            expanded = dict(base)
            expanded["regime_bucket"] = regime
            rows.append(expanded)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_regime_accuracy_summary(predictions: pd.DataFrame, threshold: float, min_obs_per_group: int) -> pd.DataFrame:
    expanded = _expanded_regime_frame(predictions, entity_column="model")
    if expanded.empty:
        return pd.DataFrame(columns=REGIME_ACCURACY_COLUMNS)
    expanded["is_correct"] = pd.to_numeric(expanded["is_correct"], errors="coerce")
    expanded = expanded[expanded["is_correct"].isin([0, 1])].copy()
    rows: list[dict[str, Any]] = []
    for keys, group in expanded.groupby(["frequency", "regime_bucket", "model", "horizon"], sort=True):
        frequency, regime, model_name, horizon = keys
        n_obs = int(len(group))
        accuracy = float(group["is_correct"].mean()) if n_obs else float("nan")
        reliable = n_obs >= int(min_obs_per_group)
        rows.append(
            {
                "frequency": str(frequency),
                "regime": str(regime),
                "model": str(model_name),
                "horizon": int(horizon),
                "n_obs": n_obs,
                "accuracy": accuracy,
                "passed_60pct": bool(reliable and accuracy >= float(threshold)),
                "reliable": bool(reliable),
            }
        )
    return pd.DataFrame(rows, columns=REGIME_ACCURACY_COLUMNS).sort_values(["frequency", "regime", "model", "horizon"]).reset_index(drop=True)


def build_regime_baseline_delta_summary(
    predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
) -> pd.DataFrame:
    model_expanded = _expanded_regime_frame(predictions, entity_column="model")
    baseline_expanded = _expanded_regime_frame(baseline_predictions, entity_column="baseline")
    if model_expanded.empty or baseline_expanded.empty:
        return pd.DataFrame(columns=REGIME_BASELINE_DELTA_COLUMNS)
    model_expanded["is_correct"] = pd.to_numeric(model_expanded["is_correct"], errors="coerce")
    baseline_expanded["is_correct"] = pd.to_numeric(baseline_expanded["is_correct"], errors="coerce")
    model_grouped = (
        model_expanded[model_expanded["is_correct"].isin([0, 1])]
        .groupby(["frequency", "regime_bucket", "model", "horizon"], sort=True)["is_correct"]
        .agg(["mean", "count"])
        .reset_index()
    )
    baseline_grouped = (
        baseline_expanded[baseline_expanded["is_correct"].isin([0, 1])]
        .groupby(["frequency", "regime_bucket", "baseline", "horizon"], sort=True)["is_correct"]
        .agg(["mean", "count"])
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for model_row in model_grouped.itertuples(index=False):
        matching = baseline_grouped[
            (baseline_grouped["frequency"].astype(str) == str(model_row.frequency))
            & (baseline_grouped["regime_bucket"].astype(str) == str(model_row.regime_bucket))
            & (pd.to_numeric(baseline_grouped["horizon"], errors="coerce") == int(model_row.horizon))
        ]
        for baseline_row in matching.itertuples(index=False):
            model_accuracy = _finite_float_or_none(model_row.mean)
            baseline_accuracy = _finite_float_or_none(baseline_row.mean)
            delta = model_accuracy - baseline_accuracy if model_accuracy is not None and baseline_accuracy is not None else None
            rows.append(
                {
                    "frequency": str(model_row.frequency),
                    "regime": str(model_row.regime_bucket),
                    "model": str(model_row.model),
                    "horizon": int(model_row.horizon),
                    "baseline": str(baseline_row.baseline),
                    "model_accuracy": model_accuracy,
                    "baseline_accuracy": baseline_accuracy,
                    "accuracy_delta": delta,
                    "model_better_than_baseline": bool(delta is not None and delta > 0.0),
                }
            )
    return pd.DataFrame(rows, columns=REGIME_BASELINE_DELTA_COLUMNS).sort_values(["frequency", "regime", "model", "horizon", "baseline"]).reset_index(drop=True)


def build_classification_accuracy_summary(
    predictions: pd.DataFrame,
    *,
    threshold: float,
    min_obs_per_group: int,
) -> pd.DataFrame:
    if predictions.empty or "target_mode" not in predictions.columns:
        return pd.DataFrame(columns=CLASSIFICATION_ACCURACY_COLUMNS)
    working = predictions[predictions["target_mode"].astype(str).str.lower().eq("classification")].copy()
    if working.empty:
        return pd.DataFrame(columns=CLASSIFICATION_ACCURACY_COLUMNS)
    working["is_correct"] = pd.to_numeric(working["is_correct"], errors="coerce")
    working = working[working["is_correct"].isin([0, 1])].copy()
    rows: list[dict[str, Any]] = []
    for keys, group in working.groupby(["frequency", "model", "horizon", "target_mode"], sort=True):
        frequency, model_name, horizon, target_mode = keys
        n_obs = int(len(group))
        accuracy = float(group["is_correct"].mean()) if n_obs else float("nan")
        reliable = n_obs >= int(min_obs_per_group)
        rows.append(
            {
                "frequency": str(frequency),
                "model": str(model_name),
                "horizon": int(horizon),
                "target_mode": str(target_mode),
                "n_obs": n_obs,
                "accuracy": accuracy,
                "passed_60pct": bool(reliable and accuracy >= float(threshold)),
                "reliable": bool(reliable),
            }
        )
    return pd.DataFrame(rows, columns=CLASSIFICATION_ACCURACY_COLUMNS).sort_values(["frequency", "model", "horizon"]).reset_index(drop=True)


def build_confidence_filter_summary(
    predictions: pd.DataFrame,
    *,
    enabled: bool,
    confidence_threshold: float,
    min_coverage_after_filter: float,
    threshold: float,
) -> pd.DataFrame:
    if not enabled or predictions.empty:
        return pd.DataFrame(columns=CONFIDENCE_FILTER_COLUMNS)
    working = predictions.copy()
    working["is_correct"] = pd.to_numeric(working["is_correct"], errors="coerce")
    working = working[working["is_correct"].isin([0, 1])].copy()
    if working.empty:
        return pd.DataFrame(columns=CONFIDENCE_FILTER_COLUMNS)
    if "filtered_out" not in working.columns:
        working["filtered_out"] = False
    filtered = _bool_summary_column(working, "filtered_out")
    rows: list[dict[str, Any]] = []
    for keys, group in working.groupby(["frequency", "model", "horizon"], sort=True):
        frequency, model_name, horizon = keys
        group_filtered = filtered.loc[group.index]
        total_rows = int(len(group))
        evaluated = group.loc[~group_filtered].copy()
        evaluated_rows = int(len(evaluated))
        coverage_ratio = evaluated_rows / total_rows if total_rows else 0.0
        unfiltered_accuracy = float(group["is_correct"].mean()) if total_rows else float("nan")
        filtered_accuracy = float(evaluated["is_correct"].mean()) if evaluated_rows else float("nan")
        coverage_ok = coverage_ratio >= float(min_coverage_after_filter)
        rows.append(
            {
                "frequency": str(frequency),
                "model": str(model_name),
                "horizon": int(horizon),
                "confidence_threshold": float(confidence_threshold),
                "total_rows": total_rows,
                "evaluated_rows": evaluated_rows,
                "coverage_ratio": coverage_ratio,
                "unfiltered_accuracy": unfiltered_accuracy,
                "filtered_accuracy": filtered_accuracy,
                "filtered_passed_60pct": bool(coverage_ok and evaluated_rows > 0 and filtered_accuracy >= float(threshold)),
                "min_coverage_after_filter": float(min_coverage_after_filter),
                "coverage_ok": bool(coverage_ok),
            }
        )
    return pd.DataFrame(rows, columns=CONFIDENCE_FILTER_COLUMNS).sort_values(["frequency", "model", "horizon"]).reset_index(drop=True)


def build_confidence_threshold_sweep_summary(
    predictions: pd.DataFrame,
    *,
    enabled: bool,
    thresholds: list[float],
    min_sweep_coverage: float,
    global_threshold: float,
    frequency: str = "hourly",
    model: str = "stacking",
    horizon: int = 1,
    target_mode: str = "classification",
) -> pd.DataFrame:
    if not enabled:
        return pd.DataFrame(columns=CONFIDENCE_THRESHOLD_SWEEP_COLUMNS)
    if predictions.empty:
        return pd.DataFrame(columns=CONFIDENCE_THRESHOLD_SWEEP_COLUMNS)

    working = predictions.copy()
    required = {"frequency", "model", "horizon", "target_mode", "confidence", "is_correct"}
    if not required <= set(working.columns):
        return pd.DataFrame(columns=CONFIDENCE_THRESHOLD_SWEEP_COLUMNS)
    working = working[
        working["frequency"].astype(str).str.lower().eq(str(frequency).lower())
        & working["model"].astype(str).str.lower().eq(str(model).lower())
        & (pd.to_numeric(working["horizon"], errors="coerce") == int(horizon))
        & working["target_mode"].astype(str).str.lower().eq(str(target_mode).lower())
    ].copy()
    if working.empty:
        return pd.DataFrame(columns=CONFIDENCE_THRESHOLD_SWEEP_COLUMNS)

    working["confidence"] = pd.to_numeric(working["confidence"], errors="coerce")
    working["is_correct"] = pd.to_numeric(working["is_correct"], errors="coerce")
    working = working[working["is_correct"].isin([0, 1])].copy()
    total_rows = int(len(working))
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        evaluated = working[working["confidence"].notna() & (working["confidence"] >= float(threshold))].copy()
        evaluated_rows = int(len(evaluated))
        coverage_ratio = evaluated_rows / total_rows if total_rows else 0.0
        filtered_accuracy = float(evaluated["is_correct"].mean()) if evaluated_rows else float("nan")
        coverage_ok = bool(coverage_ratio >= float(min_sweep_coverage))
        rows.append(
            {
                "frequency": str(frequency),
                "model": str(model),
                "horizon": int(horizon),
                "threshold": float(threshold),
                "total_rows": total_rows,
                "evaluated_rows": evaluated_rows,
                "coverage_ratio": coverage_ratio,
                "filtered_accuracy": filtered_accuracy,
                "passed_60pct": bool(coverage_ok and evaluated_rows > 0 and filtered_accuracy >= float(global_threshold)),
                "coverage_ok": coverage_ok,
                "selected_candidate": False,
            }
        )

    result = pd.DataFrame(rows, columns=CONFIDENCE_THRESHOLD_SWEEP_COLUMNS)
    eligible = result[result["coverage_ok"].astype(bool) & pd.to_numeric(result["filtered_accuracy"], errors="coerce").notna()].copy()
    if not eligible.empty:
        eligible = eligible.sort_values(
            ["filtered_accuracy", "coverage_ratio", "threshold"],
            ascending=[False, False, False],
        )
        selected_index = eligible.index[0]
        result.loc[selected_index, "selected_candidate"] = True
    return result.sort_values(["frequency", "model", "horizon", "threshold"]).reset_index(drop=True)


def _strategy_candidate_row(
    *,
    frequency: str,
    model: str,
    horizon: int,
    target_mode: str,
    regime: str,
    confidence_threshold: float | None,
    candidate_type: str,
    total_eligible_rows: int,
    n_obs: int,
    accuracy: float | None,
    threshold_60: float,
    selected_candidate: bool,
    selection_reason: str,
    min_obs_for_pass_63: int = 300,
    threshold_63: float = 0.63,
) -> dict[str, Any]:
    coverage_ratio = float(n_obs) / float(total_eligible_rows) if int(total_eligible_rows) > 0 else 0.0
    valid_accuracy = _finite_float_or_none(accuracy)
    pass_60 = bool(n_obs >= int(min_obs_for_pass_63) and valid_accuracy is not None and valid_accuracy >= float(threshold_60))
    pass_63 = bool(n_obs >= int(min_obs_for_pass_63) and valid_accuracy is not None and valid_accuracy >= float(threshold_63))
    pass_level = ""
    if pass_63:
        if candidate_type == "regime":
            pass_level = "regime_strategy_level"
        elif candidate_type == "confidence":
            pass_level = "confidence_strategy_level"
        else:
            pass_level = "strategy_level"
    return {
        "frequency": str(frequency),
        "model": str(model),
        "horizon": int(horizon),
        "target_mode": str(target_mode),
        "regime": str(regime),
        "confidence_threshold": confidence_threshold,
        "candidate_type": str(candidate_type),
        "total_eligible_rows": int(total_eligible_rows),
        "n_obs": int(n_obs),
        "evaluated_rows": int(n_obs),
        "coverage_ratio": coverage_ratio,
        "accuracy": valid_accuracy,
        "pass_60": pass_60,
        "pass_63": pass_63,
        "pass_level": pass_level,
        "selected_candidate": bool(selected_candidate),
        "selection_reason": str(selection_reason),
    }


def build_strategy_selection_summary(
    predictions: pd.DataFrame,
    confidence_sweep: pd.DataFrame,
    *,
    threshold: float,
    min_obs_for_pass_63: int = 300,
    threshold_63: float = 0.63,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=STRATEGY_SELECTION_COLUMNS)
    required = {"frequency", "model", "horizon", "target_mode", "is_correct"}
    if not required <= set(predictions.columns):
        return pd.DataFrame(columns=STRATEGY_SELECTION_COLUMNS)

    working = predictions.copy()
    working["is_correct"] = pd.to_numeric(working["is_correct"], errors="coerce")
    working = working[working["is_correct"].isin([0, 1])].copy()
    if working.empty:
        return pd.DataFrame(columns=STRATEGY_SELECTION_COLUMNS)

    group_columns = ["frequency", "model", "horizon", "target_mode"]
    group_totals: dict[tuple[str, str, int, str], int] = {}
    rows: list[dict[str, Any]] = []
    for keys, group in working.groupby(group_columns, sort=True):
        frequency, model_name, horizon, target_mode = keys
        key = (str(frequency), str(model_name), int(horizon), str(target_mode))
        total_rows = int(len(group))
        group_totals[key] = total_rows
        rows.append(
            _strategy_candidate_row(
                frequency=key[0],
                model=key[1],
                horizon=key[2],
                target_mode=key[3],
                regime="",
                confidence_threshold=None,
                candidate_type="unfiltered",
                total_eligible_rows=total_rows,
                n_obs=total_rows,
                accuracy=float(group["is_correct"].mean()) if total_rows else None,
                threshold_60=threshold,
                selected_candidate=False,
                selection_reason="unfiltered_model_horizon",
                min_obs_for_pass_63=min_obs_for_pass_63,
                threshold_63=threshold_63,
            )
        )

    if not confidence_sweep.empty:
        selected_sweep = confidence_sweep[_bool_summary_column(confidence_sweep, "selected_candidate")].copy()
        for row in selected_sweep.itertuples(index=False):
            row_dict = row._asdict()
            frequency = str(row_dict.get("frequency", ""))
            model_name = str(row_dict.get("model", ""))
            horizon = int(row_dict.get("horizon", 0))
            threshold_value = _finite_float_or_none(row_dict.get("threshold"))
            matching = working[
                working["frequency"].astype(str).eq(frequency)
                & working["model"].astype(str).eq(model_name)
                & (pd.to_numeric(working["horizon"], errors="coerce") == horizon)
            ].copy()
            if matching.empty or threshold_value is None or "confidence" not in matching.columns:
                continue
            target_modes = matching["target_mode"].astype(str).dropna().unique().tolist()
            target_mode = str(target_modes[0]) if target_modes else ""
            key = (frequency, model_name, horizon, target_mode)
            total_rows = int(group_totals.get(key, len(matching)))
            confidence = pd.to_numeric(matching["confidence"], errors="coerce")
            evaluated = matching[confidence.notna() & (confidence >= float(threshold_value))].copy()
            n_obs = int(len(evaluated))
            rows.append(
                _strategy_candidate_row(
                    frequency=frequency,
                    model=model_name,
                    horizon=horizon,
                    target_mode=target_mode,
                    regime="",
                    confidence_threshold=float(threshold_value),
                    candidate_type="confidence",
                    total_eligible_rows=total_rows,
                    n_obs=n_obs,
                    accuracy=float(evaluated["is_correct"].mean()) if n_obs else None,
                    threshold_60=threshold,
                    selected_candidate=True,
                    selection_reason="selected_confidence_threshold_sweep_candidate",
                    min_obs_for_pass_63=min_obs_for_pass_63,
                    threshold_63=threshold_63,
                )
            )

    regime_expanded = _expanded_regime_frame(working, entity_column="model")
    if not regime_expanded.empty:
        regime_expanded["is_correct"] = pd.to_numeric(regime_expanded["is_correct"], errors="coerce")
        regime_expanded = regime_expanded[regime_expanded["is_correct"].isin([0, 1])].copy()
        for keys, group in regime_expanded.groupby(["frequency", "model", "horizon", "target_mode", "regime_bucket"], sort=True):
            frequency, model_name, horizon, target_mode, regime = keys
            key = (str(frequency), str(model_name), int(horizon), str(target_mode))
            total_rows = int(group_totals.get(key, 0))
            n_obs = int(len(group))
            accuracy = float(group["is_correct"].mean()) if n_obs else None
            candidate = _strategy_candidate_row(
                frequency=key[0],
                model=key[1],
                horizon=key[2],
                target_mode=key[3],
                regime=str(regime),
                confidence_threshold=None,
                candidate_type="regime",
                total_eligible_rows=total_rows,
                n_obs=n_obs,
                accuracy=accuracy,
                threshold_60=threshold,
                selected_candidate=False,
                selection_reason="regime_filtered_candidate",
                min_obs_for_pass_63=min_obs_for_pass_63,
                threshold_63=threshold_63,
            )
            if candidate["pass_63"]:
                candidate["selected_candidate"] = True
            rows.append(candidate)

    result = pd.DataFrame(rows, columns=STRATEGY_SELECTION_COLUMNS)
    if result.empty:
        return pd.DataFrame(columns=STRATEGY_SELECTION_COLUMNS)
    result["_sort_selected"] = _bool_summary_column(result, "selected_candidate").astype(int)
    result["_sort_pass63"] = _bool_summary_column(result, "pass_63").astype(int)
    result["_sort_accuracy"] = pd.to_numeric(result["accuracy"], errors="coerce").fillna(-1.0)
    result["_sort_coverage"] = pd.to_numeric(result["coverage_ratio"], errors="coerce").fillna(-1.0)
    result = result.sort_values(
        ["_sort_selected", "_sort_pass63", "_sort_accuracy", "_sort_coverage", "frequency", "model", "horizon", "regime"],
        ascending=[False, False, False, False, True, True, True, True],
    )
    return result[STRATEGY_SELECTION_COLUMNS].reset_index(drop=True)


def _confidence_sweep_summary_fields(sweep_summary: pd.DataFrame, *, enabled: bool) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "confidence_sweep_enabled": bool(enabled),
        "best_confidence_threshold": None,
        "best_confidence_sweep_accuracy": None,
        "best_confidence_sweep_coverage": None,
        "confidence_sweep_passed_60pct": False,
    }
    if not enabled or sweep_summary.empty:
        return fields
    selected = sweep_summary[_bool_summary_column(sweep_summary, "selected_candidate")].copy()
    if selected.empty:
        return fields
    row = selected.iloc[0]
    fields.update(
        {
            "best_confidence_threshold": _finite_float_or_none(row.get("threshold")),
            "best_confidence_sweep_accuracy": _finite_float_or_none(row.get("filtered_accuracy")),
            "best_confidence_sweep_coverage": _finite_float_or_none(row.get("coverage_ratio")),
            "confidence_sweep_passed_60pct": bool(row.get("passed_60pct", False)),
        }
    )
    return fields


def _diagnostic_summary_fields(
    significance_summary: pd.DataFrame,
    baseline_delta_summary: pd.DataFrame,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "best_model_frequency": None,
        "best_model_horizon": None,
        "best_model_accuracy": None,
        "best_baseline_accuracy": None,
        "best_model_delta_vs_best_baseline": None,
        "any_model_significant_vs_50pct": False,
    }

    best_model_accuracy: float | None = None
    if not significance_summary.empty:
        significance = significance_summary.copy()
        significance["accuracy"] = pd.to_numeric(significance["accuracy"], errors="coerce")
        significance["n_obs"] = pd.to_numeric(significance["n_obs"], errors="coerce").fillna(0).astype(int)
        significance = significance[(significance["n_obs"] > 0) & significance["accuracy"].notna()].copy()
        if not significance.empty:
            best = significance.sort_values(["accuracy", "n_obs"], ascending=[False, False]).iloc[0]
            best_model_accuracy = _finite_float_or_none(best["accuracy"])
            fields.update(
                {
                    "best_model_frequency": str(best["frequency"]),
                    "best_model_horizon": int(best["horizon"]),
                    "best_model_accuracy": best_model_accuracy,
                    "any_model_significant_vs_50pct": bool(
                        significance_summary["significant_at_5pct"].fillna(False).astype(bool).any()
                    ),
                }
            )

    best_baseline_accuracy: float | None = None
    if not baseline_delta_summary.empty:
        baseline_accuracy = pd.to_numeric(baseline_delta_summary["baseline_accuracy"], errors="coerce").dropna()
        if not baseline_accuracy.empty:
            best_baseline_accuracy = float(baseline_accuracy.max())
            fields["best_baseline_accuracy"] = best_baseline_accuracy

    if best_model_accuracy is not None and best_baseline_accuracy is not None:
        fields["best_model_delta_vs_best_baseline"] = best_model_accuracy - best_baseline_accuracy

    return fields


def _regime_summary_fields(regime_accuracy_summary: pd.DataFrame) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "best_regime": None,
        "best_regime_model": None,
        "best_regime_horizon": None,
        "best_regime_accuracy": None,
    }
    if regime_accuracy_summary.empty:
        return fields
    working = regime_accuracy_summary.copy()
    working["accuracy"] = pd.to_numeric(working["accuracy"], errors="coerce")
    working["n_obs"] = pd.to_numeric(working["n_obs"], errors="coerce").fillna(0).astype(int)
    working = working[(working["n_obs"] > 0) & working["accuracy"].notna()].copy()
    if working.empty:
        return fields
    best = working.sort_values(["accuracy", "n_obs"], ascending=[False, False]).iloc[0]
    return {
        "best_regime": str(best["regime"]),
        "best_regime_model": str(best["model"]),
        "best_regime_horizon": int(best["horizon"]),
        "best_regime_accuracy": float(best["accuracy"]),
    }


def _reliability_counts(accuracy_summary: pd.DataFrame) -> tuple[int, int]:
    if accuracy_summary.empty or "reliable" not in accuracy_summary.columns:
        return 0, 0
    reliable = accuracy_summary["reliable"].astype(bool)
    return int(reliable.sum()), int((~reliable).sum())


def _effective_time_bounds(frame: pd.DataFrame, *, time_column: str, start: str, end: str) -> tuple[str, str]:
    if frame.empty or time_column not in frame.columns:
        return "", ""
    timestamps = pd.to_datetime(frame[time_column], errors="coerce")
    mask = (timestamps >= _to_date(start)) & (timestamps <= _to_end_of_day(end))
    values = timestamps.loc[mask].dropna()
    if values.empty:
        return "", ""
    return _safe_iso_date(values.min()), _safe_iso_date(values.max())


def _build_benchmark_summary(
    *,
    frequency: str,
    provider: str,
    universe: str,
    raw_data_start: str,
    raw_data_end: str,
    initial_train_start: str,
    initial_train_end: str,
    evaluation_start: str,
    evaluation_end: str,
    threshold: float,
    predictions: pd.DataFrame,
    accuracy_summary: pd.DataFrame,
    min_obs_per_group: int,
    data_gap_warnings: list[dict[str, Any]],
    effective_train_start: str = "",
    effective_train_end: str = "",
    effective_eval_start: str = "",
    effective_eval_end: str = "",
) -> dict[str, Any]:
    n_predictions = int(len(predictions))
    overall_accuracy = float(pd.to_numeric(predictions["is_correct"], errors="coerce").mean()) if n_predictions else 0.0
    reliable_group_count, unreliable_group_count = _reliability_counts(accuracy_summary)
    return {
        "frequency": frequency,
        "provider": provider,
        "universe": universe,
        "raw_data_start": raw_data_start,
        "raw_data_end": raw_data_end,
        "initial_train_start": initial_train_start,
        "initial_train_end": initial_train_end,
        "evaluation_start": evaluation_start,
        "evaluation_end": evaluation_end,
        "effective_train_start": effective_train_start,
        "effective_train_end": effective_train_end,
        "effective_eval_start": effective_eval_start,
        "effective_eval_end": effective_eval_end,
        "threshold": float(threshold),
        "overall_accuracy": overall_accuracy,
        "n_predictions": n_predictions,
        "passed": bool(n_predictions > 0 and overall_accuracy >= float(threshold)),
        "fetch_ok_count": 0,
        "fetch_partial_count": 0,
        "fetch_failed_count": 0,
        "full_range_valid_pairs": 0,
        "partial_usable_pairs": 0,
        "unusable_pairs": 0,
        "partial_cache_allowed": False,
        "data_gap_warning_count": int(len(data_gap_warnings)),
        "data_gap_warnings": data_gap_warnings[:50],
        "min_obs_per_group": int(min_obs_per_group),
        "reliable_group_count": reliable_group_count,
        "unreliable_group_count": unreliable_group_count,
        "evaluated_tickers": sorted(predictions["ticker"].dropna().astype(str).unique().tolist()) if n_predictions else [],
        "evaluated_models": sorted(predictions["model"].dropna().astype(str).unique().tolist()) if n_predictions else [],
        "evaluated_horizons": sorted(int(value) for value in predictions["horizon"].dropna().unique().tolist()) if n_predictions else [],
        "target_mode": "regression",
        "regime_evaluation_enabled": False,
        "best_regime": None,
        "best_regime_model": None,
        "best_regime_horizon": None,
        "best_regime_accuracy": None,
        "confidence_filter_enabled": False,
        "confidence_sweep_enabled": False,
        "best_confidence_threshold": None,
        "best_confidence_sweep_accuracy": None,
        "best_confidence_sweep_coverage": None,
        "confidence_sweep_passed_60pct": False,
        "horizon_tuning_enabled": False,
        "tuned_models": [],
        "tuning_trials": 0,
        "tuning_summary_path": "",
        "evaluation_type": EVALUATION_TYPE,
    }


def git_commit_if_available() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _fetch_status_counts(fetch_summary: pd.DataFrame, *, frequency: str | None = None) -> dict[str, int]:
    frame = fetch_summary.copy()
    if frequency is not None and "frequency" in frame.columns:
        frame = frame[frame["frequency"].astype(str).str.lower() == frequency.lower()].copy()
    if frame.empty or "status" not in frame.columns:
        return {"fetch_ok_count": 0, "fetch_partial_count": 0, "fetch_failed_count": 0}
    status = frame["status"].astype(str).str.lower().str.strip()
    ok_mask = status.eq("ok")
    partial_mask = status.eq("partial")
    failed_mask = status.eq("failed")
    return {
        "fetch_ok_count": int(ok_mask.sum()),
        "fetch_partial_count": int(partial_mask.sum()),
        "fetch_failed_count": int(failed_mask.sum()),
    }


def _bool_summary_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.lower().str.strip().isin({"true", "1", "yes", "y"})


def _cache_pair_summary_fields(
    fetch_summary: pd.DataFrame,
    *,
    config: BenchmarkConfig,
    frequency: str | None = None,
) -> dict[str, Any]:
    frame = fetch_summary.copy()
    if frequency is not None and "frequency" in frame.columns:
        frame = frame[frame["frequency"].astype(str).str.lower() == frequency.lower()].copy()
    if frame.empty or "status" not in frame.columns:
        return {
            "full_range_valid_pairs": 0,
            "partial_usable_pairs": 0,
            "unusable_pairs": 0,
            "partial_cache_allowed": bool(config.allow_partial_cache_for_benchmark),
        }
    status = frame["status"].astype(str).str.lower().str.strip()
    full_range_valid = status.eq("ok")
    partial_usable = status.eq("partial") & _bool_summary_column(frame, "benchmark_usable")
    unusable = ~(full_range_valid | partial_usable)
    return {
        "full_range_valid_pairs": int(full_range_valid.sum()),
        "partial_usable_pairs": int(partial_usable.sum()),
        "unusable_pairs": int(unusable.sum()),
        "partial_cache_allowed": bool(config.allow_partial_cache_for_benchmark),
    }


def _run_mode_metadata(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "cache_only": bool(config.cache_only),
        "fetch_only": bool(config.fetch_only),
        "resume_fetch": bool(config.resume_fetch),
        "provider_calls_allowed": bool(config.provider_calls_allowed),
        "checkpointing_enabled": bool(config.checkpointing_enabled),
        "partial_cache_allowed": bool(config.allow_partial_cache_for_benchmark),
        "target_mode": config.target_mode,
        "regime_evaluation_enabled": bool(config.enable_regime_evaluation),
        "confidence_filter_enabled": bool(config.enable_confidence_filter),
        "confidence_sweep_enabled": bool(config.enable_confidence_threshold_sweep),
        "horizon_tuning_enabled": bool(config.enable_horizon_tuning),
        "provider_timeout_seconds": float(config.provider_timeout_seconds),
        "model_fit_timeout_seconds": float(config.model_fit_timeout_seconds),
    }


def write_outputs(
    *,
    output_dir: Path,
    config: BenchmarkConfig,
    daily_outputs: tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    hourly_outputs: tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    fetch_summary: pd.DataFrame,
    source_health_summary: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_dir = output_dir / "daily"
    hourly_dir = output_dir / "hourly"
    daily_dir.mkdir(parents=True, exist_ok=True)
    hourly_dir.mkdir(parents=True, exist_ok=True)

    daily_predictions, daily_accuracy, daily_summary, daily_baseline, daily_model_errors, daily_baseline_predictions, daily_tuning = daily_outputs
    hourly_predictions, hourly_accuracy, hourly_summary, hourly_baseline, hourly_model_errors, hourly_baseline_predictions, hourly_tuning = hourly_outputs
    model_error_summary = pd.concat([daily_model_errors, hourly_model_errors], ignore_index=True)
    if model_error_summary.empty:
        model_error_summary = pd.DataFrame(columns=MODEL_ERROR_COLUMNS)
    tuning_summary = pd.concat([daily_tuning, hourly_tuning], ignore_index=True)
    if tuning_summary.empty:
        tuning_summary = pd.DataFrame(columns=TUNING_SUMMARY_COLUMNS)
    tuning_output_dir = Path(config.tuning_output_dir) if config.tuning_output_dir else output_dir
    tuning_summary_path = tuning_output_dir / "tuning_summary.csv"
    tuning_summary_label = str(tuning_summary_path.relative_to(output_dir)) if tuning_summary_path.is_relative_to(output_dir) else str(tuning_summary_path)
    daily_baseline_delta = build_baseline_delta_summary(daily_accuracy, daily_baseline)
    hourly_baseline_delta = build_baseline_delta_summary(hourly_accuracy, hourly_baseline)
    daily_mcnemar = build_mcnemar_summary(daily_predictions, daily_baseline_predictions)
    hourly_mcnemar = build_mcnemar_summary(hourly_predictions, hourly_baseline_predictions)
    daily_significance = build_significance_summary(
        daily_predictions,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
    )
    hourly_significance = build_significance_summary(
        hourly_predictions,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed + 100_000,
    )
    daily_regime_accuracy = build_regime_accuracy_summary(daily_predictions, config.threshold, config.min_obs_per_group)
    hourly_regime_accuracy = build_regime_accuracy_summary(hourly_predictions, config.threshold, config.min_obs_per_group)
    daily_regime_baseline_delta = build_regime_baseline_delta_summary(daily_predictions, daily_baseline_predictions)
    hourly_regime_baseline_delta = build_regime_baseline_delta_summary(hourly_predictions, hourly_baseline_predictions)
    daily_classification_accuracy = build_classification_accuracy_summary(
        daily_predictions,
        threshold=config.threshold,
        min_obs_per_group=config.min_obs_per_group,
    )
    hourly_classification_accuracy = build_classification_accuracy_summary(
        hourly_predictions,
        threshold=config.threshold,
        min_obs_per_group=config.min_obs_per_group,
    )
    daily_confidence_filter = build_confidence_filter_summary(
        daily_predictions,
        enabled=config.enable_confidence_filter,
        confidence_threshold=config.confidence_threshold,
        min_coverage_after_filter=config.min_coverage_after_filter,
        threshold=config.threshold,
    )
    hourly_confidence_filter = build_confidence_filter_summary(
        hourly_predictions,
        enabled=config.enable_confidence_filter,
        confidence_threshold=config.confidence_threshold,
        min_coverage_after_filter=config.min_coverage_after_filter,
        threshold=config.threshold,
    )
    daily_confidence_sweep = build_confidence_threshold_sweep_summary(
        daily_predictions,
        enabled=config.enable_confidence_threshold_sweep,
        thresholds=config.confidence_threshold_grid,
        min_sweep_coverage=config.min_sweep_coverage,
        global_threshold=config.threshold,
    )
    hourly_confidence_sweep = build_confidence_threshold_sweep_summary(
        hourly_predictions,
        enabled=config.enable_confidence_threshold_sweep,
        thresholds=config.confidence_threshold_grid,
        min_sweep_coverage=config.min_sweep_coverage,
        global_threshold=config.threshold,
    )
    daily_strategy_selection = build_strategy_selection_summary(
        daily_predictions,
        daily_confidence_sweep,
        threshold=config.threshold,
    )
    hourly_strategy_selection = build_strategy_selection_summary(
        hourly_predictions,
        hourly_confidence_sweep,
        threshold=config.threshold,
    )
    daily_summary.update(_fetch_status_counts(fetch_summary))
    daily_summary.update(_cache_pair_summary_fields(fetch_summary, config=config))
    daily_summary["fetch_scope"] = "daily_and_hourly_raw_inputs"
    daily_summary["model_error_count"] = int(len(daily_model_errors))
    daily_summary["tuning_summary_path"] = tuning_summary_label if config.enable_horizon_tuning else ""
    daily_summary.update(_diagnostic_summary_fields(daily_significance, daily_baseline_delta))
    daily_summary.update(_regime_summary_fields(daily_regime_accuracy if config.enable_regime_evaluation else pd.DataFrame()))
    daily_summary.update(_confidence_sweep_summary_fields(daily_confidence_sweep, enabled=config.enable_confidence_threshold_sweep))
    hourly_summary.update(_fetch_status_counts(fetch_summary, frequency="hourly"))
    hourly_summary.update(_cache_pair_summary_fields(fetch_summary, config=config, frequency="hourly"))
    hourly_summary["fetch_scope"] = "hourly_raw_inputs"
    hourly_summary["model_error_count"] = int(len(hourly_model_errors))
    hourly_summary["tuning_summary_path"] = tuning_summary_label if config.enable_horizon_tuning else ""
    hourly_summary.update(_diagnostic_summary_fields(hourly_significance, hourly_baseline_delta))
    hourly_summary.update(_regime_summary_fields(hourly_regime_accuracy if config.enable_regime_evaluation else pd.DataFrame()))
    hourly_summary.update(_confidence_sweep_summary_fields(hourly_confidence_sweep, enabled=config.enable_confidence_threshold_sweep))
    daily_predictions.to_csv(daily_dir / "predicted_vs_actual.csv", index=False)
    daily_accuracy.to_csv(daily_dir / "accuracy_summary.csv", index=False)
    daily_baseline.to_csv(daily_dir / "baseline_summary.csv", index=False)
    daily_baseline_delta.to_csv(daily_dir / "baseline_delta_summary.csv", index=False)
    daily_significance.to_csv(daily_dir / "significance_summary.csv", index=False)
    daily_mcnemar.to_csv(daily_dir / "mcnemar_summary.csv", index=False)
    daily_regime_accuracy.to_csv(daily_dir / "regime_accuracy_summary.csv", index=False)
    daily_regime_baseline_delta.to_csv(daily_dir / "regime_baseline_delta_summary.csv", index=False)
    daily_classification_accuracy.to_csv(daily_dir / "classification_accuracy_summary.csv", index=False)
    daily_confidence_filter.to_csv(daily_dir / "confidence_filter_summary.csv", index=False)
    daily_confidence_sweep.to_csv(daily_dir / "confidence_threshold_sweep_summary.csv", index=False)
    daily_strategy_selection.to_csv(daily_dir / "strategy_selection_summary.csv", index=False)
    write_json(daily_dir / "benchmark_summary.json", daily_summary)
    hourly_predictions.to_csv(hourly_dir / "predicted_vs_actual.csv", index=False)
    hourly_accuracy.to_csv(hourly_dir / "accuracy_summary.csv", index=False)
    hourly_baseline.to_csv(hourly_dir / "baseline_summary.csv", index=False)
    hourly_baseline_delta.to_csv(hourly_dir / "baseline_delta_summary.csv", index=False)
    hourly_significance.to_csv(hourly_dir / "significance_summary.csv", index=False)
    hourly_mcnemar.to_csv(hourly_dir / "mcnemar_summary.csv", index=False)
    hourly_regime_accuracy.to_csv(hourly_dir / "regime_accuracy_summary.csv", index=False)
    hourly_regime_baseline_delta.to_csv(hourly_dir / "regime_baseline_delta_summary.csv", index=False)
    hourly_classification_accuracy.to_csv(hourly_dir / "classification_accuracy_summary.csv", index=False)
    hourly_confidence_filter.to_csv(hourly_dir / "confidence_filter_summary.csv", index=False)
    hourly_confidence_sweep.to_csv(hourly_dir / "confidence_threshold_sweep_summary.csv", index=False)
    hourly_strategy_selection.to_csv(hourly_dir / "strategy_selection_summary.csv", index=False)
    write_json(hourly_dir / "benchmark_summary.json", hourly_summary)
    fetch_summary.to_csv(output_dir / "fetch_summary.csv", index=False)
    build_usable_cache_summary(fetch_summary).to_csv(output_dir / "usable_cache_summary.csv", index=False)
    source_health_summary.to_csv(output_dir / "source_health_summary.csv", index=False)
    model_error_summary.to_csv(output_dir / "model_error_summary.csv", index=False)
    sweep_parts = [frame for frame in (daily_confidence_sweep, hourly_confidence_sweep) if not frame.empty]
    combined_confidence_sweep = pd.concat(sweep_parts, ignore_index=True) if sweep_parts else pd.DataFrame(columns=CONFIDENCE_THRESHOLD_SWEEP_COLUMNS)
    if combined_confidence_sweep.empty:
        combined_confidence_sweep = pd.DataFrame(columns=CONFIDENCE_THRESHOLD_SWEEP_COLUMNS)
    combined_confidence_sweep.to_csv(output_dir / "confidence_threshold_sweep_summary.csv", index=False)
    strategy_parts = [frame for frame in (daily_strategy_selection, hourly_strategy_selection) if not frame.empty]
    combined_strategy_selection = pd.concat(strategy_parts, ignore_index=True) if strategy_parts else pd.DataFrame(columns=STRATEGY_SELECTION_COLUMNS)
    if combined_strategy_selection.empty:
        combined_strategy_selection = pd.DataFrame(columns=STRATEGY_SELECTION_COLUMNS)
    combined_strategy_selection.to_csv(output_dir / "strategy_selection_summary.csv", index=False)
    tuning_output_dir.mkdir(parents=True, exist_ok=True)
    tuning_summary.to_csv(tuning_summary_path, index=False)
    write_json(output_dir / "run_config.json", _run_config_payload(config))

    manifest = {
        **_run_mode_metadata(config),
        "provider": config.provider,
        "universe": config.universe,
        "raw_daily_range": {"start": config.daily_start, "end": config.daily_end},
        "raw_hourly_range": {"start": config.hourly_start, "end": config.hourly_end},
        "evaluation_range": {"start": config.eval_start, "end": config.eval_end},
        "train_cutoff": config.train_cutoff or "",
        "data_end": _config_data_end(config),
        "training_label_cutoff_rule": _training_label_cutoff_rule(config),
        "actual_rows_allowed_after_train_cutoff": True,
        "daily_benchmark_method": (
            "Daily OHLCV 2006-2015 plus hourly OHLCV 2016+ resampled to daily; "
            "expanding-window out-of-sample evaluation with scheduled retraining."
        ),
        "hourly_benchmark_method": (
            "Hourly OHLCV 2016+ evaluated separately with expanding-window out-of-sample "
            "evaluation and scheduled retraining."
        ),
        "partial_cache_policy": (
            "Partial cache rows are benchmark inputs only when strict coverage is partial, "
            "benchmark_usable is true, and --allow-partial-cache-for-benchmark is set; "
            "no provider fetch or synthetic history is used in cache-only mode."
        ),
        "threshold": float(config.threshold),
        "min_obs_per_group": int(config.min_obs_per_group),
        "max_daily_gap_days": int(config.max_daily_gap_days),
        "rate_limit_per_minute": int(config.rate_limit_per_minute),
        "request_sleep_seconds": config.request_sleep_seconds,
        "max_fetch_retries": int(config.max_fetch_retries),
        "fetch_batch_size": int(config.fetch_batch_size),
        "fetch_batch_cooldown_seconds": float(config.fetch_batch_cooldown_seconds),
        "source_failure_threshold": int(config.source_failure_threshold),
        "source_empty_threshold": int(config.source_empty_threshold),
        "source_cooldown_seconds": float(config.source_cooldown_seconds),
        "min_provider_daily_rows": int(config.min_provider_daily_rows),
        "min_provider_hourly_rows": int(config.min_provider_hourly_rows),
        "coverage_start_tolerance_days": int(config.coverage_start_tolerance_days),
        "coverage_end_tolerance_days": int(config.coverage_end_tolerance_days),
        "min_coverage_ratio": float(config.min_coverage_ratio),
        "min_pre_eval_rows_daily": int(config.min_pre_eval_rows_daily),
        "min_pre_eval_rows_hourly": int(config.min_pre_eval_rows_hourly),
        "min_eval_rows_daily": int(config.min_eval_rows_daily),
        "min_eval_rows_hourly": int(config.min_eval_rows_hourly),
        "target_mode": config.target_mode,
        "regime_evaluation_enabled": bool(config.enable_regime_evaluation),
        "regime_return_window": int(config.regime_return_window),
        "regime_vol_window": int(config.regime_vol_window),
        "regime_bull_threshold": float(config.regime_bull_threshold),
        "regime_bear_threshold": float(config.regime_bear_threshold),
        "regime_vol_quantile": float(config.regime_vol_quantile),
        "confidence_filter_enabled": bool(config.enable_confidence_filter),
        "confidence_threshold": float(config.confidence_threshold),
        "confidence_threshold_sweep_enabled": bool(config.enable_confidence_threshold_sweep),
        "confidence_threshold_grid": list(config.confidence_threshold_grid),
        "min_sweep_coverage": float(config.min_sweep_coverage),
        "strategy_selection_summary_path": "strategy_selection_summary.csv",
        "no_trade_band": float(config.no_trade_band),
        "min_coverage_after_filter": float(config.min_coverage_after_filter),
        "horizon_tuning_enabled": bool(config.enable_horizon_tuning),
        "tuned_models": list(config.tuning_models),
        "tuning_trials": int(config.tuning_trials),
        "tuning_metric": config.tuning_metric,
        "tuning_summary_path": tuning_summary_label,
        "effective_training_range": {
            "daily": {
                "start": daily_summary.get("effective_train_start", ""),
                "end": daily_summary.get("effective_train_end", ""),
            },
            "hourly": {
                "start": hourly_summary.get("effective_train_start", ""),
                "end": hourly_summary.get("effective_train_end", ""),
            },
        },
        "effective_evaluation_range": {
            "daily": {
                "start": daily_summary.get("effective_eval_start", ""),
                "end": daily_summary.get("effective_eval_end", ""),
            },
            "hourly": {
                "start": hourly_summary.get("effective_eval_start", ""),
                "end": hourly_summary.get("effective_eval_end", ""),
            },
        },
        "cache_pair_summary": {
            "daily_benchmark_scope": _cache_pair_summary_fields(fetch_summary, config=config),
            "hourly_benchmark_scope": _cache_pair_summary_fields(fetch_summary, config=config, frequency="hourly"),
        },
        "bootstrap_samples": int(config.bootstrap_samples),
        "bootstrap_seed": int(config.bootstrap_seed),
        "daily_data_gap_warning_count": int(daily_summary.get("data_gap_warning_count", 0)),
        "hourly_data_gap_warning_count": int(hourly_summary.get("data_gap_warning_count", 0)),
        "data_gap_warnings": {
            "daily": daily_summary.get("data_gap_warnings", [])[:50],
            "hourly": hourly_summary.get("data_gap_warnings", [])[:50],
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script_name": Path(__file__).name,
        "git_commit_if_available": git_commit_if_available(),
        "evaluation_type": EVALUATION_TYPE,
    }
    write_json(output_dir / "manifest.json", manifest)


def write_fetch_only_outputs(
    *,
    output_dir: Path,
    config: BenchmarkConfig,
    fetch_summary: pd.DataFrame,
    source_health_summary: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fetch_summary.to_csv(output_dir / "fetch_summary.csv", index=False)
    build_usable_cache_summary(fetch_summary).to_csv(output_dir / "usable_cache_summary.csv", index=False)
    source_health_summary.to_csv(output_dir / "source_health_summary.csv", index=False)
    write_json(output_dir / "run_config.json", _run_config_payload(config))
    manifest = {
        **_run_mode_metadata(config),
        "provider": config.provider,
        "universe": config.universe,
        "raw_daily_range": {"start": config.daily_start, "end": config.daily_end},
        "raw_hourly_range": {"start": config.hourly_start, "end": config.hourly_end},
        "train_cutoff": config.train_cutoff or "",
        "data_end": _config_data_end(config),
        "evaluation_range": {"start": config.eval_start, "end": config.eval_end},
        "training_label_cutoff_rule": _training_label_cutoff_rule(config),
        "actual_rows_allowed_after_train_cutoff": True,
        "fetch_status_counts": _fetch_status_counts(fetch_summary),
        "daily_fetch_status_counts": _fetch_status_counts(fetch_summary, frequency="daily"),
        "hourly_fetch_status_counts": _fetch_status_counts(fetch_summary, frequency="hourly"),
        "cache_pair_summary": _cache_pair_summary_fields(fetch_summary, config=config),
        "hourly_cache_pair_summary": _cache_pair_summary_fields(fetch_summary, config=config, frequency="hourly"),
        "min_pre_eval_rows_daily": int(config.min_pre_eval_rows_daily),
        "min_pre_eval_rows_hourly": int(config.min_pre_eval_rows_hourly),
        "min_eval_rows_daily": int(config.min_eval_rows_daily),
        "min_eval_rows_hourly": int(config.min_eval_rows_hourly),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script_name": Path(__file__).name,
        "git_commit_if_available": git_commit_if_available(),
        "evaluation_type": EVALUATION_TYPE,
    }
    write_json(output_dir / "manifest.json", manifest)


def main() -> None:
    config = parse_args()
    np.random.seed(config.seed)
    tickers = resolve_tickers(config)
    if not config.cache_only:
        _write_universe_cache(tickers, config)
    daily_frames, hourly_frames, fetch_summary, source_health_summary = load_market_data(config=config, tickers=tickers)
    if config.fetch_only:
        write_fetch_only_outputs(
            output_dir=Path(config.output_dir),
            config=config,
            fetch_summary=fetch_summary,
            source_health_summary=source_health_summary,
        )
        counts = _fetch_status_counts(fetch_summary)
        print(
            "VN100 hybrid-frequency fetch complete: "
            f"ok={counts['fetch_ok_count']} "
            f"partial={counts['fetch_partial_count']} "
            f"failed={counts['fetch_failed_count']} "
            f"output_dir={config.output_dir}"
        )
        return

    daily_raw = combine_daily_inputs(daily_frames, hourly_frames, config)
    hourly_raw = combine_hourly_inputs(hourly_frames, config)
    initial_train_end = config.train_cutoff or str((_to_date(config.eval_start) - pd.Timedelta(days=1)).date())

    daily_outputs = run_frequency_benchmark(
        raw_df=daily_raw,
        frequency="daily",
        horizons=config.daily_horizons,
        models=config.models,
        initial_train_start=config.daily_start,
        initial_train_end=initial_train_end,
        eval_start=config.eval_start,
        eval_end=config.eval_end,
        threshold=config.threshold,
        provider=config.provider,
        universe=config.universe,
        retrain_frequency=config.retrain_frequency,
        seed=config.seed,
        min_history_days=config.min_history_days,
        min_obs_per_group=config.min_obs_per_group,
        max_daily_gap_days=config.max_daily_gap_days,
        config=config,
    )
    hourly_outputs = run_frequency_benchmark(
        raw_df=hourly_raw,
        frequency="hourly",
        horizons=config.hourly_horizons,
        models=config.models,
        initial_train_start=config.hourly_start,
        initial_train_end=initial_train_end,
        eval_start=config.eval_start,
        eval_end=config.eval_end,
        threshold=config.threshold,
        provider=config.provider,
        universe=config.universe,
        retrain_frequency=config.retrain_frequency,
        seed=config.seed,
        min_history_days=config.min_history_days,
        min_obs_per_group=config.min_obs_per_group,
        max_daily_gap_days=config.max_daily_gap_days,
        config=config,
    )
    write_outputs(
        output_dir=Path(config.output_dir),
        config=config,
        daily_outputs=daily_outputs,
        hourly_outputs=hourly_outputs,
        fetch_summary=fetch_summary,
        source_health_summary=source_health_summary,
    )

    daily_summary = daily_outputs[2]
    hourly_summary = hourly_outputs[2]
    print(
        "VN100 hybrid-frequency benchmark complete: "
        f"daily_accuracy={daily_summary['overall_accuracy']:.4f} "
        f"daily_n={daily_summary['n_predictions']} "
        f"hourly_accuracy={hourly_summary['overall_accuracy']:.4f} "
        f"hourly_n={hourly_summary['n_predictions']} "
        f"output_dir={config.output_dir}"
    )


if __name__ == "__main__":
    main()
