"""Run the VN30 hourly 2015 Jan-2025 benchmark from validated gateway cache.

This runner intentionally avoids the older VN30 2005/2026 helper constants
because that track contains a superseded ticker universe. It uses the shared
benchmark engine for model and baseline logic only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


ACTIVE_UNIVERSE_SOURCE = "VN30 January 2025 review"
ACTIVE_UNIVERSE_SOURCE_RAW = "hose_january_2025_review"
UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
READINESS_ROOT = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015_benchmark_readiness"
READINESS_MANIFEST_PATH = READINESS_ROOT / "vn30_2015_benchmark_readiness_manifest.json"
READINESS_REPORT_PATH = READINESS_ROOT / "vn30_2015_benchmark_readiness_report.md"
STOCK_VALIDATION_PATH = (
    REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015" / "validation" / "vn30_hourly_2015_validation.csv"
)
INDEX_VALIDATION_PATH = (
    REPO_ROOT / "reports" / "generated" / "index_hourly_2015" / "validation" / "index_hourly_2015_validation.csv"
)
STOCK_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly_2015"
INDEX_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly_2015"
REPORT_ROOT = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015_benchmark"
LOCK_JSON_PATH = REPORT_ROOT / "pre_benchmark_readiness_lock.json"
LOCK_MD_PATH = REPORT_ROOT / "pre_benchmark_readiness_lock.md"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_2015_jan2025_benchmark"

TRAIN_START_TEXT = "2015-01-01 00:00:00"
TRAIN_CUTOFF_TEXT = "2024-12-31 23:59:59"
EVAL_START_TEXT = "2025-01-01 00:00:00"
FREQUENCY_TEXT = "1H"
TARGET_MODE = "classification"
THRESHOLD = 0.60
DEFAULT_MODELS = ["xgboost", "lightgbm", "random_forest", "stacking"]
DEFAULT_HORIZONS = [1, 4, 8, 20]
EXPECTED_INCLUDED = {"BCM", "BVH"}
EXPECTED_EXCLUDED = {"BSR", "DGC", "VPL"}
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
MODEL_INPUT_COLUMNS = ["datetime", "ticker", *OHLCV_COLUMNS]
SOURCE_HEALTH_COLUMNS = [
    "symbol",
    "asset_type",
    "cache_path",
    "first_datetime",
    "last_datetime",
    "row_count",
    "training_rows_before_2025",
    "evaluation_rows_from_2025",
    "frequency_1h",
    "usable",
    "used_in_model",
    "readiness_scope",
]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_csv_list(value: str) -> list[str]:
    return [item.strip().lower() for item in str(value).split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in parse_csv_list(value)]


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def read_universe_rows() -> list[dict[str, str]]:
    rows = read_csv_rows(UNIVERSE_PATH)
    required = {"ticker", "index", "source", "effective_period", "note"}
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Frozen VN30 universe is missing columns: {sorted(missing)}")
    return rows


def universe_tickers(rows: list[dict[str, str]]) -> list[str]:
    return [str(row.get("ticker", "")).strip().upper() for row in rows if str(row.get("ticker", "")).strip()]


def eval_end_from_readiness(manifest: dict[str, Any], stock_rows: list[dict[str, str]]) -> str:
    common_latest = str(manifest.get("common_latest_usable_data_timestamp", "")).strip()
    if common_latest:
        return common_latest
    timestamps = [
        pd.to_datetime(row.get("last_datetime", ""), errors="coerce")
        for row in stock_rows
        if as_bool(row.get("usable"))
    ]
    timestamps = [pd.Timestamp(value) for value in timestamps if not pd.isna(value)]
    if not timestamps:
        return ""
    return pd.Timestamp(min(timestamps)).strftime("%Y-%m-%d %H:%M:%S")


def validation_stock_tickers(stock_rows: list[dict[str, str]]) -> set[str]:
    return {str(row.get("ticker", "")).strip().upper() for row in stock_rows if str(row.get("ticker", "")).strip()}


def validation_index_codes(index_rows: list[dict[str, str]]) -> set[str]:
    return {
        str(row.get("index_code", "")).strip().upper()
        for row in index_rows
        if str(row.get("index_code", "")).strip()
    }


def build_pre_benchmark_lock() -> dict[str, Any]:
    universe_rows = read_universe_rows()
    tickers = universe_tickers(universe_rows)
    manifest = read_json(READINESS_MANIFEST_PATH)
    stock_rows = read_csv_rows(STOCK_VALIDATION_PATH)
    index_rows = read_csv_rows(INDEX_VALIDATION_PATH)
    eval_end = eval_end_from_readiness(manifest, stock_rows)

    stock_usable_count = sum(1 for row in stock_rows if as_bool(row.get("usable")))
    index_usable_count = sum(1 for row in index_rows if as_bool(row.get("usable")))
    stock_cache_paths = {ticker: rel(STOCK_CACHE_DIR / f"{ticker}.csv") for ticker in tickers}
    index_codes = sorted(validation_index_codes(index_rows))
    index_cache_paths = {code: rel(INDEX_CACHE_DIR / f"{code}.csv") for code in index_codes}

    blocking_reasons: list[str] = []
    if not READINESS_MANIFEST_PATH.exists():
        blocking_reasons.append(f"Missing readiness manifest: {rel(READINESS_MANIFEST_PATH)}")
    if not READINESS_REPORT_PATH.exists():
        blocking_reasons.append(f"Missing readiness report: {rel(READINESS_REPORT_PATH)}")
    if not STOCK_VALIDATION_PATH.exists():
        blocking_reasons.append(f"Missing stock validation CSV: {rel(STOCK_VALIDATION_PATH)}")
    if not INDEX_VALIDATION_PATH.exists():
        blocking_reasons.append(f"Missing index validation CSV: {rel(INDEX_VALIDATION_PATH)}")
    if len(tickers) != 30:
        blocking_reasons.append(f"Frozen universe count is {len(tickers)}, expected 30")
    if not EXPECTED_INCLUDED.issubset(set(tickers)):
        blocking_reasons.append(f"Frozen universe is missing expected additions: {sorted(EXPECTED_INCLUDED - set(tickers))}")
    excluded_present = EXPECTED_EXCLUDED.intersection(set(tickers))
    if excluded_present:
        blocking_reasons.append(f"Frozen universe still contains excluded tickers: {sorted(excluded_present)}")
    if any(str(row.get("index", "")).strip().upper() != "VN30" for row in universe_rows):
        blocking_reasons.append("Frozen universe contains non-VN30 rows")
    if any(str(row.get("source", "")).strip() != ACTIVE_UNIVERSE_SOURCE_RAW for row in universe_rows):
        blocking_reasons.append("Frozen universe source is not hose_january_2025_review for every ticker")
    manifest_tickers = [str(item).strip().upper() for item in manifest.get("active_universe_tickers", [])]
    if manifest_tickers and manifest_tickers != tickers:
        blocking_reasons.append("Readiness manifest ticker list does not match frozen universe file")
    if int(manifest.get("active_universe_count", 0) or 0) != 30:
        blocking_reasons.append("Readiness manifest active universe count is not 30")
    if not bool(manifest.get("benchmark_can_proceed", False)):
        blocking_reasons.append("Readiness manifest benchmark_can_proceed is not true")
    if bool(manifest.get("daily_data_used", True)):
        blocking_reasons.append("Readiness manifest indicates daily data was used")
    if bool(manifest.get("resampling_used", True)):
        blocking_reasons.append("Readiness manifest indicates resampling was used")
    if stock_usable_count != 30:
        blocking_reasons.append(f"Usable stock count is {stock_usable_count}, expected 30")
    if validation_stock_tickers(stock_rows) != set(tickers):
        blocking_reasons.append("Stock validation ticker set does not match frozen universe")
    if index_usable_count != 6:
        blocking_reasons.append(f"Usable index count is {index_usable_count}, expected 6")
    bad_stock_frequency = sorted(
        str(row.get("ticker", "")).strip().upper()
        for row in stock_rows
        if not as_bool(row.get("frequency_1h"))
    )
    if bad_stock_frequency:
        blocking_reasons.append(f"Stock validation has non-1H rows: {bad_stock_frequency}")
    bad_index_frequency = sorted(
        str(row.get("index_code", "")).strip().upper()
        for row in index_rows
        if not as_bool(row.get("frequency_1h"))
    )
    if bad_index_frequency:
        blocking_reasons.append(f"Index validation has non-1H rows: {bad_index_frequency}")
    missing_stock_files = [ticker for ticker, path in stock_cache_paths.items() if not (REPO_ROOT / path).exists()]
    if missing_stock_files:
        blocking_reasons.append(f"Missing stock cache files: {missing_stock_files}")
    missing_index_files = [code for code, path in index_cache_paths.items() if not (REPO_ROOT / path).exists()]
    if missing_index_files:
        blocking_reasons.append(f"Missing index cache files: {missing_index_files}")
    if not eval_end:
        blocking_reasons.append("Could not determine evaluation end from readiness manifest/cache")

    return {
        "created_at_utc": now_utc(),
        "active_universe_source": ACTIVE_UNIVERSE_SOURCE,
        "active_universe_source_raw": ACTIVE_UNIVERSE_SOURCE_RAW,
        "active_universe_count": len(tickers),
        "active_ticker_list": tickers,
        "active_universe_includes": sorted(EXPECTED_INCLUDED),
        "active_universe_excludes": sorted(EXPECTED_EXCLUDED),
        "stock_usable_count": stock_usable_count,
        "index_usable_count": index_usable_count,
        "train_start": TRAIN_START_TEXT,
        "train_cutoff": TRAIN_CUTOFF_TEXT,
        "eval_start": EVAL_START_TEXT,
        "eval_end": eval_end,
        "actual_latest_data_timestamp": str(manifest.get("actual_latest_data_timestamp", "")),
        "common_latest_usable_data_timestamp": str(manifest.get("common_latest_usable_data_timestamp", "")),
        "frequency": FREQUENCY_TEXT,
        "target": "stock_direction_classification",
        "training_label_cutoff_rule": "target_timestamp <= train_cutoff",
        "evaluation_label_rule": "target_timestamp >= eval_start; source cache rows clipped to eval_end before label generation",
        "data_cache_paths": {
            "stock_cache_dir": rel(STOCK_CACHE_DIR),
            "index_cache_dir": rel(INDEX_CACHE_DIR),
            "readiness_manifest": rel(READINESS_MANIFEST_PATH),
            "readiness_report": rel(READINESS_REPORT_PATH),
            "stock_validation_csv": rel(STOCK_VALIDATION_PATH),
            "index_validation_csv": rel(INDEX_VALIDATION_PATH),
            "stock_cache_files": stock_cache_paths,
            "index_cache_files": index_cache_paths,
        },
        "benchmark_allowed": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
        "data_fetch_performed": False,
        "daily_data_used": False,
        "resampling_used": False,
        "vn100_evidence_reused": False,
        "old_2005_2006_output_reused": False,
    }


def write_lock_markdown(lock: dict[str, Any]) -> None:
    tickers = ", ".join(lock["active_ticker_list"])
    reasons = lock.get("blocking_reasons", [])
    lines = [
        "# VN30 Hourly 2015 Jan-2025 Pre-Benchmark Readiness Lock",
        "",
        f"- Active universe source: {lock['active_universe_source']}.",
        f"- Active stock universe count: {lock['active_universe_count']}.",
        f"- Active ticker list: {tickers}.",
        f"- Stock usable count: {lock['stock_usable_count']}/30.",
        f"- Index usable count: {lock['index_usable_count']}/6.",
        f"- Train cutoff: `{lock['train_cutoff']}`.",
        f"- Evaluation start: `{lock['eval_start']}`.",
        f"- Evaluation end: `{lock['eval_end']}`.",
        f"- Stock cache directory: `{lock['data_cache_paths']['stock_cache_dir']}`.",
        f"- Index cache directory: `{lock['data_cache_paths']['index_cache_dir']}`.",
        f"- Benchmark allowed: {'yes' if lock['benchmark_allowed'] else 'no'}.",
        "- Data fetch performed: no.",
        "- Daily data used: no.",
        "- Resampling used: no.",
        "- VN100 evidence reused: no.",
        "",
    ]
    if reasons:
        lines.extend(["## Blocking Reasons", ""])
        lines.extend(f"- {reason}" for reason in reasons)
        lines.append("")
    else:
        lines.extend(["## Decision", "", "Benchmark may proceed from the validated gateway cache only.", ""])
    LOCK_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_pre_benchmark_lock(lock: dict[str, Any]) -> None:
    write_json(LOCK_JSON_PATH, lock)
    write_lock_markdown(lock)


def run_config_payload(
    output_dir: Path,
    *,
    lock: dict[str, Any],
    models: list[str],
    horizons: list[int],
    status: str,
) -> dict[str, Any]:
    return {
        "universe": "VN30",
        "active_universe_source": ACTIVE_UNIVERSE_SOURCE,
        "active_universe_source_raw": ACTIVE_UNIVERSE_SOURCE_RAW,
        "tickers": lock["active_ticker_list"],
        "active_ticker_count": len(lock["active_ticker_list"]),
        "frequency": FREQUENCY_TEXT,
        "frequency_scope": "hourly_only",
        "stock_cache_dir": rel(STOCK_CACHE_DIR),
        "index_cache_dir": rel(INDEX_CACHE_DIR),
        "index_cache_used_for_model_features": False,
        "train_start": TRAIN_START_TEXT,
        "train_cutoff": TRAIN_CUTOFF_TEXT,
        "eval_start": EVAL_START_TEXT,
        "eval_end": lock["eval_end"],
        "target_mode": TARGET_MODE,
        "target": "stock_direction",
        "training_label_cutoff_rule": "target_timestamp <= train_cutoff",
        "evaluation_label_rule": "target_timestamp >= eval_start; source cache rows clipped to eval_end before label generation",
        "models": models,
        "horizons": horizons,
        "threshold": THRESHOLD,
        "output_dir": rel(output_dir),
        "provider": "validated_hourly_gateway_cache",
        "provider_calls_allowed": False,
        "cache_only": True,
        "data_fetch_run": False,
        "daily_data_used": False,
        "resampling_used": False,
        "daily_to_hourly_resampling_used": False,
        "vn100_evidence_reused": False,
        "old_2005_2006_output_reused": False,
        "paper_or_docx_generated": False,
        "benchmark_allowed": bool(lock["benchmark_allowed"]),
        "status": status,
    }


def manifest_payload(
    *,
    lock: dict[str, Any],
    status: str,
    benchmark_run: bool,
    model_training_run: bool,
    generated_outputs: dict[str, bool] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "universe": "VN30",
        "active_universe_source": ACTIVE_UNIVERSE_SOURCE,
        "active_ticker_count": lock["active_universe_count"],
        "active_tickers": lock["active_ticker_list"],
        "stock_usable_count": lock["stock_usable_count"],
        "index_usable_count": lock["index_usable_count"],
        "frequency": FREQUENCY_TEXT,
        "train_start": TRAIN_START_TEXT,
        "train_cutoff": TRAIN_CUTOFF_TEXT,
        "eval_start": EVAL_START_TEXT,
        "eval_end": lock["eval_end"],
        "target_mode": TARGET_MODE,
        "training_label_cutoff_rule": "target_timestamp <= train_cutoff",
        "benchmark_allowed": bool(lock["benchmark_allowed"]),
        "benchmark_run": bool(benchmark_run),
        "model_training_run": bool(model_training_run),
        "data_fetch_run": False,
        "daily_data_used": False,
        "resampling_used": False,
        "daily_to_hourly_resampling_used": False,
        "vn100_evidence_reused": False,
        "old_2005_2006_output_reused": False,
        "paper_or_docx_generated": False,
        "pre_benchmark_readiness_lock_json": rel(LOCK_JSON_PATH),
        "pre_benchmark_readiness_lock_md": rel(LOCK_MD_PATH),
        "generated_outputs": generated_outputs or {},
        "status": status,
    }
    if extra:
        payload.update(extra)
    return payload


def append_run_log(output_dir: Path, message: str) -> None:
    log_path = output_dir / "hourly" / "benchmark_run_log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = "# VN30 Hourly 2015 Jan-2025 Benchmark Run Log\n\n"
    if not log_path.exists():
        log_path.write_text(prefix, encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"- {now_utc()} - {message}\n")


def source_health_rows(stock_rows: list[dict[str, str]], index_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in stock_rows:
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        rows.append(
            {
                "symbol": ticker,
                "asset_type": "stock",
                "cache_path": row.get("cache_path", rel(STOCK_CACHE_DIR / f"{ticker}.csv")),
                "first_datetime": row.get("first_datetime", ""),
                "last_datetime": row.get("last_datetime", ""),
                "row_count": row.get("row_count", ""),
                "training_rows_before_2025": row.get("training_rows_before_2025", ""),
                "evaluation_rows_from_2025": row.get("evaluation_rows_from_2025", ""),
                "frequency_1h": str(as_bool(row.get("frequency_1h"))).lower(),
                "usable": str(as_bool(row.get("usable"))).lower(),
                "used_in_model": "true",
                "readiness_scope": "model_input",
            }
        )
    for row in index_rows:
        code = str(row.get("index_code", "")).strip().upper()
        if not code:
            continue
        rows.append(
            {
                "symbol": code,
                "asset_type": "index",
                "cache_path": row.get("cache_path", rel(INDEX_CACHE_DIR / f"{code}.csv")),
                "first_datetime": row.get("first_datetime", ""),
                "last_datetime": row.get("last_datetime", ""),
                "row_count": row.get("row_count", ""),
                "training_rows_before_2025": "",
                "evaluation_rows_from_2025": "",
                "frequency_1h": str(as_bool(row.get("frequency_1h"))).lower(),
                "usable": str(as_bool(row.get("usable"))).lower(),
                "used_in_model": "false",
                "readiness_scope": "readiness_only",
            }
        )
    return rows


def read_stock_cache(ticker: str, eval_end: str) -> pd.DataFrame:
    path = STOCK_CACHE_DIR / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing stock cache for {ticker}: {rel(path)}")
    frame = pd.read_csv(path, low_memory=False)
    missing = [column for column in MODEL_INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Stock cache {rel(path)} missing columns: {missing}")
    if "frequency" in frame.columns:
        bad_frequency = frame["frequency"].dropna().astype(str).str.upper().ne(FREQUENCY_TEXT.upper())
        if bool(bad_frequency.any()):
            raise ValueError(f"Stock cache {rel(path)} contains non-{FREQUENCY_TEXT} frequency markers")
    marker_text = " ".join(
        frame[column].dropna().astype(str).str.lower().head(20).str.cat(sep=" ")
        for column in ("provider", "source", "frequency")
        if column in frame.columns
    )
    if "resample" in marker_text or "daily" in marker_text:
        raise ValueError(f"Stock cache {rel(path)} contains a daily/resampled source marker")
    prepared = frame[MODEL_INPUT_COLUMNS].copy()
    prepared["datetime"] = pd.to_datetime(prepared["datetime"], errors="coerce")
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper().str.strip()
    prepared = prepared[prepared["ticker"].eq(ticker.upper())].copy()
    for column in OHLCV_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = prepared.dropna(subset=["datetime", "ticker", *OHLCV_COLUMNS])
    prepared = prepared[(prepared["close"] > 0.0) & (prepared["volume"] >= 0.0)].copy()
    prepared = prepared[
        (prepared["datetime"] >= pd.Timestamp(TRAIN_START_TEXT))
        & (prepared["datetime"] <= pd.Timestamp(eval_end))
    ].copy()
    return (
        prepared.sort_values(["ticker", "datetime"])
        .drop_duplicates(["ticker", "datetime"], keep="last")[MODEL_INPUT_COLUMNS]
        .reset_index(drop=True)
    )


def load_stock_universe_frame(tickers: list[str], eval_end: str) -> pd.DataFrame:
    frames = [read_stock_cache(ticker, eval_end) for ticker in tickers]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=MODEL_INPUT_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["datetime"], errors="coerce")
    return (
        combined.dropna(subset=["datetime", "ticker", *OHLCV_COLUMNS])
        .sort_values(["ticker", "datetime"])
        .drop_duplicates(["ticker", "datetime"], keep="last")[MODEL_INPUT_COLUMNS]
        .reset_index(drop=True)
    )


def build_config(output_dir: Path, *, lock: dict[str, Any], models: list[str], horizons: list[int]) -> BenchmarkConfig:
    return BenchmarkConfig(
        universe="VN30",
        daily_start=TRAIN_START_TEXT,
        daily_end=TRAIN_CUTOFF_TEXT,
        hourly_start=TRAIN_START_TEXT,
        hourly_end=lock["eval_end"],
        train_cutoff=TRAIN_CUTOFF_TEXT,
        eval_start=EVAL_START_TEXT,
        eval_end=lock["eval_end"],
        models=models,
        daily_horizons=[],
        hourly_horizons=horizons,
        threshold=THRESHOLD,
        provider="validated_vn30_hourly_2015_gateway_cache",
        pull_missing=False,
        cache_dir=rel(STOCK_CACHE_DIR),
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


def prediction_counts_by_model_horizon(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    if predictions.empty:
        return []
    rows: list[dict[str, Any]] = []
    grouped = predictions.groupby(["model", "horizon"], dropna=False, sort=True)
    for (model, horizon), group in grouped:
        rows.append(
            {
                "model": str(model),
                "horizon": int(horizon),
                "n_predictions": int(len(group)),
                "accuracy": float(pd.to_numeric(group["is_correct"], errors="coerce").mean()),
            }
        )
    return rows


def generated_output_flags(output_dir: Path) -> dict[str, bool]:
    hourly = output_dir / "hourly"
    paths = {
        "run_config.json": output_dir / "run_config.json",
        "manifest.json": output_dir / "manifest.json",
        "hourly/benchmark_summary.json": hourly / "benchmark_summary.json",
        "hourly/predicted_vs_actual.csv": hourly / "predicted_vs_actual.csv",
        "hourly/accuracy_summary.csv": hourly / "accuracy_summary.csv",
        "hourly/classification_accuracy_summary.csv": hourly / "classification_accuracy_summary.csv",
        "hourly/baseline_summary.csv": hourly / "baseline_summary.csv",
        "hourly/baseline_delta_summary.csv": hourly / "baseline_delta_summary.csv",
        "hourly/significance_summary.csv": hourly / "significance_summary.csv",
        "hourly/regime_accuracy_summary.csv": hourly / "regime_accuracy_summary.csv",
        "hourly/model_error_summary.csv": hourly / "model_error_summary.csv",
        "hourly/source_health_summary.csv": hourly / "source_health_summary.csv",
        "hourly/benchmark_run_log.md": hourly / "benchmark_run_log.md",
    }
    return {name: path.exists() for name, path in paths.items()}


def write_empty_hourly_outputs(output_dir: Path, stock_rows: list[dict[str, str]], index_rows: list[dict[str, str]]) -> None:
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
    write_csv(hourly_dir / "source_health_summary.csv", source_health_rows(stock_rows, index_rows), SOURCE_HEALTH_COLUMNS)


def write_success_outputs(
    output_dir: Path,
    config: BenchmarkConfig,
    lock: dict[str, Any],
    stock_rows: list[dict[str, str]],
    index_rows: list[dict[str, str]],
    outputs: tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> dict[str, Any]:
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
            "active_universe_source": ACTIVE_UNIVERSE_SOURCE,
            "active_ticker_count": lock["active_universe_count"],
            "stock_usable_count": lock["stock_usable_count"],
            "index_usable_count": lock["index_usable_count"],
            "frequency": "hourly",
            "frequency_marker": FREQUENCY_TEXT,
            "train_cutoff": TRAIN_CUTOFF_TEXT,
            "eval_start": EVAL_START_TEXT,
            "eval_end": lock["eval_end"],
            "target_mode": TARGET_MODE,
            "target": "stock_direction",
            "training_label_cutoff_rule": "target_timestamp <= train_cutoff",
            "evaluation_label_rule": "source cache rows clipped to eval_end before label generation",
            "predictions_by_model_horizon": prediction_counts_by_model_horizon(predictions),
            "all_30_active_tickers_represented": set(lock["active_ticker_list"]).issubset(
                set(predictions["ticker"].dropna().astype(str).str.upper())
            )
            if not predictions.empty
            else False,
            "data_fetch_run": False,
            "daily_data_used": False,
            "resampling_used": False,
            "daily_to_hourly_resampling_used": False,
            "vn100_evidence_reused": False,
            "old_2005_2006_output_reused": False,
            "paper_or_docx_generated": False,
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
    write_csv(hourly_dir / "source_health_summary.csv", source_health_rows(stock_rows, index_rows), SOURCE_HEALTH_COLUMNS)
    write_json(hourly_dir / "benchmark_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VN30 hourly 2015 Jan-2025 benchmark from validated cache.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--horizons", default=",".join(str(item) for item in DEFAULT_HORIZONS))
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    models = parse_csv_list(args.models)
    horizons = parse_int_list(args.horizons)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hourly").mkdir(parents=True, exist_ok=True)
    append_run_log(output_dir, "started readiness validation")
    lock = build_pre_benchmark_lock()
    write_pre_benchmark_lock(lock)
    status = "readiness_locked" if lock["benchmark_allowed"] else "stopped_readiness_failed"
    write_json(output_dir / "run_config.json", run_config_payload(output_dir, lock=lock, models=models, horizons=horizons, status=status))
    write_json(
        output_dir / "manifest.json",
        manifest_payload(lock=lock, status=status, benchmark_run=False, model_training_run=False),
    )

    stock_rows = read_csv_rows(STOCK_VALIDATION_PATH)
    index_rows = read_csv_rows(INDEX_VALIDATION_PATH)
    if not lock["benchmark_allowed"]:
        append_run_log(output_dir, "stopped before benchmark because readiness lock failed")
        write_empty_hourly_outputs(output_dir, stock_rows, index_rows)
        write_json(output_dir / "hourly" / "benchmark_summary.json", {"benchmark_run": False, "status": status, "blocking_reasons": lock["blocking_reasons"]})
        print("VN30 hourly 2015 benchmark stopped: readiness lock failed")
        for reason in lock["blocking_reasons"]:
            print(f"- {reason}")
        return 2

    append_run_log(output_dir, f"readiness locked with eval_end={lock['eval_end']}")
    if args.preflight_only:
        append_run_log(output_dir, "preflight-only run completed before model training")
        print(f"VN30 hourly 2015 benchmark readiness locked: {rel(LOCK_JSON_PATH)}")
        return 0

    append_run_log(output_dir, f"loading validated stock cache from {rel(STOCK_CACHE_DIR)}")
    raw = load_stock_universe_frame(lock["active_ticker_list"], lock["eval_end"])
    if raw.empty:
        append_run_log(output_dir, "failed before benchmark: validated stock cache produced zero rows")
        write_empty_hourly_outputs(output_dir, stock_rows, index_rows)
        write_json(
            output_dir / "manifest.json",
            manifest_payload(lock=lock, status="failed_empty_stock_cache", benchmark_run=False, model_training_run=False),
        )
        raise RuntimeError("Validated stock cache produced zero model-input rows")

    append_run_log(output_dir, f"loaded {len(raw)} stock rows; starting benchmark models={models} horizons={horizons}")
    config = build_config(output_dir, lock=lock, models=models, horizons=horizons)
    try:
        outputs = run_frequency_benchmark(
            raw_df=raw,
            frequency="hourly",
            horizons=horizons,
            models=models,
            initial_train_start=TRAIN_START_TEXT,
            initial_train_end=TRAIN_CUTOFF_TEXT,
            eval_start=EVAL_START_TEXT,
            eval_end=lock["eval_end"],
            threshold=THRESHOLD,
            provider="validated_vn30_hourly_2015_gateway_cache",
            universe="VN30",
            retrain_frequency=config.retrain_frequency,
            seed=config.seed,
            min_history_days=config.min_history_days,
            min_obs_per_group=config.min_obs_per_group,
            max_daily_gap_days=config.max_daily_gap_days,
            config=config,
        )
    except BaseException as exc:
        append_run_log(output_dir, f"benchmark failed: {type(exc).__name__}: {exc}")
        write_empty_hourly_outputs(output_dir, stock_rows, index_rows)
        write_json(
            output_dir / "manifest.json",
            manifest_payload(
                lock=lock,
                status="failed_benchmark_exception",
                benchmark_run=False,
                model_training_run=True,
                extra={"exception_type": type(exc).__name__, "exception_message": str(exc)},
            ),
        )
        raise

    summary = write_success_outputs(output_dir, config, lock, stock_rows, index_rows, outputs)
    write_json(output_dir / "run_config.json", run_config_payload(output_dir, lock=lock, models=models, horizons=horizons, status="completed"))
    write_json(
        output_dir / "manifest.json",
        manifest_payload(
            lock=lock,
            status="completed",
            benchmark_run=True,
            model_training_run=True,
            generated_outputs=generated_output_flags(output_dir),
            extra={
                "n_predictions": int(summary.get("n_predictions", 0) or 0),
                "overall_accuracy": summary.get("overall_accuracy"),
                "passed": bool(summary.get("passed", False)),
                "models_run": summary.get("evaluated_models", []),
                "horizons_run": summary.get("evaluated_horizons", []),
            },
        ),
    )
    append_run_log(
        output_dir,
        f"benchmark completed accuracy={summary.get('overall_accuracy')} n_predictions={summary.get('n_predictions')}",
    )
    print(
        "VN30 hourly 2015 Jan-2025 benchmark complete: "
        f"accuracy={summary.get('overall_accuracy')} "
        f"n={summary.get('n_predictions')} "
        f"output_dir={rel(output_dir)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
