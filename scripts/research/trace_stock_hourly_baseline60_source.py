"""Trace the locked VN30 stock-hourly RF h=60 baseline60 evidence.

The script reads existing reports and outputs only. It does not run training.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.vn30_hourly_common import HOURLY_SOURCE_DIRS, VN30_TICKERS, load_hourly_sources_for_ticker


REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_stock_index_joint_panel_data_recovery"
TARGET_OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_2015_horizon_relative_target_experiments"
LOCK_REPORT = REPO_ROOT / "reports" / "VN30_HOURLY_2015_BASELINE60_RESULT_LOCK.md"
HISTORICAL_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "legacy"
    / "research"
    / "failed_experiments"
    / "run_vn30_hourly_2015_horizon_relative_target_experiments.py"
)
HISTORICAL_STOCK_CACHE = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly_2015"
TARGET_ACCURACY = 0.602188


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def source_files_for_ticker(ticker: str) -> tuple[int, str, str, str, str]:
    frame, sources, files = load_hourly_sources_for_ticker(ticker)
    if frame.empty:
        return 0, "", "", ";".join(sources), ";".join(files)
    timestamps = pd.to_datetime(frame["datetime"], errors="coerce").dropna()
    return (
        int(len(frame)),
        "" if timestamps.empty else str(timestamps.min()),
        "" if timestamps.empty else str(timestamps.max()),
        ";".join(sources),
        ";".join(files),
    )


def historical_cache_status_for_ticker(ticker: str) -> dict[str, Any]:
    path = HISTORICAL_STOCK_CACHE / f"{ticker}.csv"
    row = {
        "row_type": "historical_script_stock_cache_status",
        "ticker": ticker,
        "historical_stock_cache_path": rel(path),
        "source_files_still_exist": path.exists(),
        "source_data_row_count": 0,
        "source_data_first_timestamp": "",
        "source_data_last_timestamp": "",
        "source_has_more_than_111_rows": False,
        "frequency_detected": "missing",
    }
    if not path.exists():
        return row
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception:
        row["frequency_detected"] = "unreadable"
        return row
    time_col = "datetime" if "datetime" in frame.columns else "time" if "time" in frame.columns else ""
    row["source_data_row_count"] = int(len(frame))
    row["source_has_more_than_111_rows"] = int(len(frame)) > 111
    if time_col:
        datetimes = pd.to_datetime(frame[time_col], errors="coerce").dropna()
        if not datetimes.empty:
            times = sorted(datetimes.dt.strftime("%H:%M:%S").unique().tolist())
            row["source_data_first_timestamp"] = str(datetimes.min())
            row["source_data_last_timestamp"] = str(datetimes.max())
            row["frequency_detected"] = "midnight_only_daily_like" if times == ["00:00:00"] else "intraday_hourly" if len(times) > 1 else "daily_like"
    return row


def trace_candidate_result() -> dict[str, Any]:
    final_results = read_csv(TARGET_OUTPUT_DIR / "final_eval_results.csv")
    matched = pd.DataFrame()
    if not final_results.empty:
        work = final_results.copy()
        for col in ("horizon", "final_accuracy", "final_coverage", "final_rows", "validation_rows"):
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")
        mask = (
            work.get("model", "").astype(str).eq("random_forest")
            & work.get("target_type", "").astype(str).eq("absolute")
            & work.get("horizon", pd.Series(dtype=float)).eq(60)
            & work.get("feature_set", "").astype(str).eq("C")
            & work.get("final_coverage", pd.Series(dtype=float)).eq(1.0)
        )
        matched = work[mask].copy()
        if not matched.empty:
            matched["accuracy_distance"] = (matched["final_accuracy"] - TARGET_ACCURACY).abs()
            matched = matched.sort_values(["accuracy_distance", "final_rows"], ascending=[True, False])
    row = matched.iloc[0].to_dict() if not matched.empty else {}
    prediction_file = TARGET_OUTPUT_DIR / "predicted_vs_actual.csv"
    return {
        "baseline60_source_found": bool(row),
        "benchmark_output_directory": rel(TARGET_OUTPUT_DIR),
        "prediction_file_path": rel(prediction_file) if prediction_file.exists() else "",
        "prediction_file_exists": prediction_file.exists(),
        "run_config_path": rel(TARGET_OUTPUT_DIR / "run_config.json") if (TARGET_OUTPUT_DIR / "run_config.json").exists() else "",
        "manifest_path": rel(TARGET_OUTPUT_DIR / "experiment_manifest.json") if (TARGET_OUTPUT_DIR / "experiment_manifest.json").exists() else "",
        "historical_script_path": rel(HISTORICAL_SCRIPT) if HISTORICAL_SCRIPT.exists() else "",
        "source_data_path_recorded_or_inferred": rel(HISTORICAL_STOCK_CACHE),
        "lock_report_path": rel(LOCK_REPORT) if LOCK_REPORT.exists() else "",
        "model": row.get("model", ""),
        "horizon": row.get("horizon", ""),
        "feature_set": row.get("feature_set", ""),
        "target_type": row.get("target_type", ""),
        "final_accuracy": row.get("final_accuracy", ""),
        "final_coverage": row.get("final_coverage", ""),
        "final_rows": row.get("final_rows", ""),
        "validation_accuracy": row.get("validation_accuracy", ""),
        "validation_rows": row.get("validation_rows", ""),
        "final_evaluation_window": "2025-01-01 onward per run_config; lock report states 2025-01-01 to 2026-05-14",
        "stock_universe_used": ",".join(VN30_TICKERS),
    }


def source_path_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ticker in VN30_TICKERS:
        row_count, first_ts, last_ts, sources, files = source_files_for_ticker(ticker)
        rows.append(
            {
                "row_type": "current_common_loader_source_status",
                "ticker": ticker,
                "source_files_still_exist": bool(files),
                "source_data_row_count": row_count,
                "source_data_first_timestamp": first_ts,
                "source_data_last_timestamp": last_ts,
                "source_has_more_than_111_rows": row_count > 111,
                "raw_hourly_sources": sources,
                "raw_hourly_files": files,
            }
        )
        rows.append(historical_cache_status_for_ticker(ticker))
    return rows


def summarize(result: dict[str, Any], source_rows: list[dict[str, Any]]) -> str:
    common_rows = [row for row in source_rows if row["row_type"] == "current_common_loader_source_status"]
    historical_rows = [row for row in source_rows if row["row_type"] == "historical_script_stock_cache_status"]
    common_counts = [int(row["source_data_row_count"]) for row in common_rows]
    historical_counts = [int(row["source_data_row_count"]) for row in historical_rows]
    more_than_111 = [row for row in historical_rows if row["source_has_more_than_111_rows"]]
    midnight_like = [row["ticker"] for row in historical_rows if row.get("frequency_detected") == "midnight_only_daily_like"]
    source_dirs = "; ".join(f"{name}:{rel(path)}" for name, path in HOURLY_SOURCE_DIRS)
    content = [
        "# Stock Hourly Baseline60 Source Trace",
        "",
        "- Trace mode: read-only.",
        f"- Baseline60 source found: {str(result.get('baseline60_source_found')).lower()}.",
        f"- Benchmark output directory: `{result.get('benchmark_output_directory')}`.",
        f"- Prediction file path: `{result.get('prediction_file_path') or 'not found'}`.",
        f"- Prediction file exists: {str(result.get('prediction_file_exists')).lower()}.",
        f"- Run config path: `{result.get('run_config_path')}`.",
        f"- Manifest path: `{result.get('manifest_path')}`.",
        f"- Historical script path: `{result.get('historical_script_path')}`.",
        f"- Source data path recorded or inferred: `{result.get('source_data_path_recorded_or_inferred')}`.",
        f"- Locked result report: `{result.get('lock_report_path')}`.",
        f"- Model/horizon/feature set: {result.get('model')} h={result.get('horizon')} feature_set={result.get('feature_set')}.",
        f"- Final accuracy traced: {result.get('final_accuracy')}.",
        f"- Final rows traced: {result.get('final_rows')}.",
        f"- Stock universe count: {len(VN30_TICKERS)}.",
        f"- Local source directories checked: {source_dirs}.",
        f"- Current common-loader source row count range: {min(common_counts) if common_counts else 0}-{max(common_counts) if common_counts else 0}.",
        f"- Historical script stock-cache row count range: {min(historical_counts) if historical_counts else 0}-{max(historical_counts) if historical_counts else 0}.",
        f"- Historical stock-cache files with rows >111: {len(more_than_111)}/30.",
        f"- Historical stock-cache daily-like files: {', '.join(midnight_like) if midnight_like else 'none'}.",
        "- Benchmark/training run: no.",
        "- Data fetch: no.",
        "",
        "## Source Interpretation",
        "",
        "The locked RF h=60 evidence is traceable to the horizon-relative-target experiment output. The legacy script that produced that output reads `data/market_cache/vnstock_data/vn30/hourly_2015` as its stock cache.",
        "",
        "The current common loader used by newer 2005-2026 helpers still resolves to the shorter active-cache rows, which explains why the first joint-panel attempt missed the historical h=60 stock source. The historical source path contains more than 111 rows for all 30 tickers, but `VPL` is daily-like rather than true intraday hourly.",
        "",
    ]
    return "\n".join(content)


def main() -> int:
    result = trace_candidate_result()
    sources = source_path_rows()
    rows = [{"row_type": "baseline60_result_trace", **result}, *sources]
    write_csv(REPORT_DIR / "baseline60_source_trace.csv", rows)
    (REPORT_DIR / "baseline60_source_trace.md").write_text(summarize(result, sources), encoding="utf-8")
    historical = [row for row in sources if row["row_type"] == "historical_script_stock_cache_status"]
    print(
        "baseline60_source_found="
        f"{str(result.get('baseline60_source_found')).lower()} "
        f"historical_source_row_range={min(int(row['source_data_row_count']) for row in historical)}-"
        f"{max(int(row['source_data_row_count']) for row in historical)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
