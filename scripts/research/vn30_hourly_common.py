"""Shared helpers for the frozen VN30 hourly 2005-2026 research rerun."""

from __future__ import annotations

import base64
import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
REPORT_ROOT = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2005_2026"
BENCHMARK_OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_official_2005_2026_traincutoff"

TRAIN_START = pd.Timestamp("2005-01-01 00:00:00")
TRAIN_CUTOFF = pd.Timestamp("2024-12-31 23:59:59")
EVAL_START = pd.Timestamp("2025-01-01 00:00:00")
EVAL_END = pd.Timestamp("2026-05-31 23:59:59")

TRAIN_START_TEXT = "2005-01-01 00:00:00"
TRAIN_CUTOFF_TEXT = "2024-12-31 23:59:59"
EVAL_START_TEXT = "2025-01-01 00:00:00"
EVAL_END_TEXT = "2026-05-31 23:59:59"

VN30_TICKERS = [
    "ACB",
    "BID",
    "CTG",
    "DGC",
    "FPT",
    "GAS",
    "GVR",
    "HDB",
    "HPG",
    "LPB",
    "MBB",
    "MSN",
    "MWG",
    "PLX",
    "SAB",
    "SHB",
    "SSB",
    "SSI",
    "STB",
    "TCB",
    "TPB",
    "VCB",
    "VHM",
    "VIB",
    "VIC",
    "VJC",
    "VNM",
    "VPB",
    "VPL",
    "VRE",
]

HOURLY_SOURCE_DIRS = (
    ("vn30_provider_cache_hourly", REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly"),
    ("legacy_raw_provider_cache_hourly", REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn100" / "hourly"),
    ("hourly_market_split_data", REPO_ROOT / "data" / "hourly_market_split_data"),
)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
HOURLY_COLUMNS = ["datetime", "ticker", *OHLCV_COLUMNS]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_universe(path: Path = UNIVERSE_PATH) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Frozen VN30 universe file is missing: {rel(path)}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"ticker", "index", "source", "effective_period", "note"}
    missing_columns = required.difference(rows[0].keys() if rows else [])
    if missing_columns:
        raise ValueError(f"Frozen VN30 universe is missing columns: {sorted(missing_columns)}")
    tickers = [str(row.get("ticker", "")).strip().upper() for row in rows if str(row.get("ticker", "")).strip()]
    if tickers != VN30_TICKERS:
        raise ValueError(
            "Frozen VN30 universe must contain exactly the requested 30 tickers in order. "
            f"Expected {VN30_TICKERS}; got {tickers}."
        )
    bad_index = [row.get("ticker", "") for row in rows if str(row.get("index", "")).strip().upper() != "VN30"]
    if bad_index:
        raise ValueError(f"Frozen universe contains non-VN30 rows: {bad_index}")
    return tickers


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def markdown_table(headers: list[str], rows: list[dict[str, Any]], *, max_rows: int | None = None) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    displayed = rows if max_rows is None else rows[:max_rows]
    for row in displayed:
        lines.append("| " + " | ".join(format_cell(row.get(header, "")) for header in headers) + " |")
    if max_rows is not None and len(rows) > max_rows:
        lines.append("| " + " | ".join(["..."] + ["" for _ in headers[1:]]) + " |")
    return "\n".join(lines)


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if 0.0 <= value <= 1.0:
            return f"{value * 100:.2f}%"
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def timestamp_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def find_time_column(frame: pd.DataFrame) -> str | None:
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in ("datetime", "time", "timestamp", "date", "trading_date"):
        if candidate in normalized:
            return str(normalized[candidate])
    return None


def standardize_hourly_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=HOURLY_COLUMNS)
    prepared = frame.copy()
    time_column = find_time_column(prepared)
    if time_column is None:
        return pd.DataFrame(columns=HOURLY_COLUMNS)
    if time_column != "datetime":
        prepared = prepared.rename(columns={time_column: "datetime"})
    prepared["datetime"] = pd.to_datetime(prepared["datetime"], errors="coerce")
    prepared["ticker"] = prepared.get("ticker", ticker)
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper().str.strip()
    prepared.loc[prepared["ticker"].eq("") | prepared["ticker"].isna(), "ticker"] = ticker
    for column in OHLCV_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = pd.NA
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = prepared.dropna(subset=["datetime", "open", "high", "low", "close", "volume"])
    prepared = prepared[(prepared["close"] > 0.0) & (prepared["volume"] >= 0.0)]
    prepared = prepared[prepared["ticker"].eq(ticker.upper())].copy()
    if prepared.empty:
        return pd.DataFrame(columns=HOURLY_COLUMNS)
    prepared = prepared.sort_values(["ticker", "datetime"]).drop_duplicates(["ticker", "datetime"], keep="last")
    return prepared[HOURLY_COLUMNS].reset_index(drop=True)


def load_hourly_sources_for_ticker(ticker: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    frames: list[pd.DataFrame] = []
    sources: list[str] = []
    files: list[str] = []
    for source_name, source_dir in HOURLY_SOURCE_DIRS:
        path = source_dir / f"{ticker.upper()}.csv"
        if not path.exists():
            continue
        try:
            raw = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        standardized = standardize_hourly_frame(raw, ticker)
        if standardized.empty:
            continue
        standardized["raw_source"] = source_name
        frames.append(standardized)
        sources.append(source_name)
        files.append(rel(path))
    if not frames:
        return pd.DataFrame(columns=[*HOURLY_COLUMNS, "raw_source"]), [], []
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["ticker", "datetime", "raw_source"]).drop_duplicates(
        ["ticker", "datetime"],
        keep="last",
    )
    return combined.reset_index(drop=True), sorted(set(sources)), files


def audit_hourly_coverage() -> list[dict[str, Any]]:
    tickers = read_universe()
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        frame, sources, files = load_hourly_sources_for_ticker(ticker)
        timestamps = pd.to_datetime(frame["datetime"], errors="coerce").dropna() if not frame.empty else pd.Series(dtype="datetime64[ns]")
        first_ts = pd.Timestamp(timestamps.min()) if not timestamps.empty else None
        last_ts = pd.Timestamp(timestamps.max()) if not timestamps.empty else None
        training_rows = int(((timestamps >= TRAIN_START) & (timestamps <= TRAIN_CUTOFF)).sum()) if not timestamps.empty else 0
        evaluation_rows = int(((timestamps >= EVAL_START) & (timestamps <= EVAL_END)).sum()) if not timestamps.empty else 0

        reasons: list[str] = []
        missing_training = False
        missing_evaluation = False
        if timestamps.empty or first_ts is None or last_ts is None:
            missing_training = True
            missing_evaluation = True
            reasons.append("hourly_file_missing_or_empty")
        else:
            if first_ts > TRAIN_START:
                missing_training = True
                reasons.append(f"training_start_after_requested_start:{timestamp_text(first_ts)}>{TRAIN_START_TEXT}")
            if last_ts < TRAIN_CUTOFF:
                missing_training = True
                reasons.append(f"training_end_before_train_cutoff:{timestamp_text(last_ts)}<{TRAIN_CUTOFF_TEXT}")
            if training_rows == 0:
                missing_training = True
                reasons.append("no_training_rows_in_requested_window")
            if first_ts > EVAL_START:
                missing_evaluation = True
                reasons.append(f"evaluation_start_after_requested_start:{timestamp_text(first_ts)}>{EVAL_START_TEXT}")
            if last_ts < EVAL_END:
                missing_evaluation = True
                reasons.append(f"evaluation_end_before_requested_end:{timestamp_text(last_ts)}<{EVAL_END_TEXT}")
            if evaluation_rows == 0:
                missing_evaluation = True
                reasons.append("no_evaluation_rows_in_requested_window")

        rows.append(
            {
                "ticker": ticker,
                "first_available_hourly_timestamp": timestamp_text(first_ts),
                "last_available_hourly_timestamp": timestamp_text(last_ts),
                "hourly_rows": int(len(frame)),
                "training_rows_2005_2024": training_rows,
                "evaluation_rows_2025_2026": evaluation_rows,
                "missing_training_coverage": bool(missing_training),
                "missing_evaluation_coverage": bool(missing_evaluation),
                "benchmark_usable": bool(not missing_training and not missing_evaluation),
                "missing_reason": "usable" if not reasons else "; ".join(dict.fromkeys(reasons)),
                "raw_hourly_sources": ";".join(sources),
                "raw_hourly_files": ";".join(files),
            }
        )
    return rows


def usable_tickers_from_audit(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["ticker"]) for row in rows if as_bool(row.get("benchmark_usable"))]


def failed_tickers_from_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if not as_bool(row.get("benchmark_usable"))]


def write_missing_evidence_report(path: Path, audit_rows: list[dict[str, Any]], *, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    usable = usable_tickers_from_audit(audit_rows)
    failed = failed_tickers_from_audit(audit_rows)
    feasible = len(usable) == len(VN30_TICKERS)
    headers = [
        "ticker",
        "first_available_hourly_timestamp",
        "last_available_hourly_timestamp",
        "hourly_rows",
        "benchmark_usable",
        "missing_reason",
    ]
    content = [
        "# VN30 Hourly Benchmark Missing Evidence",
        "",
        "## Design",
        "",
        "- Universe: frozen VN30, exactly 30 tickers.",
        "- Frequency: hourly only.",
        f"- Training-label period: {TRAIN_START_TEXT} to {TRAIN_CUTOFF_TEXT}.",
        f"- Evaluation/comparison period: {EVAL_START_TEXT} to {EVAL_END_TEXT}.",
        "- Leakage rule: training labels require target_timestamp <= train_cutoff.",
        "",
        "## Result",
        "",
        f"- Source script: `{source}`.",
        f"- Benchmark-usable tickers: {len(usable)} of 30.",
        f"- Failed tickers: {len(failed)}.",
        f"- Requested 2005-2026 hourly design feasible: {str(feasible).lower()}.",
        "",
        "The VN30 hourly rerun did not achieve full 30-ticker benchmark usability under the requested 2005-2026 hourly design.",
        "",
        "## Failed Tickers",
        "",
        markdown_table(headers, failed if failed else audit_rows),
        "",
        "## Claim Boundary",
        "",
        "- No final VN30 hourly paper claims should be written from this run.",
        "- No daily data, daily-to-hourly resampling, VN100 seven-ticker evidence, or shortened period is accepted as a substitute.",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def load_hourly_universe_frame(tickers: list[str] | None = None) -> pd.DataFrame:
    selected = tickers or read_universe()
    frames: list[pd.DataFrame] = []
    for ticker in selected:
        frame, _sources, _files = load_hourly_sources_for_ticker(ticker)
        if frame.empty:
            continue
        frame = frame[(frame["datetime"] >= TRAIN_START) & (frame["datetime"] <= EVAL_END)].copy()
        if not frame.empty:
            frames.append(frame[HOURLY_COLUMNS])
    if not frames:
        return pd.DataFrame(columns=HOURLY_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["ticker", "datetime"]).drop_duplicates(["ticker", "datetime"], keep="last").reset_index(drop=True)


def load_hourly_predictions(artifact_dir: Path = BENCHMARK_OUTPUT_DIR) -> pd.DataFrame:
    path = artifact_dir / "hourly" / "predicted_vs_actual.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        return frame
    if "frequency" not in frame.columns:
        frame["frequency"] = "hourly"
    for column in ("horizon", "confidence", "actual_return", "actual_direction", "predicted_direction", "is_correct"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["timestamp_sort"] = pd.to_datetime(
        frame["timestamp"] if "timestamp" in frame.columns else frame.get("date"),
        errors="coerce",
    )
    for column in ("model", "ticker", "frequency"):
        if column in frame.columns:
            frame[column] = frame[column].astype(str)
    return frame


def save_placeholder_figure(path: Path, title: str, message: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.axis("off")
        ax.text(0.5, 0.65, title, ha="center", va="center", fontsize=14, weight="bold")
        ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10, wrap=True)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return "rendered"
    except Exception as exc:
        tiny_png = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
            "/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
        )
        path.write_bytes(base64.b64decode(tiny_png))
        note = path.with_suffix(".md")
        note.write_text(
            f"# Figure Placeholder Fallback\n\nMatplotlib was unavailable: `{exc}`.\n",
            encoding="utf-8",
        )
        return f"fallback_png: {exc}"
