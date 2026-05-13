"""Shared helpers for VN30 hourly available-window research artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.research.vn30_hourly_common import (
    REPO_ROOT,
    VN30_TICKERS,
    as_bool,
    load_hourly_sources_for_ticker,
    markdown_table,
    read_csv_rows,
    read_json,
    rel,
    save_placeholder_figure,
    timestamp_text,
    write_csv,
    write_json,
)


REPORT_ROOT = REPO_ROOT / "reports" / "generated" / "vn30_hourly_available_window"
AUDIT_DIR = REPORT_ROOT / "audit"
CONFIDENCE_DIR = REPORT_ROOT / "confidence"
REGIME_DIR = REPORT_ROOT / "regime"
COST_SLIPPAGE_DIR = REPORT_ROOT / "cost_slippage"
PAPER_TABLE_DIR = REPORT_ROOT / "paper_tables"
PAPER_FIGURE_DIR = REPORT_ROOT / "paper_figures"
PAPER_NOTE_DIR = REPORT_ROOT / "paper_notes"
BENCHMARK_OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_available_window_benchmark"
DESIGN_DECISION_MD = REPORT_ROOT / "vn30_hourly_available_window_design_decision.md"
DESIGN_DECISION_JSON = REPORT_ROOT / "vn30_hourly_available_window_design_decision.json"

DEFAULT_MODELS = ["xgboost", "lightgbm", "random_forest", "stacking"]
DEFAULT_HORIZONS = [1, 4, 8, 20]
TARGET_MODE = "classification"
FREQUENCY = "hourly"

# These are deliberately modest, but non-trivial, because this is a local-data
# available-window study rather than the failed 2005-2026 full-history design.
MIN_TRAIN_ROWS_PER_TICKER = 250
MIN_EVAL_ROWS_PER_TICKER = 100
MIN_COMMON_TIMESTAMPS = MIN_TRAIN_ROWS_PER_TICKER + MIN_EVAL_ROWS_PER_TICKER + max(DEFAULT_HORIZONS)
TRAIN_FRACTION = 0.60


def load_design_decision(path: Path = DESIGN_DECISION_JSON) -> dict[str, Any]:
    payload = read_json(path)
    if not payload:
        raise FileNotFoundError(f"Available-window design decision is missing or empty: {rel(path)}")
    return payload


def final_paper_can_proceed(decision: dict[str, Any] | None = None) -> bool:
    decision = decision or load_design_decision()
    return as_bool(decision.get("final_paper_can_proceed"))


def selected_tickers(decision: dict[str, Any] | None = None) -> list[str]:
    decision = decision or load_design_decision()
    return [str(item).strip().upper() for item in decision.get("selected_tickers", []) if str(item).strip()]


def excluded_tickers(decision: dict[str, Any] | None = None) -> list[str]:
    decision = decision or load_design_decision()
    return [str(item).strip().upper() for item in decision.get("excluded_tickers", []) if str(item).strip()]


def load_selected_hourly_frame(decision: dict[str, Any] | None = None) -> pd.DataFrame:
    decision = decision or load_design_decision()
    tickers = selected_tickers(decision)
    start = pd.Timestamp(decision["training_start"])
    end = pd.Timestamp(decision["evaluation_end"])
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        frame, _sources, _files = load_hourly_sources_for_ticker(ticker)
        if frame.empty:
            continue
        filtered = frame[(frame["datetime"] >= start) & (frame["datetime"] <= end)].copy()
        if not filtered.empty:
            frames.append(filtered[["datetime", "ticker", "open", "high", "low", "close", "volume"]])
    if not frames:
        return pd.DataFrame(columns=["datetime", "ticker", "open", "high", "low", "close", "volume"])
    combined = pd.concat(frames, ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["datetime"], errors="coerce")
    return (
        combined.dropna(subset=["datetime"])
        .sort_values(["ticker", "datetime"])
        .drop_duplicates(["ticker", "datetime"], keep="last")
        .reset_index(drop=True)
    )


def load_available_window_predictions(artifact_dir: Path = BENCHMARK_OUTPUT_DIR) -> pd.DataFrame:
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
    source = frame["timestamp"] if "timestamp" in frame.columns else frame.get("date")
    frame["timestamp_sort"] = pd.to_datetime(source, errors="coerce")
    for column in ("model", "ticker", "frequency"):
        if column in frame.columns:
            frame[column] = frame[column].astype(str)
    return frame


def write_docx_build_notes(path: Path, decision: dict[str, Any]) -> None:
    paper_path = REPO_ROOT / "reports" / "NCKH_FULL_PAPER_DRAFT_VN30_HOURLY_AVAILABLE_WINDOW_V1_WITH_FIGURES.md"
    content = [
        "# VN30 Hourly Available-Window DOCX Build Notes",
        "",
        "## Source Markdown",
        "",
        f"- `{rel(paper_path)}`" if paper_path.exists() else "- Final paper Markdown has not been generated.",
        "",
        "## Selected Design",
        "",
        f"- Selected tickers: {', '.join(selected_tickers(decision)) or 'None'}.",
        f"- Excluded tickers: {', '.join(excluded_tickers(decision)) or 'None'}.",
        f"- Training window: {decision.get('training_start', '')} to {decision.get('training_cutoff', '')}.",
        f"- Evaluation window: {decision.get('evaluation_start', '')} to {decision.get('evaluation_end', '')}.",
        f"- Final paper can proceed: {str(final_paper_can_proceed(decision)).lower()}.",
        "",
        "## Market Index Context",
        "",
        "- `VNINDEX`, `VN30INDEX`, and `VNXALL` index context should be included only if exact-code local hourly index data overlaps the available-window design.",
        "- For the full 2005-2026 design, `VN30INDEX` is not required before `2012-02-06 00:00:00` and `VNXALL` is not required before `2016-10-24 00:00:00`.",
        "- The full-design comparison/evaluation window remains aligned from `2025-01-01 00:00:00` to `2026-05-31 23:59:59`.",
        "- Missing pre-start `VN30INDEX`/`VNXALL` rows are not a readiness failure.",
        "",
        "## Artifact Directories",
        "",
        f"- Tables: `{rel(PAPER_TABLE_DIR)}`.",
        f"- Figures: `{rel(PAPER_FIGURE_DIR)}`.",
        f"- Notes: `{rel(PAPER_NOTE_DIR)}`.",
        "",
        "## Citation Styles Needed",
        "",
        "- Vietnamese APA.",
        "- Vietnamese IEEE.",
        "- English APA.",
        "- English IEEE.",
        "",
        "## Expected DOCX Outputs If Final Paper Exists",
        "",
        "- `NCKH_VN30_HOURLY_AVAILABLE_WINDOW_FULL_PAPER_VI_APA.docx`",
        "- `NCKH_VN30_HOURLY_AVAILABLE_WINDOW_FULL_PAPER_VI_IEEE.docx`",
        "- `NCKH_VN30_HOURLY_AVAILABLE_WINDOW_FULL_PAPER_EN_APA.docx`",
        "- `NCKH_VN30_HOURLY_AVAILABLE_WINDOW_FULL_PAPER_EN_IEEE.docx`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


__all__ = [
    "AUDIT_DIR",
    "BENCHMARK_OUTPUT_DIR",
    "CONFIDENCE_DIR",
    "COST_SLIPPAGE_DIR",
    "DEFAULT_HORIZONS",
    "DEFAULT_MODELS",
    "DESIGN_DECISION_JSON",
    "DESIGN_DECISION_MD",
    "FREQUENCY",
    "MIN_COMMON_TIMESTAMPS",
    "MIN_EVAL_ROWS_PER_TICKER",
    "MIN_TRAIN_ROWS_PER_TICKER",
    "PAPER_FIGURE_DIR",
    "PAPER_NOTE_DIR",
    "PAPER_TABLE_DIR",
    "REGIME_DIR",
    "REPORT_ROOT",
    "TARGET_MODE",
    "TRAIN_FRACTION",
    "VN30_TICKERS",
    "excluded_tickers",
    "final_paper_can_proceed",
    "load_available_window_predictions",
    "load_design_decision",
    "load_selected_hourly_frame",
    "markdown_table",
    "read_csv_rows",
    "rel",
    "save_placeholder_figure",
    "selected_tickers",
    "timestamp_text",
    "write_csv",
    "write_docx_build_notes",
    "write_json",
]
