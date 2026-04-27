"""Context availability coverage diagnostics for walk-forward feature frames."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CONTEXT_COVERAGE_DIAGNOSTIC_COLUMNS = [
    "ticker",
    "fold_id",
    "step_size",
    "forecast_sequence_index",
    "prediction_date",
    "horizon",
    "row_count",
    "breadth_available_count",
    "breadth_missing_count",
    "breadth_available_rate",
    "breadth_missing_rate",
    "foreign_flow_available_count",
    "foreign_flow_missing_count",
    "foreign_flow_available_rate",
    "foreign_flow_missing_rate",
    "foreign_flow_context_mode",
    "foreign_flow_coverage_status",
    "coverage_warning_level",
    "coverage_metadata_status",
    "coverage_note",
    "train_start",
    "train_end",
    "eval_start",
    "eval_end",
]

CONTEXT_COVERAGE_SUMMARY_COLUMNS = [
    "ticker",
    "horizon",
    "fold_count",
    "mean_breadth_missing_rate",
    "max_breadth_missing_rate",
    "mean_foreign_flow_missing_rate",
    "max_foreign_flow_missing_rate",
    "weak_coverage_fold_count",
    "review_fold_count",
    "overall_coverage_warning_level",
]

WARNING_LEVEL_ORDER = {
    "ok": 0,
    "metadata_unavailable": 1,
    "review": 2,
    "weak_coverage": 3,
}


def empty_context_coverage_diagnostics_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CONTEXT_COVERAGE_DIAGNOSTIC_COLUMNS)


def empty_context_coverage_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CONTEXT_COVERAGE_SUMMARY_COLUMNS)


def coverage_warning_level(*rates: float) -> str:
    valid_rates = [float(rate) for rate in rates if pd.notna(rate)]
    if not valid_rates:
        return "metadata_unavailable"
    max_missing_rate = max(valid_rates)
    if max_missing_rate > 0.25:
        return "weak_coverage"
    if max_missing_rate > 0.05:
        return "review"
    return "ok"


def _context_counts(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    row_count = int(len(frame))
    available_col = f"{prefix}_context_available"
    missing_col = f"{prefix}_context_missing"
    mode_col = f"{prefix}_context_mode"
    status_col = f"{prefix}_coverage_status"
    if prefix == "foreign_flow":
        mode_values = (
            set(frame[mode_col].dropna().astype(str).str.lower())
            if mode_col in frame.columns
            else set()
        )
        status_values = (
            set(frame[status_col].dropna().astype(str).str.lower())
            if status_col in frame.columns
            else set()
        )
        if "disabled" in mode_values or "disabled" in status_values:
            return {
                "available_count": 0,
                "missing_count": 0,
                "available_rate": np.nan,
                "missing_rate": np.nan,
                "metadata_available": True,
                "context_mode": "disabled",
                "coverage_status": "disabled",
            }
    if available_col not in frame.columns and missing_col not in frame.columns:
        return {
            "available_count": np.nan,
            "missing_count": np.nan,
            "available_rate": np.nan,
            "missing_rate": np.nan,
            "metadata_available": False,
            "context_mode": "auto",
            "coverage_status": "metadata_missing",
        }

    if available_col in frame.columns:
        available_series = frame[available_col].fillna(False).astype(bool)
        available_count = int(available_series.sum())
    else:
        missing_series = frame[missing_col].fillna(False).astype(bool)
        available_count = int(row_count - int(missing_series.sum()))

    if missing_col in frame.columns:
        missing_series = frame[missing_col].fillna(False).astype(bool)
        missing_count = int(missing_series.sum())
    else:
        missing_count = int(row_count - available_count)

    return {
        "available_count": available_count,
        "missing_count": missing_count,
        "available_rate": float(available_count / row_count) if row_count else np.nan,
        "missing_rate": float(missing_count / row_count) if row_count else np.nan,
        "metadata_available": True,
        "context_mode": "auto",
        "coverage_status": "measured",
    }


def build_context_coverage_rows(
    *,
    feature_frame: pd.DataFrame,
    fold_context: dict[str, Any],
) -> pd.DataFrame:
    if feature_frame is None:
        feature_frame = pd.DataFrame()
    frame = feature_frame.copy()
    row_count = int(len(frame))
    breadth = _context_counts(frame, "breadth")
    foreign_flow = _context_counts(frame, "foreign_flow")
    warning_level = coverage_warning_level(
        breadth["missing_rate"],
        foreign_flow["missing_rate"],
    )

    missing_metadata = []
    if not breadth["metadata_available"]:
        missing_metadata.append("breadth")
    if not foreign_flow["metadata_available"]:
        missing_metadata.append("foreign_flow")
    metadata_status = "available" if not missing_metadata else "missing_" + "_and_".join(missing_metadata)
    if foreign_flow["coverage_status"] == "disabled":
        note = "Foreign-flow context intentionally disabled; missing-rate excluded from warning calculation."
    elif missing_metadata:
        note = f"Missing context availability metadata for: {', '.join(missing_metadata)}."
    else:
        note = "Context availability metadata present."

    row = {
        **fold_context,
        "row_count": row_count,
        "breadth_available_count": breadth["available_count"],
        "breadth_missing_count": breadth["missing_count"],
        "breadth_available_rate": breadth["available_rate"],
        "breadth_missing_rate": breadth["missing_rate"],
        "foreign_flow_available_count": foreign_flow["available_count"],
        "foreign_flow_missing_count": foreign_flow["missing_count"],
        "foreign_flow_available_rate": foreign_flow["available_rate"],
        "foreign_flow_missing_rate": foreign_flow["missing_rate"],
        "foreign_flow_context_mode": foreign_flow["context_mode"],
        "foreign_flow_coverage_status": foreign_flow["coverage_status"],
        "coverage_warning_level": warning_level,
        "coverage_metadata_status": metadata_status,
        "coverage_note": note,
    }
    return pd.DataFrame([row]).reindex(columns=CONTEXT_COVERAGE_DIAGNOSTIC_COLUMNS)


def summarize_context_coverage(coverage_rows: pd.DataFrame) -> pd.DataFrame:
    if coverage_rows is None or coverage_rows.empty:
        return empty_context_coverage_summary_frame()

    working = coverage_rows.copy()
    for column in ("breadth_missing_rate", "foreign_flow_missing_rate"):
        working[column] = pd.to_numeric(working.get(column), errors="coerce")
    if "coverage_warning_level" not in working.columns:
        working["coverage_warning_level"] = "metadata_unavailable"

    rows: list[dict[str, Any]] = []
    for keys, group in working.groupby(["ticker", "horizon"], sort=True):
        levels = [str(value) for value in group["coverage_warning_level"].dropna()]
        overall_level = (
            max(levels, key=lambda value: WARNING_LEVEL_ORDER.get(value, -1))
            if levels
            else "metadata_unavailable"
        )
        rows.append(
            {
                "ticker": keys[0],
                "horizon": keys[1],
                "fold_count": int(group["fold_id"].nunique()) if "fold_id" in group.columns else int(len(group)),
                "mean_breadth_missing_rate": float(group["breadth_missing_rate"].mean())
                if group["breadth_missing_rate"].notna().any()
                else np.nan,
                "max_breadth_missing_rate": float(group["breadth_missing_rate"].max())
                if group["breadth_missing_rate"].notna().any()
                else np.nan,
                "mean_foreign_flow_missing_rate": float(group["foreign_flow_missing_rate"].mean())
                if group["foreign_flow_missing_rate"].notna().any()
                else np.nan,
                "max_foreign_flow_missing_rate": float(group["foreign_flow_missing_rate"].max())
                if group["foreign_flow_missing_rate"].notna().any()
                else np.nan,
                "weak_coverage_fold_count": int((group["coverage_warning_level"] == "weak_coverage").sum()),
                "review_fold_count": int((group["coverage_warning_level"] == "review").sum()),
                "overall_coverage_warning_level": overall_level,
            }
        )

    return pd.DataFrame(rows).reindex(columns=CONTEXT_COVERAGE_SUMMARY_COLUMNS)
