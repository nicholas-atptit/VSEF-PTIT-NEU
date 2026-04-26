"""Foreign-flow artifact validation utilities.

These helpers are read-only governance checks. They classify whether a
foreign-flow frame is suitable for interpreting ticker/date context coverage
for a requested window, but they do not load, fetch, repair, or fabricate data.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

REQUIRED_COLUMNS = {"ticker", "date"}
PROVENANCE_COLUMNS = {"source", "source_date", "retrieved_at", "provider", "coverage_note"}
FIXTURE_TICKERS = {"TEST", "DUMMY", "SAMPLE"}
FLOW_MEASURE_PREFIXES = ("foreign_", "fr_")
NON_MEASURE_COLUMNS = {"ticker", "date", *PROVENANCE_COLUMNS}


def _normalize_tickers(tickers: list[str] | tuple[str, ...] | str | None) -> list[str]:
    if tickers is None:
        return []
    values = tickers.split(",") if isinstance(tickers, str) else list(tickers)
    return [str(value).strip().upper() for value in values if str(value).strip()]


def _prepare_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    frame = df.copy()
    if "time" in frame.columns and "date" not in frame.columns:
        frame = frame.rename(columns={"time": "date"})
    if "ticker" in frame.columns:
        frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    return frame


def _numeric_measure_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        name = str(column)
        if name in NON_MEASURE_COLUMNS:
            continue
        if not name.startswith(FLOW_MEASURE_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(pd.to_numeric(frame[column], errors="coerce")):
            columns.append(name)
    return columns


def summarize_foreign_flow_coverage(
    df: pd.DataFrame | None,
    tickers: list[str] | tuple[str, ...] | str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> dict[str, Any]:
    """Summarize ticker/date coverage for a foreign-flow artifact."""

    frame = _prepare_frame(df)
    requested_tickers = _normalize_tickers(tickers)
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    requested_dates = pd.bdate_range(start_ts, end_ts)
    requested_pairs = {(ticker, date) for ticker in requested_tickers for date in requested_dates}

    if frame.empty:
        present_pairs: set[tuple[str, pd.Timestamp]] = set()
        ticker_coverage: list[str] = []
        date_min = None
        date_max = None
    elif {"ticker", "date"} <= set(frame.columns):
        valid = frame.dropna(subset=["ticker", "date"]).copy()
        present_pairs = set(zip(valid["ticker"].astype(str).str.upper(), pd.to_datetime(valid["date"]).dt.normalize()))
        ticker_coverage = sorted(valid["ticker"].astype(str).str.upper().unique().tolist())
        date_min = str(valid["date"].min().date()) if not valid.empty else None
        date_max = str(valid["date"].max().date()) if not valid.empty else None
    else:
        present_pairs = set()
        ticker_coverage = sorted(frame["ticker"].dropna().astype(str).str.upper().unique().tolist()) if "ticker" in frame.columns else []
        date_min = None
        date_max = None

    matched_pairs = requested_pairs & present_pairs
    requested_pair_count = len(requested_pairs)
    coverage_rate = float(len(matched_pairs) / requested_pair_count) if requested_pair_count else 0.0
    fixture_tickers_present = sorted(set(ticker_coverage) & FIXTURE_TICKERS)

    return {
        "row_count": int(len(frame)),
        "requested_tickers": requested_tickers,
        "requested_start_date": str(start_ts.date()),
        "requested_end_date": str(end_ts.date()),
        "requested_business_date_count": int(len(requested_dates)),
        "requested_ticker_date_count": int(requested_pair_count),
        "matched_ticker_date_count": int(len(matched_pairs)),
        "requested_ticker_date_coverage_rate": coverage_rate,
        "ticker_coverage": ticker_coverage,
        "date_min": date_min,
        "date_max": date_max,
        "fixture_tickers_present": fixture_tickers_present,
    }


def validate_foreign_flow_artifact(
    df: pd.DataFrame | None,
    tickers: list[str] | tuple[str, ...] | str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> dict[str, Any]:
    """Validate schema, provenance, and requested-window coverage."""

    frame = _prepare_frame(df)
    columns = set(frame.columns)
    missing_required = sorted(REQUIRED_COLUMNS - columns)
    measure_columns = _numeric_measure_columns(frame)
    missing_provenance = sorted(PROVENANCE_COLUMNS - columns)
    coverage = summarize_foreign_flow_coverage(frame, tickers, start_date, end_date)
    only_fixture_tickers = bool(coverage["ticker_coverage"]) and set(coverage["ticker_coverage"]) <= FIXTURE_TICKERS

    schema_valid = not missing_required and bool(measure_columns)
    if frame.empty:
        classification = "empty_or_missing"
    elif not schema_valid:
        classification = "schema_invalid"
    elif only_fixture_tickers:
        classification = "fixture_only"
    elif coverage["requested_ticker_date_coverage_rate"] >= 1.0:
        classification = "usable_for_requested_window"
    elif coverage["requested_ticker_date_coverage_rate"] > 0.0:
        classification = "partial_coverage"
    else:
        classification = "partial_coverage"

    return {
        **coverage,
        "required_columns_present": not missing_required,
        "missing_required_columns": missing_required,
        "numeric_flow_measure_columns": measure_columns,
        "has_numeric_flow_measure": bool(measure_columns),
        "provenance_columns_present": sorted(PROVENANCE_COLUMNS & columns),
        "missing_provenance_columns": missing_provenance,
        "provenance_complete": not missing_provenance,
        "only_fixture_tickers": only_fixture_tickers,
        "artifact_classification": classification,
        "suitable_for_foreign_feature_interpretation": classification == "usable_for_requested_window",
    }


def classify_foreign_flow_artifact(
    df: pd.DataFrame | None,
    tickers: list[str] | tuple[str, ...] | str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> str:
    """Return only the artifact classification label."""

    return str(validate_foreign_flow_artifact(df, tickers, start_date, end_date)["artifact_classification"])
