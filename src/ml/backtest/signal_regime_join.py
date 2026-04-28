"""Join precomputed regime labels onto saved prediction outputs.

The helpers in this module enrich copies of prediction CSVs for downstream
signal-effectiveness diagnostics. They do not infer regimes, fetch data, or
prove that a regime source is leakage-free. They only normalize join keys,
attach an existing regime label, and report coverage/governance flags.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

JOIN_MODE_DATE = "date"
JOIN_MODE_TICKER_DATE = "ticker_date"
SUPPORTED_JOIN_MODES = {JOIN_MODE_DATE, JOIN_MODE_TICKER_DATE}

DEFAULT_PREDICTION_DATE_COLUMN = "prediction_date"
DEFAULT_REGIME_DATE_COLUMN = "date"
DEFAULT_TICKER_COLUMN = "ticker"
OUTPUT_REGIME_COLUMN = "regime"

REGIME_COLUMN_CANDIDATES = ["regime", "market_regime", "regime_label", "market_state"]
SUSPICIOUS_REGIME_COLUMN_PATTERNS = [
    "future",
    "forward",
    "lead",
    "target",
    "realized",
    "actual_return",
    "future_return",
]

JOIN_GOVERNANCE_SAFE_IF_TRAILING = "safe_if_regime_source_is_trailing"
JOIN_GOVERNANCE_REQUIRES_SOURCE_REVIEW = "requires_source_review"
JOIN_GOVERNANCE_SCHEMA_INVALID = "schema_invalid"

JOIN_SUMMARY_FIELDS = [
    "schema_valid",
    "join_governance",
    "join_mode",
    "prediction_rows",
    "regime_rows",
    "enriched_rows",
    "matched_prediction_rows",
    "unmatched_prediction_rows",
    "matched_rate",
    "min_prediction_date",
    "max_prediction_date",
    "min_regime_date",
    "max_regime_date",
    "prediction_date_column",
    "regime_date_column",
    "ticker_column",
    "regime_column_used",
    "output_regime_column",
    "overwrite_regime",
    "existing_regime_column_present",
    "existing_regime_values_preserved",
    "join_applied",
    "duplicate_regime_keys_exist",
    "duplicate_regime_key_count",
    "duplicate_regime_key_sample",
    "suspicious_columns_present",
    "suspicious_columns",
    "regime_label_distribution",
]

_PREDICTION_JOIN_DATE = "__prediction_join_date"
_REGIME_JOIN_DATE = "__regime_join_date"
_PREDICTION_JOIN_TICKER = "__prediction_join_ticker"
_REGIME_JOIN_TICKER = "__regime_join_ticker"
_JOINED_REGIME_VALUE = "__joined_regime_value"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_frame(source: str | Path | pd.DataFrame, *, source_name: str) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"{source_name} CSV does not exist: {path}")
    return pd.read_csv(path)


def load_prediction_frame(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    return _load_frame(source, source_name="Prediction")


def load_regime_frame(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    return _load_frame(source, source_name="Regime")


def _normalized_dates(frame: pd.DataFrame, column: str, *, source_name: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"{JOIN_GOVERNANCE_SCHEMA_INVALID}: {source_name} missing required date column: {column}")
    dates = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    return dates


def _normalized_ticker(frame: pd.DataFrame, column: str, *, source_name: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(
            f"{JOIN_GOVERNANCE_SCHEMA_INVALID}: "
            f"{source_name} missing required ticker column for ticker_date join: {column}"
        )
    return frame[column].astype(str).str.upper().str.strip()


def detect_regime_column(frame: pd.DataFrame, requested: str | None = None) -> str:
    if requested is not None and str(requested).strip():
        column = str(requested).strip()
        if column not in frame.columns:
            raise ValueError(f"{JOIN_GOVERNANCE_SCHEMA_INVALID}: Regime input missing requested regime column: {column}")
        return column
    lower_lookup = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in REGIME_COLUMN_CANDIDATES:
        resolved = lower_lookup.get(candidate.lower())
        if resolved is not None:
            return resolved
    raise ValueError(
        f"{JOIN_GOVERNANCE_SCHEMA_INVALID}: Regime input missing regime label column. Expected one of: "
        + ", ".join(REGIME_COLUMN_CANDIDATES)
    )


def _suspicious_columns(columns: pd.Index) -> list[str]:
    suspicious: list[str] = []
    for column in columns:
        normalized = str(column).lower()
        if any(pattern in normalized for pattern in SUSPICIOUS_REGIME_COLUMN_PATTERNS):
            suspicious.append(str(column))
    return sorted(dict.fromkeys(suspicious))


def _date_bounds(values: pd.Series) -> tuple[str | None, str | None]:
    clean = pd.to_datetime(values, errors="coerce").dropna()
    if clean.empty:
        return None, None
    return str(pd.Timestamp(clean.min()).date()), str(pd.Timestamp(clean.max()).date())


def _distribution(values: pd.Series) -> dict[str, int]:
    clean = values.dropna().astype(str).str.strip()
    clean = clean[clean.ne("")]
    return {str(label): int(count) for label, count in clean.value_counts().sort_index().items()}


def _summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for field in JOIN_SUMMARY_FIELDS:
        value = summary.get(field)
        if isinstance(value, (dict, list)):
            row[field] = json.dumps(value, sort_keys=True, default=_json_default)
        else:
            row[field] = value
    return row


def _duplicate_key_summary(regime: pd.DataFrame, key_columns: list[str]) -> tuple[bool, int, list[dict[str, Any]]]:
    duplicate_mask = regime.duplicated(key_columns, keep=False)
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count == 0:
        return False, 0, []
    sample = regime.loc[duplicate_mask, key_columns].drop_duplicates().head(10).copy()
    for column in sample.columns:
        if pd.api.types.is_datetime64_any_dtype(sample[column]):
            sample[column] = sample[column].dt.strftime("%Y-%m-%d")
    return True, duplicate_count, sample.to_dict(orient="records")


def _build_summary(
    *,
    predictions: pd.DataFrame,
    regime: pd.DataFrame,
    enriched: pd.DataFrame,
    matched_rows: int,
    duplicate_exists: bool,
    duplicate_count: int,
    duplicate_sample: list[dict[str, Any]],
    suspicious_columns: list[str],
    join_mode: str,
    prediction_date_column: str,
    regime_date_column: str,
    ticker_column: str,
    regime_column_used: str,
    overwrite_regime: bool,
    existing_regime_column_present: bool,
    existing_regime_values_preserved: bool,
    join_applied: bool,
    prediction_dates: pd.Series,
    regime_dates: pd.Series,
) -> dict[str, Any]:
    prediction_min, prediction_max = _date_bounds(prediction_dates)
    regime_min, regime_max = _date_bounds(regime_dates)
    prediction_rows = int(len(predictions))
    matched_prediction_rows = int(matched_rows)
    unmatched_prediction_rows = int(prediction_rows - matched_prediction_rows)
    flags_present = bool(duplicate_exists or suspicious_columns)
    governance = JOIN_GOVERNANCE_REQUIRES_SOURCE_REVIEW if flags_present else JOIN_GOVERNANCE_SAFE_IF_TRAILING
    summary = {
        "schema_valid": True,
        "join_governance": governance,
        "join_mode": join_mode,
        "prediction_rows": prediction_rows,
        "regime_rows": int(len(regime)),
        "enriched_rows": int(len(enriched)),
        "matched_prediction_rows": matched_prediction_rows,
        "unmatched_prediction_rows": unmatched_prediction_rows,
        "matched_rate": float(matched_prediction_rows / prediction_rows) if prediction_rows else np.nan,
        "min_prediction_date": prediction_min,
        "max_prediction_date": prediction_max,
        "min_regime_date": regime_min,
        "max_regime_date": regime_max,
        "prediction_date_column": prediction_date_column,
        "regime_date_column": regime_date_column,
        "ticker_column": ticker_column if join_mode == JOIN_MODE_TICKER_DATE else None,
        "regime_column_used": regime_column_used,
        "output_regime_column": OUTPUT_REGIME_COLUMN,
        "overwrite_regime": bool(overwrite_regime),
        "existing_regime_column_present": bool(existing_regime_column_present),
        "existing_regime_values_preserved": bool(existing_regime_values_preserved),
        "join_applied": bool(join_applied),
        "duplicate_regime_keys_exist": bool(duplicate_exists),
        "duplicate_regime_key_count": int(duplicate_count),
        "duplicate_regime_key_sample": duplicate_sample,
        "suspicious_columns_present": bool(suspicious_columns),
        "suspicious_columns": suspicious_columns,
        "regime_label_distribution": _distribution(enriched[OUTPUT_REGIME_COLUMN])
        if OUTPUT_REGIME_COLUMN in enriched.columns
        else {},
    }
    return {field: summary.get(field) for field in JOIN_SUMMARY_FIELDS}


def join_regime_labels(
    predictions: str | Path | pd.DataFrame,
    regime: str | Path | pd.DataFrame,
    *,
    join_mode: str = JOIN_MODE_DATE,
    prediction_date_column: str = DEFAULT_PREDICTION_DATE_COLUMN,
    regime_date_column: str = DEFAULT_REGIME_DATE_COLUMN,
    ticker_column: str = DEFAULT_TICKER_COLUMN,
    regime_column: str | None = None,
    overwrite_regime: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if join_mode not in SUPPORTED_JOIN_MODES:
        raise ValueError(f"Unsupported join_mode={join_mode!r}. Supported: {sorted(SUPPORTED_JOIN_MODES)}")

    prediction_frame = load_prediction_frame(predictions)
    regime_frame = load_regime_frame(regime)
    prediction_dates = _normalized_dates(
        prediction_frame,
        prediction_date_column,
        source_name="Predictions",
    )
    regime_dates = _normalized_dates(regime_frame, regime_date_column, source_name="Regime input")
    regime_column_used = detect_regime_column(regime_frame, requested=regime_column)

    working_predictions = prediction_frame.copy()
    working_regime = regime_frame.copy()
    working_predictions[_PREDICTION_JOIN_DATE] = prediction_dates
    working_regime[_REGIME_JOIN_DATE] = regime_dates

    prediction_key_columns = [_PREDICTION_JOIN_DATE]
    regime_key_columns = [_REGIME_JOIN_DATE]
    if join_mode == JOIN_MODE_TICKER_DATE:
        working_predictions[_PREDICTION_JOIN_TICKER] = _normalized_ticker(
            working_predictions,
            ticker_column,
            source_name="Predictions",
        )
        working_regime[_REGIME_JOIN_TICKER] = _normalized_ticker(
            working_regime,
            ticker_column,
            source_name="Regime input",
        )
        prediction_key_columns.append(_PREDICTION_JOIN_TICKER)
        regime_key_columns.append(_REGIME_JOIN_TICKER)

    duplicate_exists, duplicate_count, duplicate_sample = _duplicate_key_summary(working_regime, regime_key_columns)
    suspicious_columns = _suspicious_columns(regime_frame.columns)

    join_source = working_regime[[*regime_key_columns, regime_column_used]].copy()
    join_source = join_source.rename(columns={regime_column_used: _JOINED_REGIME_VALUE})
    join_source = join_source.drop_duplicates(subset=regime_key_columns, keep="first")

    merged = working_predictions.merge(
        join_source,
        left_on=prediction_key_columns,
        right_on=regime_key_columns,
        how="left",
        sort=False,
    )
    matched_rows = int(merged[_JOINED_REGIME_VALUE].notna().sum())
    existing_regime_column_present = OUTPUT_REGIME_COLUMN in prediction_frame.columns
    existing_regime_values_preserved = bool(existing_regime_column_present and not overwrite_regime)
    join_applied = bool(overwrite_regime or not existing_regime_column_present)

    if join_applied:
        merged[OUTPUT_REGIME_COLUMN] = merged[_JOINED_REGIME_VALUE]
    elif OUTPUT_REGIME_COLUMN not in merged.columns:
        merged[OUTPUT_REGIME_COLUMN] = np.nan

    temp_columns = [
        _PREDICTION_JOIN_DATE,
        _REGIME_JOIN_DATE,
        _PREDICTION_JOIN_TICKER,
        _REGIME_JOIN_TICKER,
        _JOINED_REGIME_VALUE,
    ]
    enriched = merged.drop(columns=[column for column in temp_columns if column in merged.columns])
    enriched = enriched.reindex(columns=[*prediction_frame.columns, *([] if OUTPUT_REGIME_COLUMN in prediction_frame.columns else [OUTPUT_REGIME_COLUMN])])

    summary = _build_summary(
        predictions=prediction_frame,
        regime=regime_frame,
        enriched=enriched,
        matched_rows=matched_rows,
        duplicate_exists=duplicate_exists,
        duplicate_count=duplicate_count,
        duplicate_sample=duplicate_sample,
        suspicious_columns=suspicious_columns,
        join_mode=join_mode,
        prediction_date_column=prediction_date_column,
        regime_date_column=regime_date_column,
        ticker_column=ticker_column,
        regime_column_used=regime_column_used,
        overwrite_regime=overwrite_regime,
        existing_regime_column_present=existing_regime_column_present,
        existing_regime_values_preserved=existing_regime_values_preserved,
        join_applied=join_applied,
        prediction_dates=prediction_dates,
        regime_dates=regime_dates,
    )
    return enriched, summary


def write_enriched_predictions(enriched: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(path, index=False)
    return path


def write_join_summary(summary: dict[str, Any], summary_path: str | Path) -> Path:
    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        pd.DataFrame([_summary_row(summary)]).reindex(columns=JOIN_SUMMARY_FIELDS).to_csv(path, index=False)
    else:
        path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    return path


def join_regime_csvs(
    *,
    predictions_path: str | Path,
    regime_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
    join_mode: str = JOIN_MODE_DATE,
    prediction_date_column: str = DEFAULT_PREDICTION_DATE_COLUMN,
    regime_date_column: str = DEFAULT_REGIME_DATE_COLUMN,
    ticker_column: str = DEFAULT_TICKER_COLUMN,
    regime_column: str | None = None,
    overwrite_regime: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    enriched, summary = join_regime_labels(
        predictions_path,
        regime_path,
        join_mode=join_mode,
        prediction_date_column=prediction_date_column,
        regime_date_column=regime_date_column,
        ticker_column=ticker_column,
        regime_column=regime_column,
        overwrite_regime=overwrite_regime,
    )
    write_enriched_predictions(enriched, output_path)
    write_join_summary(summary, summary_path)
    return enriched, summary


__all__ = [
    "DEFAULT_PREDICTION_DATE_COLUMN",
    "DEFAULT_REGIME_DATE_COLUMN",
    "DEFAULT_TICKER_COLUMN",
    "JOIN_GOVERNANCE_REQUIRES_SOURCE_REVIEW",
    "JOIN_GOVERNANCE_SAFE_IF_TRAILING",
    "JOIN_GOVERNANCE_SCHEMA_INVALID",
    "JOIN_MODE_DATE",
    "JOIN_MODE_TICKER_DATE",
    "JOIN_SUMMARY_FIELDS",
    "OUTPUT_REGIME_COLUMN",
    "REGIME_COLUMN_CANDIDATES",
    "SUPPORTED_JOIN_MODES",
    "SUSPICIOUS_REGIME_COLUMN_PATTERNS",
    "detect_regime_column",
    "join_regime_csvs",
    "join_regime_labels",
    "load_prediction_frame",
    "load_regime_frame",
    "write_enriched_predictions",
    "write_join_summary",
]
