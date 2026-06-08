"""Canonical offline forecast panel contract."""

from __future__ import annotations

import pandas as pd

from src.governance.claim_boundary import claim_label

FORECAST_PANEL_COLUMNS = (
    "forecast_id",
    "run_timestamp",
    "asof_timestamp",
    "asset_code",
    "asset_type",
    "horizon",
    "target_timestamp",
    "direction_model_id",
    "direction_target",
    "direction_probability",
    "predicted_direction",
    "direction_confidence_label",
    "return_model_id",
    "predicted_return",
    "predicted_log_return",
    "predicted_close_mid",
    "range_model_id",
    "predicted_return_p10",
    "predicted_return_p50",
    "predicted_return_p90",
    "predicted_close_low",
    "predicted_close_high",
    "predicted_low_price",
    "predicted_high_price",
    "predicted_range_pct",
    "ranking_model_id",
    "rank_score",
    "cross_sectional_rank",
    "actual_return",
    "actual_close",
    "actual_high",
    "actual_low",
    "correct_direction",
    "interval_hit",
    "claim_label",
)


def build_forecast_panel(rows: pd.DataFrame | list[dict[str, object]]) -> pd.DataFrame:
    panel = pd.DataFrame(rows).copy()
    for column in FORECAST_PANEL_COLUMNS:
        if column not in panel:
            panel[column] = pd.NA
    panel["claim_label"] = panel["claim_label"].fillna("offline_diagnostic_forecast_only")
    panel["asof_timestamp"] = pd.to_datetime(panel["asof_timestamp"], errors="coerce")
    panel["target_timestamp"] = pd.to_datetime(panel["target_timestamp"], errors="coerce")
    panel = panel[list(FORECAST_PANEL_COLUMNS)]
    validate_forecast_panel(panel)
    return panel


def validate_forecast_panel(panel: pd.DataFrame) -> None:
    missing = set(FORECAST_PANEL_COLUMNS).difference(panel.columns)
    if missing:
        raise ValueError(f"Forecast panel is missing columns: {sorted(missing)}")
    invalid_time = panel["asof_timestamp"].notna() & panel["target_timestamp"].notna() & (panel["target_timestamp"] <= panel["asof_timestamp"])
    if invalid_time.any():
        raise ValueError("target_timestamp must be after asof_timestamp")
    probability = pd.to_numeric(panel["direction_probability"], errors="coerce").dropna()
    if not probability.between(0.0, 1.0).all():
        raise ValueError("direction_probability must be between 0 and 1")
    low = pd.to_numeric(panel["predicted_close_low"], errors="coerce")
    high = pd.to_numeric(panel["predicted_close_high"], errors="coerce")
    if (low.notna() & high.notna() & (low > high)).any():
        raise ValueError("predicted_close_low exceeds predicted_close_high")
