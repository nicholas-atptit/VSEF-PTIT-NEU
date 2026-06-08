"""Canonical offline forecast panel contract."""

from __future__ import annotations

import pandas as pd

from src.governance.claim_boundary import claim_label

FORECAST_PANEL_COLUMNS = (
    "asset_code",
    "asset_type",
    "asof_timestamp",
    "target_timestamp",
    "horizon",
    "direction_probability",
    "predicted_direction",
    "predicted_return",
    "predicted_close_low",
    "predicted_close_mid",
    "predicted_close_high",
    "predicted_range_pct",
    "rank_score",
    "actual_direction",
    "actual_return",
    "actual_close",
    "claim_label",
)


def build_forecast_panel(rows: pd.DataFrame | list[dict[str, object]]) -> pd.DataFrame:
    panel = pd.DataFrame(rows).copy()
    for column in FORECAST_PANEL_COLUMNS:
        if column not in panel:
            panel[column] = pd.NA
    panel["claim_label"] = panel["claim_label"].fillna(claim_label())
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
