from src.forecasting.panels import FORECAST_PANEL_COLUMNS, build_forecast_panel


def test_forecast_panel_has_v1_machine_readable_schema():
    panel = build_forecast_panel([{"asset_code": "VN30", "asset_type": "index", "asof_timestamp": "2026-01-01", "target_timestamp": "2026-01-02", "horizon": 5}])
    assert tuple(panel.columns) == FORECAST_PANEL_COLUMNS
    assert panel.iloc[0]["claim_label"] == "offline_diagnostic_forecast_only"
