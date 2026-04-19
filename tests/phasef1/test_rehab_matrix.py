from __future__ import annotations

from src.evaluation.forecast_rehab import build_forecast_rehab_core_frame, build_forecast_rehab_matrix_config


def test_forecast_rehab_matrix_contains_expected_axes() -> None:
    config = build_forecast_rehab_matrix_config("smoke")

    assert config["preset"] == "smoke"
    assert config["target_names"] == ["forward_return", "forward_log_return", "direction_binary"]
    assert "current_full" in config["feature_families"]
    assert config["policy_baseline"]["policy_variant"] == "regime_threshold_adaptive_drawdown"


def test_forecast_rehab_core_frame_expands_groups_horizons_targets_and_families() -> None:
    config = build_forecast_rehab_matrix_config("smoke")
    frame = build_forecast_rehab_core_frame(config)

    expected_rows = (
        len(config["ticker_groups"])
        * len(config["horizons"])
        * len(config["target_names"])
        * len(config["feature_families"])
    )
    assert len(frame) == expected_rows
    assert {"core_run_id", "group_name", "horizon", "target_name", "feature_family", "target_tradable"} <= set(frame.columns)
    assert frame["target_tradable"].isin([True, False]).all()
