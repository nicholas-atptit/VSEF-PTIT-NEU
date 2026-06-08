import pandas as pd
import pytest

from src.forecasting.panels import build_forecast_panel
from src.forecasting.selectors import select_direction, select_range, select_return


def test_forecast_panel_applies_schema_and_claim_label():
    panel = build_forecast_panel(
        [
            {
                "asset_code": "VN30",
                "asset_type": "index",
                "asof_timestamp": "2026-01-01",
                "target_timestamp": "2026-01-02",
                "horizon": 1,
                "direction_probability": 0.6,
                "predicted_close_low": 1000,
                "predicted_close_high": 1100,
            }
        ]
    )
    assert panel.loc[0, "claim_label"] == "offline_diagnostic_only"


def test_direction_selector_uses_validation_only_and_rejects_collapse():
    frame = pd.DataFrame(
        [
            {"model": "final_best", "split": "final", "balanced_accuracy": 0.99, "macro_f1": 0.99, "mcc": 0.99, "prediction_balance": 0.5},
            {"model": "collapsed", "split": "validation", "balanced_accuracy": 0.9, "macro_f1": 0.1, "mcc": 0.0, "prediction_balance": 1.0},
            {"model": "valid", "split": "validation", "balanced_accuracy": 0.7, "macro_f1": 0.7, "mcc": 0.4, "prediction_balance": 0.5},
        ]
    )
    assert select_direction(frame)["model"] == "valid"


def test_return_and_range_selectors_apply_validation_gates():
    returns = pd.DataFrame(
        [
            {"model": "bad", "split": "validation", "beats_baseline": False, "rmse": 0.1, "mae": 0.1},
            {"model": "good", "split": "validation", "beats_baseline": True, "rmse": 0.2, "mae": 0.2},
        ]
    )
    ranges = pd.DataFrame(
        [
            {"model": "narrow", "split": "validation", "interval_coverage": 0.5, "winkler_score": 1.0, "average_interval_width": 1.0},
            {"model": "covered", "split": "validation", "interval_coverage": 0.9, "winkler_score": 2.0, "average_interval_width": 2.0},
        ]
    )
    assert select_return(returns)["model"] == "good"
    assert select_range(ranges)["model"] == "covered"
