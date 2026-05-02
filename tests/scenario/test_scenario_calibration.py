from __future__ import annotations

import pandas as pd
import pytest

from src.scenario.calibration import apply_probability_calibration


def _probability_rows() -> pd.DataFrame:
    rows = []
    realized_values = [0.05, -0.02, -0.03, -0.04]
    for idx, realized in enumerate(realized_values):
        rows.append(
            {
                "scenario_id": f"AAA|2024-01-0{idx + 1}|h05|forward_return|research_core|run|bull",
                "timestamp": pd.Timestamp(f"2024-01-0{idx + 1}"),
                "ticker": "AAA",
                "horizon": 5,
                "target_type": "forward_return",
                "run_mode": "research_core",
                "core_run_id": "run",
                "scenario_label": "bull",
                "scenario_probability": 0.80,
                "confidence_adjusted_probability": 0.80,
                "realized_outcome": realized,
                "realized_available": True,
                "calibration_error": float("nan"),
                "historical_hit_rate": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def test_calibration_bins_compute_observed_frequency() -> None:
    calibrated, summary = apply_probability_calibration(_probability_rows(), bins=5, lookback=None)

    bull_bin = summary[(summary["scenario_label"] == "bull") & (summary["probability_bin"] == "0.80-1.00")].iloc[0]
    assert bull_bin["prediction_count"] == 4
    assert bull_bin["observed_frequency"] == pytest.approx(0.25)
    assert calibrated["historical_hit_rate"].dropna().unique().tolist() == pytest.approx([0.25])


def test_poor_calibration_reduces_adjusted_confidence() -> None:
    calibrated, _ = apply_probability_calibration(_probability_rows(), bins=5, lookback=None)

    assert calibrated["calibration_error"].iloc[0] == pytest.approx(0.55)
    assert calibrated["confidence_adjusted_probability"].max() < 0.80
