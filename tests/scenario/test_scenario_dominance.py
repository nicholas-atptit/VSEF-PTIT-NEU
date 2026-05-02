from __future__ import annotations

import pandas as pd

from src.scenario.dominance import evaluate_scenario_dominance
from src.scenario.schema import SCENARIO_LABELS


def test_dominant_scenario_selected_from_adjusted_probability_gap() -> None:
    probabilities = {
        "bull": 0.55,
        "bear": 0.10,
        "sideway": 0.12,
        "high_volatility": 0.07,
        "drawdown": 0.04,
        "recovery": 0.08,
        "uncertain": 0.04,
    }
    rows = []
    for label in SCENARIO_LABELS:
        rows.append(
            {
                "scenario_id": f"AAA|2024-01-02|h05|forward_return|research_core|run|{label}",
                "timestamp": pd.Timestamp("2024-01-02"),
                "ticker": "AAA",
                "horizon": 5,
                "target_type": "forward_return",
                "run_mode": "research_core",
                "core_run_id": "run",
                "scenario_label": label,
                "scenario_probability": probabilities[label],
                "confidence_adjusted_probability": probabilities[label],
                "uncertainty_score": 0.20,
                "calibration_error": 0.05,
                "downside_risk": 0.02,
            }
        )
    ranked_probability, rankings, summary = evaluate_scenario_dominance(pd.DataFrame(rows))

    assert summary.loc[0, "dominant_scenario"] == "bull"
    assert summary.loc[0, "dominance_label"] == "dominant"
    assert summary.loc[0, "dominant_scenario_flag"]
    top_rank = rankings[rankings["scenario_rank"] == 1].iloc[0]
    assert top_rank["scenario_label"] == "bull"
    assert ranked_probability[ranked_probability["scenario_label"] == "bull"].iloc[0]["dominant_scenario_flag"]
