from __future__ import annotations

from typing import Any

import pandas as pd


def allocation_candidates(*records: dict[str, Any]) -> pd.DataFrame:
    base = {
        "candidate_id": "decision_lane_v2|packet_001",
        "source_packet_id": "packet_001",
        "timestamp": "2026-01-02",
        "ticker": "AAA",
        "horizon": 5,
        "target_type": "forward_return",
        "run_mode": "research_core",
        "core_run_id": "small_banks_h05_forward_return",
        "candidate_status": "diagnostic_candidate",
        "risk_adjusted_confidence": 0.80,
        "risk_adjusted_candidate_score": 0.08,
        "risk_score": 0.20,
        "risk_level": "level_1_soft_adjustment",
        "risk_action": "pass",
        "disagreement_score": 0.10,
        "dominance_score": 0.35,
        "dominant_scenario": "bull",
        "dominant_scenario_probability": 0.58,
        "scenario_alignment": "aligned",
        "scenario_confidence_bucket": "medium",
        "reason_codes": "scenario_aligned|high_model_agreement",
    }
    rows = []
    for index, updates in enumerate(records or ({},), start=1):
        packet_id = f"packet_{index:03d}"
        row = {
            **base,
            "candidate_id": f"decision_lane_v2|{packet_id}",
            "source_packet_id": packet_id,
            **updates,
        }
        rows.append(row)
    return pd.DataFrame(rows)
