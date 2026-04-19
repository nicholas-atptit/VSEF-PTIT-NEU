"""Logic for deterministic summary generation and outcome attachment for historical cases."""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_deterministic_summary(packet_data: Dict[str, Any]) -> str:
    """Build a structured, retrieval-friendly summary string."""
    
    ticker = packet_data.get("ticker", "UNKNOWN")
    horizon = packet_data.get("horizon", 0)
    target_type = packet_data.get("target_type", "unknown_target")
    regime = packet_data.get("regime_label")
    if regime is None and "regime_summary" in packet_data:
        rem_sum = packet_data["regime_summary"]
        if isinstance(rem_sum, dict):
            regime = rem_sum.get("regime_label", "unknown")
        else:
            regime = "unknown"
    else:
        regime = regime or "unknown"
    volatility = packet_data.get("volatility_bucket", "unknown")
    agreement = packet_data.get("agreement_bucket", "unknown")
    
    # Derivations
    prediction = packet_data.get("primary_prediction")
    threshold_status = "threshold passed" if abs(prediction or 0) >= 0.01 else "threshold not met"
    
    policy_summary = packet_data.get("policy_summary")
    if isinstance(policy_summary, str):
        # Handle JSON strings if necessary, though packets usually have them parsed
        import json
        try:
            policy_summary = json.loads(policy_summary)
        except:
            policy_summary = {}
    
    pos_size = policy_summary.get("mean_position_size") if isinstance(policy_summary, dict) else None
    policy_action = "no position taken"
    if pos_size is not None:
        if pos_size == 0:
            policy_action = "position zeroed by risk"
        elif pos_size < 1.0:
            policy_action = "size reduced by risk policy"
        else:
            policy_action = "full position recommended"

    return (
        f"Ticker {ticker}, horizon {horizon}, {target_type}, "
        f"{regime} regime, {volatility} volatility, {agreement} agreement, "
        f"{threshold_status}, {policy_action}."
    )


def attach_realized_outcomes(packet_data: Dict[str, Any]) -> Dict[str, Any]:
    """Attach outcome fields if available from source artifacts."""
    
    y_true = packet_data.get("realized_y_true")
    if y_true is None:
        return {
            "realized_return": None,
            "realized_direction": None,
            "realized_outcome_label": None
        }
    
    direction = 1 if y_true > 0 else (-1 if y_true < 0 else 0)
    
    # Simple labels for analysis
    if abs(y_true) < 0.005:
        label = "flat"
    else:
        label = "gain" if y_true > 0 else "loss"
        
    return {
        "realized_return": float(y_true),
        "realized_direction": direction,
        "realized_outcome_label": label
    }


def build_case_tags(packet_data: Dict[str, Any]) -> list[str]:
    """Generate structured tags for retrieval filtering."""
    tags = [
        f"ticker:{packet_data.get('ticker')}",
        f"horizon:{packet_data.get('horizon')}",
        f"regime:{packet_data.get('regime_label')}",
        f"volatility:{packet_data.get('volatility_bucket')}",
        f"agreement:{packet_data.get('agreement_bucket')}",
        f"mode:{packet_data.get('run_mode')}"
    ]
    if packet_data.get("sign_conflict"):
        tags.append("conflict:sign")
    return tags
