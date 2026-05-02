"""Manifest helpers for Risk Governance Layer v1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.risk_governance.schema import (
    RISK_GOVERNANCE_ARTIFACT_FILENAMES,
    RISK_GOVERNANCE_SUMMARY_COLUMNS,
    RiskGovernanceConfig,
)


def _deterministic_generated_at(summary: pd.DataFrame) -> str | None:
    if summary.empty or "timestamp" not in summary.columns:
        return None
    timestamps = pd.to_datetime(summary["timestamp"], errors="coerce").dropna()
    if timestamps.empty:
        return None
    return pd.Timestamp(timestamps.max()).isoformat()


def build_risk_manifest(
    *,
    config: RiskGovernanceConfig,
    artifact_paths: dict[str, str],
    input_row_counts: dict[str, int],
    risk_governance_summary: pd.DataFrame,
    risk_adjusted_candidates: pd.DataFrame,
    risk_override_log: pd.DataFrame,
) -> dict[str, Any]:
    """Build a deterministic manifest for risk governance artifacts."""

    return {
        "manifest_type": "risk_governance_layer_v1_manifest",
        "version": config.version,
        "scoring_weights": dict(config.scoring_weights),
        "thresholds": {
            **dict(config.score_thresholds),
            "weak_candidate_score_threshold": config.weak_candidate_score_threshold,
            "pass_risk_score_threshold": config.pass_risk_score_threshold,
            "volatility_reference": config.volatility_reference,
            "downside_risk_reference": config.downside_risk_reference,
            "calibration_error_reference": config.calibration_error_reference,
        },
        "risk_levels": [
            "level_1_soft_adjustment",
            "level_2_candidate_filtering",
            "level_3_hard_override",
        ],
        "risk_actions": [
            "pass",
            "adjust_confidence",
            "reduce_candidate",
            "block_candidate",
            "force_hold",
        ],
        "required_fields": list(RISK_GOVERNANCE_SUMMARY_COLUMNS),
        "artifact_filenames": dict(RISK_GOVERNANCE_ARTIFACT_FILENAMES),
        "artifact_paths": dict(artifact_paths),
        "input_row_counts": dict(input_row_counts),
        "output_row_counts": {
            "risk_governance_summary": int(len(risk_governance_summary)),
            "risk_adjusted_candidates": int(len(risk_adjusted_candidates)),
            "risk_override_log": int(len(risk_override_log)),
        },
        "authority": "diagnostic_only_no_buy_sell_recommendation_authority",
        "diagnostic_only_authority": True,
        "no_buy_sell_recommendation_authority": True,
        "generated_at": _deterministic_generated_at(risk_governance_summary),
    }


def write_risk_manifest(output_dir: str | Path, manifest: dict[str, Any]) -> Path:
    """Write risk_manifest.json."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / RISK_GOVERNANCE_ARTIFACT_FILENAMES["risk_manifest"]
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path
