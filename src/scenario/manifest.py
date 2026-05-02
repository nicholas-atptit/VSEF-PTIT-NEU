"""Scenario engine manifest helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.scenario.schema import (
    SCENARIO_ARTIFACT_FILENAMES,
    SCENARIO_LABELS,
    SCENARIO_REQUIRED_FIELDS,
    ScenarioEngineConfig,
)


def _deterministic_generated_at(probability_df: pd.DataFrame) -> str | None:
    if probability_df.empty or "timestamp" not in probability_df.columns:
        return None
    timestamps = pd.to_datetime(probability_df["timestamp"], errors="coerce").dropna()
    if timestamps.empty:
        return None
    return pd.Timestamp(timestamps.max()).isoformat()


def build_scenario_manifest(
    *,
    config: ScenarioEngineConfig,
    artifact_paths: dict[str, str],
    scenario_probability: pd.DataFrame,
    scenario_rankings: pd.DataFrame,
    scenario_dominance_summary: pd.DataFrame,
    scenario_uncertainty_summary: pd.DataFrame,
    scenario_calibration_summary: pd.DataFrame,
    source_counts: dict[str, int],
) -> dict[str, Any]:
    """Build a deterministic manifest for scenario artifacts."""

    probability_sums: dict[str, float] = {}
    if not scenario_probability.empty:
        group_columns = [
            column
            for column in ("timestamp", "ticker", "horizon", "target_type", "run_mode", "core_run_id")
            if column in scenario_probability.columns
        ]
        for keys, group in scenario_probability.groupby(group_columns, sort=True, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key = "|".join(str(value) for value in keys)
            probability_sums[key] = float(pd.to_numeric(group["scenario_probability"], errors="coerce").sum())

    return {
        "manifest_type": "scenario_evaluation_engine_v1_manifest",
        "scenario_engine_version": "v1",
        "probability_method": config.probability_method,
        "calibration_lookback": config.calibration_lookback,
        "calibration_bins": config.calibration_bins,
        "scenario_labels": list(SCENARIO_LABELS),
        "dominance_authority": "diagnostic_only_no_buy_sell_recommendation",
        "required_fields": list(SCENARIO_REQUIRED_FIELDS),
        "artifact_filenames": dict(SCENARIO_ARTIFACT_FILENAMES),
        "artifact_paths": dict(artifact_paths),
        "source_counts": dict(source_counts),
        "row_counts": {
            "scenario_probability": int(len(scenario_probability)),
            "scenario_rankings": int(len(scenario_rankings)),
            "scenario_dominance_summary": int(len(scenario_dominance_summary)),
            "scenario_uncertainty_summary": int(len(scenario_uncertainty_summary)),
            "scenario_calibration_summary": int(len(scenario_calibration_summary)),
        },
        "probability_sum_by_context": probability_sums,
        "generated_at": _deterministic_generated_at(scenario_probability),
    }


def write_scenario_manifest(output_dir: str | Path, manifest: dict[str, Any]) -> Path:
    """Write scenario_manifest.json."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / SCENARIO_ARTIFACT_FILENAMES["scenario_manifest"]
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path
