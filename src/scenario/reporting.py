"""High-level Scenario Evaluation Engine v1 orchestration and artifact writing."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.scenario.calibration import apply_probability_calibration
from src.scenario.dominance import evaluate_scenario_dominance
from src.scenario.manifest import build_scenario_manifest, write_scenario_manifest
from src.scenario.probability import build_scenario_probability_frame
from src.scenario.schema import (
    SCENARIO_ARTIFACT_FILENAMES,
    SCENARIO_CONTEXT_COLUMNS,
    SCENARIO_REQUIRED_FIELDS,
    ScenarioEngineConfig,
    ScenarioEvaluationResult,
    present_columns,
)
from src.scenario.uncertainty import attach_uncertainty_scores


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default, sort_keys=True)


def _shared_columns(left: pd.DataFrame, right: pd.DataFrame, preferred: tuple[str, ...] | list[str]) -> list[str]:
    return [column for column in preferred if column in left.columns and column in right.columns]


def _lookup_by_keys(frame: pd.DataFrame, columns: list[str]) -> dict[tuple[Any, ...], dict[str, Any]]:
    if frame.empty or not columns:
        return {}
    lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        lookup[tuple(row.get(column) for column in columns)] = row
    return lookup


def _groups_by_keys(frame: pd.DataFrame, columns: list[str]) -> dict[tuple[Any, ...], pd.DataFrame]:
    if frame.empty or not columns:
        return {}
    groups: dict[tuple[Any, ...], pd.DataFrame] = {}
    for keys, group in frame.groupby(columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        groups[keys] = group.copy()
    return groups


def _key(row: dict[str, Any], columns: list[str]) -> tuple[Any, ...]:
    return tuple(row.get(column) for column in columns)


def _source_counts(
    *,
    forecasts_df: pd.DataFrame,
    consensus_df: pd.DataFrame | None,
    risk_df: pd.DataFrame | None,
    regime_df: pd.DataFrame | None,
    strategy_metrics_df: pd.DataFrame | None,
    analysis_packets_df: pd.DataFrame | None,
    model_health_df: pd.DataFrame | None,
) -> dict[str, int]:
    return {
        "forecasts": int(len(forecasts_df)),
        "model_consensus_summary": int(len(consensus_df)) if consensus_df is not None else 0,
        "risk_summary": int(len(risk_df)) if risk_df is not None else 0,
        "regime_summary": int(len(regime_df)) if regime_df is not None else 0,
        "strategy_metrics": int(len(strategy_metrics_df)) if strategy_metrics_df is not None else 0,
        "analysis_packets": int(len(analysis_packets_df)) if analysis_packets_df is not None else 0,
        "model_health_summary": int(len(model_health_df)) if model_health_df is not None else 0,
    }


def enrich_analysis_packets_with_scenarios(
    packets_df: pd.DataFrame,
    scenario_rankings: pd.DataFrame,
    dominance_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Attach compact scenario diagnostics to analysis packet rows."""

    if packets_df.empty:
        enriched = packets_df.copy()
        for column in (
            "scenario_summary",
            "dominant_scenario",
            "dominant_scenario_probability",
            "scenario_uncertainty_score",
            "scenario_dominance_score",
            "scenario_calibration_error",
            "scenario_confidence_bucket",
            "alternative_scenarios",
        ):
            if column not in enriched.columns:
                enriched[column] = pd.Series(dtype=object)
        return enriched

    enriched = packets_df.copy()
    shared = _shared_columns(enriched, dominance_summary, SCENARIO_CONTEXT_COLUMNS)
    ranking_shared = _shared_columns(enriched, scenario_rankings, SCENARIO_CONTEXT_COLUMNS)
    dominance_lookup = _lookup_by_keys(dominance_summary, shared)
    ranking_groups = _groups_by_keys(scenario_rankings, ranking_shared)

    rows: list[dict[str, Any]] = []
    for record in enriched.to_dict(orient="records"):
        summary = dominance_lookup.get(_key(record, shared), {}) if shared else {}
        ranking = ranking_groups.get(_key(record, ranking_shared), pd.DataFrame()) if ranking_shared else pd.DataFrame()
        if not ranking.empty:
            ordered = ranking.sort_values(["scenario_rank", "scenario_label"])
            scenario_items = [
                {
                    "scenario_label": row.get("scenario_label"),
                    "scenario_probability": row.get("scenario_probability"),
                    "confidence_adjusted_probability": row.get("confidence_adjusted_probability"),
                    "scenario_rank": row.get("scenario_rank"),
                    "dominance_label": row.get("dominance_label"),
                }
                for row in ordered.to_dict(orient="records")
            ]
            dominant = str(summary.get("dominant_scenario", ""))
            alternatives = [
                item
                for item in scenario_items
                if str(item.get("scenario_label")) != dominant
            ][:3]
        else:
            scenario_items = []
            alternatives = []
        record.update(
            {
                "scenario_summary": _json_dumps(scenario_items),
                "dominant_scenario": summary.get("dominant_scenario", ""),
                "dominant_scenario_probability": summary.get("dominant_scenario_adjusted_probability", np.nan),
                "scenario_uncertainty_score": summary.get("uncertainty_score", np.nan),
                "scenario_dominance_score": summary.get("dominance_score", np.nan),
                "scenario_calibration_error": summary.get("calibration_error", np.nan),
                "scenario_confidence_bucket": summary.get("scenario_confidence_bucket", "unknown"),
                "alternative_scenarios": _json_dumps(alternatives),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows).sort_values(present_columns(enriched, SCENARIO_CONTEXT_COLUMNS)).reset_index(drop=True)


def _ordered_probability_columns(frame: pd.DataFrame) -> list[str]:
    return [*SCENARIO_REQUIRED_FIELDS, *[column for column in frame.columns if column not in SCENARIO_REQUIRED_FIELDS]]


def run_scenario_evaluation(
    *,
    forecasts_df: pd.DataFrame,
    consensus_df: pd.DataFrame | None = None,
    risk_df: pd.DataFrame | None = None,
    regime_df: pd.DataFrame | None = None,
    strategy_metrics_df: pd.DataFrame | None = None,
    analysis_packets_df: pd.DataFrame | None = None,
    model_health_df: pd.DataFrame | None = None,
    config: ScenarioEngineConfig | None = None,
) -> ScenarioEvaluationResult:
    """Run deterministic Scenario Evaluation Engine v1 in memory."""

    resolved_config = config or ScenarioEngineConfig()
    probability = build_scenario_probability_frame(
        forecasts_df,
        consensus_df=consensus_df,
        risk_df=risk_df,
        regime_df=regime_df,
        strategy_metrics_df=strategy_metrics_df,
        analysis_packets_df=analysis_packets_df,
        model_health_df=model_health_df,
        probability_method=resolved_config.probability_method,
    )
    probability, calibration_summary = apply_probability_calibration(
        probability,
        bins=resolved_config.calibration_bins,
        lookback=resolved_config.calibration_lookback,
    )
    probability, uncertainty_summary = attach_uncertainty_scores(probability)
    probability, rankings, dominance_summary = evaluate_scenario_dominance(probability)
    probability = probability[_ordered_probability_columns(probability)]
    enriched_packets = enrich_analysis_packets_with_scenarios(
        analysis_packets_df if analysis_packets_df is not None else pd.DataFrame(),
        rankings,
        dominance_summary,
    )
    source_counts = _source_counts(
        forecasts_df=forecasts_df,
        consensus_df=consensus_df,
        risk_df=risk_df,
        regime_df=regime_df,
        strategy_metrics_df=strategy_metrics_df,
        analysis_packets_df=analysis_packets_df,
        model_health_df=model_health_df,
    )
    manifest = build_scenario_manifest(
        config=resolved_config,
        artifact_paths={},
        scenario_probability=probability,
        scenario_rankings=rankings,
        scenario_dominance_summary=dominance_summary,
        scenario_uncertainty_summary=uncertainty_summary,
        scenario_calibration_summary=calibration_summary,
        source_counts=source_counts,
    )
    return ScenarioEvaluationResult(
        scenario_probability=probability,
        scenario_rankings=rankings,
        scenario_dominance_summary=dominance_summary,
        scenario_uncertainty_summary=uncertainty_summary,
        scenario_calibration_summary=calibration_summary,
        analysis_packets=enriched_packets,
        manifest=manifest,
    )


def write_scenario_outputs(output_dir: str | Path, result: ScenarioEvaluationResult) -> dict[str, str]:
    """Write all Scenario Evaluation Engine v1 artifacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    table_map = {
        "scenario_probability": result.scenario_probability,
        "scenario_rankings": result.scenario_rankings,
        "scenario_dominance_summary": result.scenario_dominance_summary,
        "scenario_uncertainty_summary": result.scenario_uncertainty_summary,
        "scenario_calibration_summary": result.scenario_calibration_summary,
    }
    paths: dict[str, str] = {}
    for name, frame in table_map.items():
        path = destination / SCENARIO_ARTIFACT_FILENAMES[name]
        frame.to_csv(path, index=False)
        paths[name] = str(path)

    manifest_path = destination / SCENARIO_ARTIFACT_FILENAMES["scenario_manifest"]
    manifest = dict(result.manifest)
    manifest["artifact_paths"] = {**paths, "scenario_manifest": str(manifest_path)}
    result.manifest = manifest
    write_scenario_manifest(destination, manifest)
    paths["scenario_manifest"] = str(manifest_path)
    return paths
