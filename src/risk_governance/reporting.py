"""High-level Risk Governance Layer v1 orchestration and artifact writing."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.risk_governance.actions import apply_risk_action_fields
from src.risk_governance.manifest import build_risk_manifest, write_risk_manifest
from src.risk_governance.schema import (
    JOIN_CONTEXT_COLUMNS,
    RISK_GOVERNANCE_ARTIFACT_FILENAMES,
    RISK_GOVERNANCE_SUMMARY_COLUMNS,
    RiskGovernanceConfig,
    RiskGovernanceResult,
    empty_summary_frame,
    present_columns,
)
from src.risk_governance.scoring import (
    build_reason_codes,
    build_risk_components,
    calculate_weighted_risk_score,
    normalized_model_health_component,
    safe_float,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def _normalize_key_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return pd.Timestamp(value).isoformat()
    return value


def _key(row: dict[str, Any], columns: list[str]) -> tuple[Any, ...]:
    return tuple(_normalize_key_value(row.get(column)) for column in columns)


def _lookup_by_keys(
    frame: pd.DataFrame,
    preferred_columns: tuple[str, ...] | list[str] = JOIN_CONTEXT_COLUMNS,
) -> tuple[list[str], dict[tuple[Any, ...], dict[str, Any]]]:
    if frame.empty:
        return [], {}
    columns = present_columns(frame, preferred_columns)
    if not columns:
        return [], {}
    lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        lookup[_key(row, columns)] = row
    return columns, lookup


def _groups_by_keys(
    frame: pd.DataFrame,
    preferred_columns: tuple[str, ...] | list[str] = JOIN_CONTEXT_COLUMNS,
) -> tuple[list[str], dict[tuple[Any, ...], pd.DataFrame]]:
    if frame.empty:
        return [], {}
    columns = present_columns(frame, preferred_columns)
    if not columns:
        return [], {}
    prepared = frame.copy()
    for column in columns:
        if column == "timestamp":
            prepared[column] = pd.to_datetime(prepared[column], errors="coerce").map(
                lambda value: value.isoformat() if pd.notna(value) else value
            )
    groups: dict[tuple[Any, ...], pd.DataFrame] = {}
    for keys, group in prepared.groupby(columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        groups[tuple(_normalize_key_value(value) for value in keys)] = group.copy()
    return columns, groups


def _candidate_lookup(candidates_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if candidates_df.empty or "packet_id" not in candidates_df.columns:
        return {}
    return {
        str(row.get("packet_id")): row
        for row in candidates_df.to_dict(orient="records")
        if str(row.get("packet_id", "")).strip()
    }


def _parse_core_run_id(record: dict[str, Any]) -> str:
    value = record.get("core_run_id")
    if value is not None and str(value).strip() and str(value) != "nan":
        return str(value)
    packet_id = str(record.get("packet_id", "") or "")
    parts = packet_id.split("|")
    return parts[-1] if len(parts) >= 6 else ""


def _merge_embedded_packet_sources(packet: dict[str, Any]) -> dict[str, Any]:
    source: dict[str, Any] = {}
    risk_summary = _json_loads(packet.get("risk_summary"))
    if isinstance(risk_summary, dict):
        source.update(risk_summary)
    for column in (
        "vol_forecast",
        "volatility_bucket",
        "model_agreement_score",
        "model_disagreement_score",
        "agreement_bucket",
        "dispersion_score",
        "sign_conflict",
        "scenario_uncertainty_score",
        "scenario_dominance_score",
        "scenario_calibration_error",
        "scenario_confidence_bucket",
        "dominant_scenario",
    ):
        if column in packet and pd.notna(packet.get(column)):
            source[column] = packet.get(column)
    if "scenario_dominance_score" in source and "dominance_score" not in source:
        source["dominance_score"] = source["scenario_dominance_score"]
    if "model_disagreement_score" in source and "disagreement_score" not in source:
        source["disagreement_score"] = source["model_disagreement_score"]
    return source


def _extract_packet_model_names(packet: dict[str, Any]) -> list[str]:
    names: list[str] = []
    primary = str(packet.get("primary_model_name", "") or "").strip()
    if primary:
        names.append(primary)
    records = _json_loads(packet.get("model_by_model_predictions"))
    if isinstance(records, list):
        for record in records:
            if isinstance(record, dict):
                name = str(record.get("model_name", "") or "").strip()
                if name:
                    names.append(name)
    return list(dict.fromkeys(names))


def _model_health_lookup(model_health_df: pd.DataFrame) -> dict[str, str]:
    if model_health_df.empty or "model_name" not in model_health_df.columns:
        return {}
    lookup: dict[str, str] = {}
    for row in model_health_df.to_dict(orient="records"):
        model_name = str(row.get("model_name", "") or "").strip()
        if not model_name:
            continue
        status = str(row.get("health_status", "") or "").strip().lower()
        if not status:
            continue
        lookup[model_name] = status
    return lookup


def _resolve_model_health_status(
    packet: dict[str, Any],
    candidate: dict[str, Any],
    health_by_model: dict[str, str],
    config: RiskGovernanceConfig,
) -> str:
    model_names = [
        str(candidate.get("primary_model_name", "") or "").strip(),
        *_extract_packet_model_names(packet),
    ]
    statuses = [
        health_by_model[name]
        for name in dict.fromkeys(model_names)
        if name and name in health_by_model
    ]
    if not statuses:
        return "healthy"
    return max(statuses, key=lambda status: normalized_model_health_component(status, config))


def _max_scenario_downside_risk(probability_group: pd.DataFrame) -> float:
    if probability_group.empty:
        return float("nan")
    downside = pd.to_numeric(probability_group.get("downside_risk"), errors="coerce").dropna()
    if downside.empty:
        return float("nan")
    return float(downside.abs().max())


def _dominant_probability_record(probability_group: pd.DataFrame) -> dict[str, Any]:
    if probability_group.empty:
        return {}
    ranked = probability_group.copy()
    sort_column = (
        "confidence_adjusted_probability"
        if "confidence_adjusted_probability" in ranked.columns
        else "scenario_probability"
    )
    ranked["_sort_probability"] = pd.to_numeric(ranked.get(sort_column), errors="coerce").fillna(0.0)
    return ranked.sort_values(["_sort_probability", "scenario_label"], ascending=[False, True]).iloc[0].to_dict()


def _build_input_row_counts(
    *,
    candidates_df: pd.DataFrame,
    packets_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    consensus_df: pd.DataFrame,
    model_health_df: pd.DataFrame,
    scenario_dominance_df: pd.DataFrame,
    scenario_uncertainty_df: pd.DataFrame,
    scenario_probability_df: pd.DataFrame,
) -> dict[str, int]:
    return {
        "decision_lane_candidates": int(len(candidates_df)),
        "analysis_packets": int(len(packets_df)),
        "risk_summary": int(len(risk_df)),
        "model_consensus_summary": int(len(consensus_df)),
        "model_health_summary": int(len(model_health_df)),
        "scenario_dominance_summary": int(len(scenario_dominance_df)),
        "scenario_uncertainty_summary": int(len(scenario_uncertainty_df)),
        "scenario_probability": int(len(scenario_probability_df)),
    }


def _empty_result(config: RiskGovernanceConfig, input_row_counts: dict[str, int]) -> RiskGovernanceResult:
    summary = empty_summary_frame()
    adjusted = pd.DataFrame(columns=["risk_adjusted_candidate_score"])
    override = summary.copy()
    manifest = build_risk_manifest(
        config=config,
        artifact_paths={},
        input_row_counts=input_row_counts,
        risk_governance_summary=summary,
        risk_adjusted_candidates=adjusted,
        risk_override_log=override,
    )
    return RiskGovernanceResult(
        risk_governance_summary=summary,
        risk_adjusted_candidates=adjusted,
        risk_override_log=override,
        manifest=manifest,
    )


def run_risk_governance(
    *,
    candidates_df: pd.DataFrame,
    packets_df: pd.DataFrame,
    risk_df: pd.DataFrame | None = None,
    consensus_df: pd.DataFrame | None = None,
    model_health_df: pd.DataFrame | None = None,
    scenario_dominance_df: pd.DataFrame | None = None,
    scenario_uncertainty_df: pd.DataFrame | None = None,
    scenario_probability_df: pd.DataFrame | None = None,
    config: RiskGovernanceConfig | None = None,
) -> RiskGovernanceResult:
    """Run deterministic Risk Governance Layer v1 in memory."""

    resolved = config or RiskGovernanceConfig()
    candidates = candidates_df.copy() if candidates_df is not None else pd.DataFrame()
    packets = packets_df.copy() if packets_df is not None else pd.DataFrame()
    risk = risk_df.copy() if risk_df is not None else pd.DataFrame()
    consensus = consensus_df.copy() if consensus_df is not None else pd.DataFrame()
    model_health = model_health_df.copy() if model_health_df is not None else pd.DataFrame()
    dominance = scenario_dominance_df.copy() if scenario_dominance_df is not None else pd.DataFrame()
    uncertainty = scenario_uncertainty_df.copy() if scenario_uncertainty_df is not None else pd.DataFrame()
    probability = scenario_probability_df.copy() if scenario_probability_df is not None else pd.DataFrame()

    input_row_counts = _build_input_row_counts(
        candidates_df=candidates,
        packets_df=packets,
        risk_df=risk,
        consensus_df=consensus,
        model_health_df=model_health,
        scenario_dominance_df=dominance,
        scenario_uncertainty_df=uncertainty,
        scenario_probability_df=probability,
    )
    if candidates.empty:
        return _empty_result(resolved, input_row_counts)

    packet_by_id = _candidate_lookup(packets)
    risk_columns, risk_lookup = _lookup_by_keys(risk)
    consensus_columns, consensus_lookup = _lookup_by_keys(consensus)
    dominance_columns, dominance_lookup = _lookup_by_keys(dominance)
    uncertainty_columns, uncertainty_lookup = _lookup_by_keys(uncertainty)
    probability_columns, probability_groups = _groups_by_keys(probability)
    health_by_model = _model_health_lookup(model_health)

    summary_rows: list[dict[str, Any]] = []
    adjusted_rows: list[dict[str, Any]] = []
    for candidate in candidates.to_dict(orient="records"):
        packet_id = str(candidate.get("packet_id", "") or "")
        packet = packet_by_id.get(packet_id, {})
        context = {**packet, **candidate}
        context["core_run_id"] = _parse_core_run_id(context)

        source = _merge_embedded_packet_sources(packet)
        source.update({key: value for key, value in candidate.items() if pd.notna(value)})

        for columns, lookup in (
            (risk_columns, risk_lookup),
            (consensus_columns, consensus_lookup),
            (dominance_columns, dominance_lookup),
            (uncertainty_columns, uncertainty_lookup),
        ):
            if columns:
                source.update(lookup.get(_key(context, columns), {}))

        probability_group = (
            probability_groups.get(_key(context, probability_columns), pd.DataFrame())
            if probability_columns
            else pd.DataFrame()
        )
        dominant_probability = _dominant_probability_record(probability_group)
        if dominant_probability:
            for column in (
                "scenario_label",
                "scenario_probability",
                "confidence_adjusted_probability",
                "calibration_error",
            ):
                if column in dominant_probability:
                    source[f"dominant_{column}"] = dominant_probability[column]
            if "scenario_label" in dominant_probability and "dominant_scenario" not in source:
                source["dominant_scenario"] = dominant_probability["scenario_label"]
        scenario_downside_risk = _max_scenario_downside_risk(probability_group)
        if pd.notna(scenario_downside_risk):
            source["scenario_downside_risk"] = scenario_downside_risk

        if "agreement_score" in source and "model_agreement_score" not in source:
            source["model_agreement_score"] = source["agreement_score"]
        if "disagreement_score" in source and "model_disagreement_score" not in source:
            source["model_disagreement_score"] = source["disagreement_score"]
        if "confidence_bucket" in source and "scenario_confidence_bucket" not in source:
            source["scenario_confidence_bucket"] = source["confidence_bucket"]

        model_health_status = _resolve_model_health_status(packet, candidate, health_by_model, resolved)
        components = build_risk_components(source, model_health_status=model_health_status, config=resolved)
        risk_score = calculate_weighted_risk_score(components, resolved)
        action_fields = apply_risk_action_fields(candidate, risk_score=risk_score, config=resolved)
        reason_codes = build_reason_codes(source, components, model_health_status=model_health_status)

        summary_row = {
            "packet_id": packet_id,
            "timestamp": context.get("timestamp"),
            "ticker": context.get("ticker"),
            "horizon": context.get("horizon"),
            "target_type": context.get("target_type"),
            "run_mode": context.get("run_mode"),
            "core_run_id": context.get("core_run_id"),
            "risk_score": risk_score,
            "risk_level": action_fields["risk_level"],
            "risk_action": action_fields["risk_action"],
            "confidence_adjustment_factor": action_fields["confidence_adjustment_factor"],
            "block_candidate": bool(action_fields["block_candidate"]),
            "force_hold": bool(action_fields["force_hold"]),
            "risk_reason_codes": reason_codes,
            **components,
        }
        summary_rows.append(summary_row)

        adjusted_row = {
            **candidate,
            **summary_row,
            "risk_adjusted_candidate_score": action_fields["risk_adjusted_candidate_score"],
            "candidate_eligible_after_risk": not bool(action_fields["block_candidate"]),
            "model_health_status": model_health_status,
        }
        adjusted_rows.append(adjusted_row)

    sort_columns = [
        column
        for column in ("timestamp", "ticker", "horizon", "target_type", "run_mode", "core_run_id", "packet_id")
        if column in candidates.columns or column in RISK_GOVERNANCE_SUMMARY_COLUMNS
    ]
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary[list(RISK_GOVERNANCE_SUMMARY_COLUMNS)].sort_values(sort_columns).reset_index(drop=True)
    else:
        summary = empty_summary_frame()

    adjusted = pd.DataFrame(adjusted_rows)
    if not adjusted.empty:
        adjusted = adjusted.sort_values(
            ["risk_adjusted_candidate_score", "candidate_score", "packet_id"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
    override = adjusted[
        adjusted.get("risk_action", pd.Series(dtype="object")).isin(
            ["reduce_candidate", "block_candidate", "force_hold"]
        )
        | adjusted.get("block_candidate", pd.Series(dtype=bool)).astype(bool)
        | adjusted.get("force_hold", pd.Series(dtype=bool)).astype(bool)
    ].copy() if not adjusted.empty else pd.DataFrame(columns=list(RISK_GOVERNANCE_SUMMARY_COLUMNS))

    manifest = build_risk_manifest(
        config=resolved,
        artifact_paths={},
        input_row_counts=input_row_counts,
        risk_governance_summary=summary,
        risk_adjusted_candidates=adjusted,
        risk_override_log=override,
    )
    return RiskGovernanceResult(
        risk_governance_summary=summary,
        risk_adjusted_candidates=adjusted,
        risk_override_log=override,
        manifest=manifest,
    )


def write_risk_governance_outputs(output_dir: str | Path, result: RiskGovernanceResult) -> dict[str, str]:
    """Write all Risk Governance Layer v1 artifacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    table_map = {
        "risk_governance_summary": result.risk_governance_summary,
        "risk_adjusted_candidates": result.risk_adjusted_candidates,
        "risk_override_log": result.risk_override_log,
    }
    paths: dict[str, str] = {}
    for name, frame in table_map.items():
        path = destination / RISK_GOVERNANCE_ARTIFACT_FILENAMES[name]
        frame.to_csv(path, index=False)
        paths[name] = str(path)

    manifest_path = destination / RISK_GOVERNANCE_ARTIFACT_FILENAMES["risk_manifest"]
    manifest = dict(result.manifest)
    manifest["artifact_paths"] = {**paths, "risk_manifest": str(manifest_path)}
    result.manifest = manifest
    write_risk_manifest(destination, manifest)
    paths["risk_manifest"] = str(manifest_path)
    return paths
