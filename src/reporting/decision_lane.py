"""Decision Lane v2 enrichment helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DECISION_LANE_VERSION = "v2"

DECISION_LANE_ENRICHED_FILENAME = "decision_lane_enriched_candidates.csv"
DECISION_LANE_MANIFEST_FILENAME = "decision_lane_manifest.json"

DECISION_LANE_CONTEXT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "ticker",
    "horizon",
    "target_type",
    "run_mode",
    "core_run_id",
)

ENRICHED_DECISION_LANE_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "source_packet_id",
    "timestamp",
    "ticker",
    "horizon",
    "target_type",
    "run_mode",
    "core_run_id",
    "primary_model_name",
    "primary_prediction",
    "candidate_score",
    "model_agreement_score",
    "disagreement_score",
    "agreement_bucket",
    "sign_conflict",
    "dominant_scenario",
    "dominant_scenario_probability",
    "scenario_confidence_bucket",
    "scenario_alignment",
    "risk_score",
    "risk_level",
    "risk_action",
    "risk_adjusted_confidence",
    "risk_adjusted_candidate_score",
    "candidate_status",
    "reason_codes",
    "reason_summary",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(numeric) or not np.isfinite(numeric):
        return default
    return float(numeric)


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value) if pd.notna(value) else False


def _present_columns(frame: pd.DataFrame, preferred: tuple[str, ...] | list[str]) -> list[str]:
    return [column for column in preferred if column in frame.columns]


def _normalize_key_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return pd.Timestamp(value).isoformat()
    return value


def _key(row: dict[str, Any], columns: list[str]) -> tuple[Any, ...]:
    return tuple(_normalize_key_value(row.get(column)) for column in columns)


def _lookup_by_packet_id(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.empty:
        return {}
    packet_column = "packet_id" if "packet_id" in frame.columns else "source_packet_id"
    if packet_column not in frame.columns:
        return {}
    return {
        str(row.get(packet_column)): row
        for row in frame.to_dict(orient="records")
        if str(row.get(packet_column, "")).strip()
    }


def _lookup_by_context(frame: pd.DataFrame) -> tuple[list[str], dict[tuple[Any, ...], dict[str, Any]]]:
    if frame.empty:
        return [], {}
    columns = _present_columns(frame, DECISION_LANE_CONTEXT_COLUMNS)
    if not columns:
        return [], {}
    lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        lookup[_key(row, columns)] = row
    return columns, lookup


def _groups_by_context(frame: pd.DataFrame) -> tuple[list[str], dict[tuple[Any, ...], pd.DataFrame]]:
    if frame.empty:
        return [], {}
    columns = _present_columns(frame, DECISION_LANE_CONTEXT_COLUMNS)
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


def _scenario_alignment(primary_prediction: Any, dominant_scenario: Any) -> str:
    prediction = _safe_float(primary_prediction, default=0.0)
    scenario = str(dominant_scenario or "").strip().lower()
    if not scenario:
        return "unknown"
    if prediction > 0.0 and scenario in {"bull", "recovery"}:
        return "aligned"
    if prediction > 0.0 and scenario in {"sideway", "uncertain"}:
        return "weakly_aligned"
    if prediction > 0.0 and scenario in {"bear", "drawdown", "high_volatility"}:
        return "misaligned_or_risky"
    return "unknown"


def _candidate_status(record: dict[str, Any]) -> str:
    if _safe_bool(record.get("force_hold")):
        return "force_hold"
    if _safe_bool(record.get("block_candidate")):
        return "blocked"
    risk_action = str(record.get("risk_action", "") or "").strip().lower()
    if risk_action == "reduce_candidate":
        return "reduced"
    if risk_action == "adjust_confidence":
        return "adjusted"
    return "diagnostic_candidate"


def _combine_reason_codes(record: dict[str, Any]) -> str:
    codes: list[str] = []
    alignment = str(record.get("scenario_alignment", "") or "").strip().lower()
    if alignment == "aligned":
        codes.append("scenario_aligned")
    elif alignment == "weakly_aligned":
        codes.append("scenario_weakly_aligned")
    elif alignment == "misaligned_or_risky":
        codes.append("scenario_misaligned_or_risky")

    agreement_bucket = str(record.get("agreement_bucket", "") or "").strip().lower()
    if agreement_bucket == "high":
        codes.append("high_model_agreement")
    elif agreement_bucket == "medium":
        codes.append("medium_model_agreement")
    if _safe_bool(record.get("sign_conflict")):
        codes.append("sign_conflict")

    risk_action = str(record.get("risk_action", "") or "").strip().lower()
    status = str(record.get("candidate_status", "") or "").strip().lower()
    confidence_factor = _safe_float(record.get("confidence_adjustment_factor"), default=1.0)
    if risk_action == "adjust_confidence" or confidence_factor < 1.0:
        codes.append("risk_adjusted")
    if risk_action == "reduce_candidate" or status == "reduced":
        codes.append("risk_reduced")
    if _safe_bool(record.get("block_candidate")) or status == "blocked":
        codes.append("risk_blocked")
    if _safe_bool(record.get("force_hold")) or status == "force_hold":
        codes.append("force_hold")

    volatility_bucket = str(record.get("volatility_bucket", "") or "").strip().lower()
    if volatility_bucket == "high":
        codes.append("high_volatility")

    risk_reason_codes = str(record.get("risk_reason_codes", "") or "")
    risk_reason_set = {code.strip() for code in risk_reason_codes.split("|") if code.strip()}
    if "volatility_spike" in risk_reason_set:
        codes.append("high_volatility")
    for code in (
        "elevated_drawdown",
        "severe_drawdown",
        "weak_model_health",
        "failing_model_health",
    ):
        if code in risk_reason_set:
            codes.append(code)

    return "|".join(dict.fromkeys(codes)) or "none"


def _reason_summary(record: dict[str, Any]) -> str:
    scenario = str(record.get("dominant_scenario", "") or "").strip()
    alignment = str(record.get("scenario_alignment", "") or "").strip()
    agreement = str(record.get("agreement_bucket", "unknown") or "unknown").strip().lower()
    status = str(record.get("candidate_status", "diagnostic_candidate") or "diagnostic_candidate")
    parts: list[str] = []

    if alignment == "aligned" and scenario:
        parts.append(f"aligned with {scenario} scenario")
    elif alignment == "weakly_aligned" and scenario:
        parts.append(f"weakly aligned with {scenario} scenario")
    elif alignment == "misaligned_or_risky" and scenario:
        parts.append(f"misaligned or risk-sensitive under {scenario} scenario")
    else:
        parts.append("missing scenario context")

    parts.append(f"has {agreement} model agreement")
    if _safe_bool(record.get("sign_conflict")):
        parts.append("has model sign conflict")

    if status == "force_hold":
        parts.append("received a force-hold risk override")
    elif status == "blocked":
        parts.append("was blocked by risk governance")
    elif status == "reduced":
        parts.append("was reduced by risk governance")
    elif status == "adjusted":
        parts.append("was confidence-adjusted due to elevated risk")
    else:
        parts.append("remains a diagnostic candidate")

    return "Candidate is " + ", ".join(parts[:-1]) + f", and {parts[-1]}."


def _source_counts(
    *,
    candidates_df: pd.DataFrame,
    packets_df: pd.DataFrame,
    risk_adjusted_candidates_df: pd.DataFrame,
    scenario_dominance_df: pd.DataFrame,
    scenario_probability_df: pd.DataFrame,
) -> dict[str, int]:
    return {
        "decision_lane_candidates": int(len(candidates_df)),
        "analysis_packets": int(len(packets_df)),
        "risk_adjusted_candidates": int(len(risk_adjusted_candidates_df)),
        "scenario_dominance_summary": int(len(scenario_dominance_df)),
        "scenario_probability": int(len(scenario_probability_df)),
    }


def build_decision_lane_candidates(packets_df: pd.DataFrame) -> pd.DataFrame:
    """Build the legacy conservative candidate view for analyst review."""

    if packets_df.empty:
        return pd.DataFrame(columns=["packet_id", "ticker", "timestamp", "primary_prediction", "agreement_bucket"])
    candidates = packets_df.copy()
    tradable = candidates.get("target_tradable", pd.Series(False, index=candidates.index)).fillna(False).astype(bool)
    agreement = candidates.get("agreement_bucket", pd.Series("unknown", index=candidates.index)).astype(str)
    prediction = pd.to_numeric(candidates.get("primary_prediction"), errors="coerce")
    active_signal_count = pd.to_numeric(candidates.get("active_signal_count"), errors="coerce").fillna(0.0)
    candidates = candidates[
        tradable
        & prediction.gt(0.0)
        & agreement.isin(["medium", "high"])
        & active_signal_count.gt(0.0)
    ].copy()
    candidates["candidate_score"] = (
        prediction.fillna(0.0)
        * pd.to_numeric(candidates.get("model_agreement_score"), errors="coerce").fillna(0.0)
    )
    ordered_columns = [
        "packet_id",
        "timestamp",
        "ticker",
        "group_name",
        "horizon",
        "target_type",
        "run_mode",
        "primary_model_name",
        "primary_prediction",
        "model_agreement_score",
        "agreement_bucket",
        "regime_label",
        "volatility_bucket",
        "active_signal_count",
        "top_policy_model",
        "top_policy_sharpe",
        "candidate_score",
    ]
    return candidates.sort_values(["candidate_score", "top_policy_sharpe"], ascending=[False, False]).reset_index(
        drop=True
    )[ordered_columns]


def build_enriched_decision_lane_candidates(
    candidates_df: pd.DataFrame,
    packets_df: pd.DataFrame,
    risk_adjusted_candidates_df: pd.DataFrame | None = None,
    scenario_dominance_df: pd.DataFrame | None = None,
    scenario_probability_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build Decision Lane v2 diagnostic candidate surface."""

    candidates = candidates_df.copy() if candidates_df is not None else pd.DataFrame()
    packets = packets_df.copy() if packets_df is not None else pd.DataFrame()
    risk_adjusted = (
        risk_adjusted_candidates_df.copy() if risk_adjusted_candidates_df is not None else pd.DataFrame()
    )
    dominance = scenario_dominance_df.copy() if scenario_dominance_df is not None else pd.DataFrame()
    probability = scenario_probability_df.copy() if scenario_probability_df is not None else pd.DataFrame()

    if candidates.empty:
        return pd.DataFrame(columns=list(ENRICHED_DECISION_LANE_COLUMNS))

    packet_lookup = _lookup_by_packet_id(packets)
    risk_lookup = _lookup_by_packet_id(risk_adjusted)
    dominance_columns, dominance_lookup = _lookup_by_context(dominance)
    probability_columns, probability_groups = _groups_by_context(probability)

    rows: list[dict[str, Any]] = []
    for candidate in candidates.to_dict(orient="records"):
        source_packet_id = str(candidate.get("packet_id", "") or candidate.get("source_packet_id", "") or "")
        packet = packet_lookup.get(source_packet_id, {})
        risk = risk_lookup.get(source_packet_id, {})
        context = {**packet, **candidate, **risk}
        if "source_packet_id" not in context:
            context["source_packet_id"] = source_packet_id

        dominance_row = dominance_lookup.get(_key(context, dominance_columns), {}) if dominance_columns else {}
        probability_group = (
            probability_groups.get(_key(context, probability_columns), pd.DataFrame())
            if probability_columns
            else pd.DataFrame()
        )
        probability_row = _dominant_probability_record(probability_group)

        dominant_scenario = (
            context.get("dominant_scenario")
            or dominance_row.get("dominant_scenario")
            or probability_row.get("scenario_label")
            or ""
        )
        dominant_probability = (
            context.get("dominant_scenario_probability")
            or dominance_row.get("dominant_scenario_probability")
            or dominance_row.get("dominant_scenario_adjusted_probability")
            or probability_row.get("confidence_adjusted_probability")
            or probability_row.get("scenario_probability")
        )
        scenario_confidence_bucket = (
            context.get("scenario_confidence_bucket")
            or dominance_row.get("scenario_confidence_bucket")
            or context.get("confidence_bucket")
            or ""
        )
        primary_prediction = context.get("primary_prediction")
        alignment = _scenario_alignment(primary_prediction, dominant_scenario)
        agreement_score = _safe_float(context.get("model_agreement_score"), default=0.0)
        disagreement_score = _safe_float(context.get("model_disagreement_score"))
        if pd.isna(disagreement_score):
            disagreement_score = _safe_float(context.get("disagreement_score"))
        if pd.isna(disagreement_score):
            disagreement_score = 1.0 - agreement_score
        confidence_factor = _safe_float(context.get("confidence_adjustment_factor"), default=1.0)
        risk_adjusted_confidence = agreement_score * confidence_factor
        risk_adjusted_candidate_score = _safe_float(
            context.get("risk_adjusted_candidate_score"),
            default=_safe_float(context.get("candidate_score"), default=0.0) * confidence_factor,
        )

        row = {
            "candidate_id": f"decision_lane_v2|{source_packet_id}",
            "source_packet_id": source_packet_id,
            "timestamp": context.get("timestamp"),
            "ticker": context.get("ticker"),
            "horizon": context.get("horizon"),
            "target_type": context.get("target_type"),
            "run_mode": context.get("run_mode"),
            "core_run_id": context.get("core_run_id"),
            "primary_model_name": context.get("primary_model_name"),
            "primary_prediction": _safe_float(primary_prediction),
            "candidate_score": _safe_float(context.get("candidate_score"), default=0.0),
            "model_agreement_score": agreement_score,
            "disagreement_score": disagreement_score,
            "agreement_bucket": context.get("agreement_bucket", "unknown"),
            "sign_conflict": _safe_bool(context.get("sign_conflict")),
            "dominant_scenario": dominant_scenario,
            "dominant_scenario_probability": _safe_float(dominant_probability),
            "scenario_confidence_bucket": scenario_confidence_bucket,
            "scenario_alignment": alignment,
            "risk_score": _safe_float(context.get("risk_score")),
            "risk_level": context.get("risk_level", ""),
            "risk_action": context.get("risk_action", "pass"),
            "risk_adjusted_confidence": round(float(risk_adjusted_confidence), 6),
            "risk_adjusted_candidate_score": round(float(risk_adjusted_candidate_score), 6),
            "confidence_adjustment_factor": confidence_factor,
            "block_candidate": _safe_bool(context.get("block_candidate")),
            "force_hold": _safe_bool(context.get("force_hold")),
            "risk_reason_codes": context.get("risk_reason_codes", ""),
            "volatility_bucket": context.get("volatility_bucket", ""),
        }
        row["candidate_status"] = _candidate_status(row)
        row["reason_codes"] = _combine_reason_codes(row)
        row["reason_summary"] = _reason_summary(row)
        rows.append(row)

    enriched = pd.DataFrame(rows)
    if enriched.empty:
        return pd.DataFrame(columns=list(ENRICHED_DECISION_LANE_COLUMNS))
    enriched = enriched.sort_values(
        ["risk_adjusted_candidate_score", "candidate_score", "candidate_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return enriched[list(ENRICHED_DECISION_LANE_COLUMNS)]


def build_decision_lane_manifest(
    *,
    candidates_df: pd.DataFrame,
    packets_df: pd.DataFrame,
    enriched_candidates_df: pd.DataFrame,
    risk_adjusted_candidates_df: pd.DataFrame | None = None,
    scenario_dominance_df: pd.DataFrame | None = None,
    scenario_probability_df: pd.DataFrame | None = None,
    artifact_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build Decision Lane v2 manifest metadata."""

    risk_adjusted = risk_adjusted_candidates_df if risk_adjusted_candidates_df is not None else pd.DataFrame()
    dominance = scenario_dominance_df if scenario_dominance_df is not None else pd.DataFrame()
    probability = scenario_probability_df if scenario_probability_df is not None else pd.DataFrame()
    return {
        "manifest_type": "decision_lane_v2_manifest",
        "version": DECISION_LANE_VERSION,
        "artifact_paths": dict(artifact_paths or {}),
        "artifact_filenames": {
            "decision_lane_enriched_candidates": DECISION_LANE_ENRICHED_FILENAME,
            "decision_lane_manifest": DECISION_LANE_MANIFEST_FILENAME,
        },
        "required_fields": list(ENRICHED_DECISION_LANE_COLUMNS),
        "input_row_counts": _source_counts(
            candidates_df=candidates_df,
            packets_df=packets_df,
            risk_adjusted_candidates_df=risk_adjusted,
            scenario_dominance_df=dominance,
            scenario_probability_df=probability,
        ),
        "output_row_counts": {
            "decision_lane_enriched_candidates": int(len(enriched_candidates_df)),
        },
        "scenario_alignment_rules": {
            "bull_recovery_positive": "aligned",
            "sideway_uncertain_positive": "weakly_aligned",
            "bear_drawdown_high_volatility_positive": "misaligned_or_risky",
            "missing_scenario": "unknown",
        },
        "diagnostic_only_authority": True,
        "no_buy_sell_recommendation_authority": True,
    }


def write_decision_lane_outputs(
    output_dir: str | Path,
    *,
    enriched_candidates_df: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, str]:
    """Write Decision Lane v2 enriched candidates and manifest artifacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    enriched_path = destination / DECISION_LANE_ENRICHED_FILENAME
    manifest_path = destination / DECISION_LANE_MANIFEST_FILENAME
    enriched_candidates_df.to_csv(enriched_path, index=False)
    paths = {
        "decision_lane_enriched_candidates": str(enriched_path),
        "decision_lane_manifest": str(manifest_path),
    }
    updated_manifest = {**manifest, "artifact_paths": paths}
    manifest_path.write_text(
        json.dumps(updated_manifest, default=_json_default, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return paths
