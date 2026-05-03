"""Deterministic allocation orchestration for Portfolio Allocator v1."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.portfolio_allocator.gating import combine_reason_codes, evaluate_allocation_gate
from src.portfolio_allocator.manifest import build_allocator_manifest
from src.portfolio_allocator.schema import (
    ALLOCATION_OUTPUT_COLUMNS,
    PORTFOLIO_RISK_SUMMARY_COLUMNS,
    PORTFOLIO_SUMMARY_COLUMNS,
    PortfolioAllocatorConfig,
    PortfolioAllocatorResult,
    bounded,
    empty_allocation_frame,
    normalize_text,
    safe_float,
)
from src.portfolio_allocator.sizing import attach_allocation_priority_scores, calculate_raw_weight


CONTEXT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "ticker",
    "horizon",
    "target_type",
    "run_mode",
    "core_run_id",
)


def _frame(value: pd.DataFrame | None) -> pd.DataFrame:
    return value.copy() if value is not None else pd.DataFrame()


def _present_columns(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _series(frame: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def _standardize_packet_id(frame: pd.DataFrame) -> pd.DataFrame:
    standardized = frame.copy()
    if "source_packet_id" not in standardized.columns and "packet_id" in standardized.columns:
        standardized["source_packet_id"] = standardized["packet_id"]
    return standardized


def _fill_missing_column(left: pd.DataFrame, column: str, right_column: str) -> pd.DataFrame:
    optional = f"{right_column}__optional"
    if optional not in left.columns:
        return left
    if column not in left.columns:
        left[column] = left[optional]
    else:
        left[column] = left[column].where(left[column].notna() & (left[column].astype(str) != ""), left[optional])
    return left.drop(columns=[optional])


def _merge_risk_adjusted(candidates: pd.DataFrame, risk_adjusted: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty or risk_adjusted.empty:
        return candidates
    left = _standardize_packet_id(candidates)
    right = _standardize_packet_id(risk_adjusted)
    if "source_packet_id" not in left.columns or "source_packet_id" not in right.columns:
        return left
    keep = [
        "source_packet_id",
        *[
            column
            for column in (
                "risk_score",
                "risk_level",
                "risk_action",
                "risk_adjusted_candidate_score",
                "candidate_status",
            )
            if column in right.columns
        ],
    ]
    right = right[keep].drop_duplicates("source_packet_id", keep="first")
    renamed = right.rename(columns={column: f"{column}__optional" for column in keep if column != "source_packet_id"})
    merged = left.merge(renamed, on="source_packet_id", how="left")
    for column in keep:
        if column != "source_packet_id":
            merged = _fill_missing_column(merged, column, column)
    return merged


def _merge_scenario_dominance(candidates: pd.DataFrame, dominance: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty or dominance.empty:
        return candidates
    keys = _present_columns(candidates, CONTEXT_COLUMNS)
    keys = [column for column in keys if column in dominance.columns]
    if not keys:
        return candidates
    keep = [
        *keys,
        *[
            column
            for column in (
                "dominance_score",
                "dominant_scenario",
                "dominant_scenario_probability",
                "dominant_scenario_adjusted_probability",
                "scenario_confidence_bucket",
            )
            if column in dominance.columns
        ],
    ]
    right = dominance[keep].drop_duplicates(keys, keep="first")
    renamed = right.rename(columns={column: f"{column}__optional" for column in keep if column not in keys})
    merged = candidates.merge(renamed, on=keys, how="left")
    for column in ("dominance_score", "dominant_scenario", "scenario_confidence_bucket"):
        merged = _fill_missing_column(merged, column, column)

    adjusted = "dominant_scenario_adjusted_probability__optional"
    probability = "dominant_scenario_probability__optional"
    if "dominant_scenario_probability" not in merged.columns:
        merged["dominant_scenario_probability"] = pd.NA
    if adjusted in merged.columns:
        merged["dominant_scenario_probability"] = merged["dominant_scenario_probability"].where(
            merged["dominant_scenario_probability"].notna()
            & (merged["dominant_scenario_probability"].astype(str) != ""),
            merged[adjusted],
        )
        merged = merged.drop(columns=[adjusted])
    if probability in merged.columns:
        merged["dominant_scenario_probability"] = merged["dominant_scenario_probability"].where(
            merged["dominant_scenario_probability"].notna()
            & (merged["dominant_scenario_probability"].astype(str) != ""),
            merged[probability],
        )
        merged = merged.drop(columns=[probability])
    return merged


def _prepare_candidates(
    decision_lane_enriched_candidates_df: pd.DataFrame,
    *,
    risk_adjusted_candidates_df: pd.DataFrame | None,
    scenario_dominance_df: pd.DataFrame | None,
) -> pd.DataFrame:
    candidates = _standardize_packet_id(decision_lane_enriched_candidates_df)
    candidates = _merge_risk_adjusted(candidates, _frame(risk_adjusted_candidates_df))
    candidates = _merge_scenario_dominance(candidates, _frame(scenario_dominance_df))
    if candidates.empty:
        return candidates

    prepared = candidates.copy()
    if "candidate_id" not in prepared.columns:
        packet_ids = _series(prepared, "source_packet_id", "").astype(str)
        prepared["candidate_id"] = [
            f"decision_lane_v2|{packet_id}" if packet_id and packet_id != "nan" else f"candidate_{idx:04d}"
            for idx, packet_id in enumerate(packet_ids, start=1)
        ]
    if "source_packet_id" not in prepared.columns:
        prepared["source_packet_id"] = ""

    if "dominance_score" not in prepared.columns and "scenario_dominance_score" in prepared.columns:
        prepared["dominance_score"] = prepared["scenario_dominance_score"]
    if "disagreement_score" not in prepared.columns and "model_disagreement_score" in prepared.columns:
        prepared["disagreement_score"] = prepared["model_disagreement_score"]
    if (
        "dominant_scenario_probability" not in prepared.columns
        and "dominant_scenario_adjusted_probability" in prepared.columns
    ):
        prepared["dominant_scenario_probability"] = prepared["dominant_scenario_adjusted_probability"]

    defaults: dict[str, Any] = {
        "timestamp": "",
        "ticker": "",
        "horizon": "",
        "target_type": "",
        "run_mode": "",
        "risk_level": "level_1_soft_adjustment",
        "risk_action": "pass",
        "candidate_status": "diagnostic_candidate",
        "scenario_alignment": "unknown",
        "scenario_confidence_bucket": "unknown",
        "dominant_scenario": "",
        "reason_codes": "",
    }
    for column, default in defaults.items():
        if column not in prepared.columns:
            prepared[column] = default

    for column, default in (
        ("risk_adjusted_confidence", 0.0),
        ("risk_adjusted_candidate_score", 0.0),
        ("risk_score", 1.0),
        ("disagreement_score", 1.0),
        ("dominance_score", 0.0),
        ("dominant_scenario_probability", 0.0),
    ):
        prepared[column] = pd.to_numeric(_series(prepared, column, default), errors="coerce").fillna(default)

    return attach_allocation_priority_scores(prepared)


def _allocation_id(row: pd.Series, index: int) -> str:
    candidate_id = normalize_text(row.get("candidate_id"), default=f"candidate_{index:04d}")
    return f"portfolio_allocator_v1|{candidate_id}"


def _base_output_row(row: pd.Series, allocation_id: str) -> dict[str, Any]:
    return {
        "allocation_id": allocation_id,
        "candidate_id": normalize_text(row.get("candidate_id")),
        "source_packet_id": normalize_text(row.get("source_packet_id")),
        "timestamp": row.get("timestamp", ""),
        "ticker": normalize_text(row.get("ticker")),
        "horizon": row.get("horizon", ""),
        "target_type": normalize_text(row.get("target_type")),
        "run_mode": normalize_text(row.get("run_mode")),
        "risk_adjusted_confidence": round(bounded(row.get("risk_adjusted_confidence"), default=0.0), 10),
        "risk_adjusted_candidate_score": round(safe_float(row.get("risk_adjusted_candidate_score"), default=0.0), 10),
        "risk_score": round(bounded(row.get("risk_score"), default=1.0), 10),
        "risk_level": normalize_text(row.get("risk_level"), default="level_1_soft_adjustment"),
        "risk_action": normalize_text(row.get("risk_action"), default="pass"),
        "disagreement_score": round(bounded(row.get("disagreement_score"), default=1.0), 10),
        "dominance_score": round(bounded(row.get("dominance_score"), default=0.0), 10),
        "dominant_scenario": normalize_text(row.get("dominant_scenario")),
        "dominant_scenario_probability": round(bounded(row.get("dominant_scenario_probability"), default=0.0), 10),
        "scenario_alignment": normalize_text(row.get("scenario_alignment"), default="unknown"),
        "allocation_priority_score": round(safe_float(row.get("allocation_priority_score"), default=0.0), 10),
        "normalized_risk_adjusted_candidate_score": round(
            bounded(row.get("normalized_risk_adjusted_candidate_score"), default=0.0),
            10,
        ),
    }


def _empty_summary(config: PortfolioAllocatorConfig, *, no_allocation_count: int, candidate_count: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "portfolio_status": "all_cash",
                "candidate_count": int(candidate_count),
                "allocation_candidate_count": 0,
                "no_allocation_count": int(no_allocation_count),
                "total_exposure": 0.0,
                "cash_weight": 1.0,
                "min_cash_buffer": float(config.min_cash_buffer),
                "max_total_exposure": float(config.max_total_exposure),
                "effective_max_exposure": float(config.effective_max_exposure),
                "diagnostic_only_authority": True,
                "no_buy_sell_recommendation_authority": True,
                "no_forced_trade_rule": True,
            }
        ],
        columns=list(PORTFOLIO_SUMMARY_COLUMNS),
    )


def _build_summary(allocation: pd.DataFrame, config: PortfolioAllocatorConfig, *, candidate_count: int) -> pd.DataFrame:
    if allocation.empty:
        return _empty_summary(config, no_allocation_count=0, candidate_count=candidate_count)
    allocated = allocation[allocation["allocation_status"] == "allocation_candidate"].copy()
    total_exposure = round(float(pd.to_numeric(allocated.get("final_weight"), errors="coerce").fillna(0.0).sum()), 10)
    allocation_count = int(len(allocated))
    no_allocation_count = int((allocation["allocation_status"] == "no_allocation").sum())
    return pd.DataFrame(
        [
            {
                "portfolio_status": "allocation_candidate" if allocation_count else "all_cash",
                "candidate_count": int(candidate_count),
                "allocation_candidate_count": allocation_count,
                "no_allocation_count": no_allocation_count,
                "total_exposure": total_exposure,
                "cash_weight": round(1.0 - total_exposure, 10),
                "min_cash_buffer": float(config.min_cash_buffer),
                "max_total_exposure": float(config.max_total_exposure),
                "effective_max_exposure": float(config.effective_max_exposure),
                "diagnostic_only_authority": True,
                "no_buy_sell_recommendation_authority": True,
                "no_forced_trade_rule": True,
            }
        ],
        columns=list(PORTFOLIO_SUMMARY_COLUMNS),
    )


def _build_risk_summary(allocation: pd.DataFrame, summary: pd.DataFrame, config: PortfolioAllocatorConfig) -> pd.DataFrame:
    if allocation.empty:
        total_exposure = 0.0
        cash_weight = 1.0
        allocated = allocation.copy()
    else:
        total_exposure = float(summary.loc[0, "total_exposure"])
        cash_weight = float(summary.loc[0, "cash_weight"])
        allocated = allocation[allocation["allocation_status"] == "allocation_candidate"].copy()

    if allocated.empty or total_exposure <= 0.0:
        weighted_risk = 0.0
        max_allocated_risk = 0.0
        max_single_position = 0.0
    else:
        weights = pd.to_numeric(allocated["final_weight"], errors="coerce").fillna(0.0)
        risks = pd.to_numeric(allocated["risk_score"], errors="coerce").fillna(1.0)
        weighted_risk = round(float((weights * risks).sum() / total_exposure), 10)
        max_allocated_risk = round(float(risks.max()), 10)
        max_single_position = round(float(weights.max()), 10)

    risk_levels = allocation.get("risk_level", pd.Series(dtype="object")).astype(str) if not allocation.empty else pd.Series(dtype="object")
    return pd.DataFrame(
        [
            {
                "portfolio_status": summary.loc[0, "portfolio_status"],
                "allocation_candidate_count": int(summary.loc[0, "allocation_candidate_count"]),
                "total_exposure": total_exposure,
                "cash_weight": cash_weight,
                "max_position_weight": float(config.max_position_weight),
                "max_single_position_weight": max_single_position,
                "weighted_average_risk_score": weighted_risk,
                "max_allocated_risk_score": max_allocated_risk,
                "level_1_soft_adjustment_count": int((risk_levels == "level_1_soft_adjustment").sum()),
                "level_2_candidate_filtering_count": int((risk_levels == "level_2_candidate_filtering").sum()),
                "level_3_hard_override_count": int((risk_levels == "level_3_hard_override").sum()),
                "no_allocation_count": int(summary.loc[0, "no_allocation_count"]),
            }
        ],
        columns=list(PORTFOLIO_RISK_SUMMARY_COLUMNS),
    )


def _decision_cards(allocation: pd.DataFrame) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for row in allocation.to_dict(orient="records"):
        cards.append(
            {
                "allocation_id": row.get("allocation_id"),
                "ticker": row.get("ticker"),
                "allocation_status": row.get("allocation_status"),
                "final_weight": row.get("final_weight"),
                "dominant_scenario": row.get("dominant_scenario"),
                "risk_level": row.get("risk_level"),
                "risk_adjusted_confidence": row.get("risk_adjusted_confidence"),
                "no_allocation_reason": row.get("no_allocation_reason"),
                "allocation_reason_codes": row.get("allocation_reason_codes"),
                "diagnostic_only_authority": True,
                "no_buy_sell_recommendation_authority": True,
            }
        )
    return cards


def _missing_enriched_result(config: PortfolioAllocatorConfig) -> PortfolioAllocatorResult:
    allocation = pd.DataFrame(
        [
            {
                "allocation_id": "portfolio_allocator_v1|missing_enriched_candidates",
                "candidate_id": "",
                "source_packet_id": "",
                "timestamp": "",
                "ticker": "",
                "horizon": "",
                "target_type": "",
                "run_mode": "",
                "allocation_status": "no_allocation",
                "no_allocation_reason": "missing_enriched_candidates",
                "risk_adjusted_confidence": 0.0,
                "risk_adjusted_candidate_score": 0.0,
                "risk_score": 1.0,
                "risk_level": "",
                "risk_action": "",
                "disagreement_score": 1.0,
                "dominance_score": 0.0,
                "dominant_scenario": "",
                "dominant_scenario_probability": 0.0,
                "scenario_alignment": "unknown",
                "raw_weight": 0.0,
                "final_weight": 0.0,
                "exposure_before_allocation": 0.0,
                "exposure_after_allocation": 0.0,
                "cash_buffer_after_allocation": 1.0,
                "allocation_reason_codes": "missing_enriched_candidates",
                "allocation_priority_score": 0.0,
                "normalized_risk_adjusted_candidate_score": 0.0,
            }
        ],
        columns=list(ALLOCATION_OUTPUT_COLUMNS),
    )
    summary = _build_summary(allocation, config, candidate_count=0)
    risk_summary = _build_risk_summary(allocation, summary, config)
    cards = _decision_cards(allocation)
    manifest = build_allocator_manifest(
        config=config,
        input_row_counts={
            "decision_lane_enriched_candidates": 0,
            "risk_adjusted_candidates": 0,
            "scenario_dominance_summary": 0,
        },
        portfolio_allocation=allocation,
        portfolio_summary=summary,
        portfolio_risk_summary=risk_summary,
        decision_card_count=len(cards),
        artifact_paths={},
        missing_enriched_candidates=True,
    )
    return PortfolioAllocatorResult(allocation, summary, risk_summary, cards, manifest)


def run_portfolio_allocator(
    decision_lane_enriched_candidates_df: pd.DataFrame | None = None,
    *,
    enriched_candidates_df: pd.DataFrame | None = None,
    risk_adjusted_candidates_df: pd.DataFrame | None = None,
    scenario_dominance_df: pd.DataFrame | None = None,
    config: PortfolioAllocatorConfig | None = None,
    missing_enriched_candidates: bool = False,
) -> PortfolioAllocatorResult:
    """Convert Decision Lane v2 enriched candidates into allocation diagnostics."""

    resolved = config or PortfolioAllocatorConfig()
    if decision_lane_enriched_candidates_df is None and enriched_candidates_df is not None:
        decision_lane_enriched_candidates_df = enriched_candidates_df
    if missing_enriched_candidates or decision_lane_enriched_candidates_df is None:
        return _missing_enriched_result(resolved)

    enriched = _frame(decision_lane_enriched_candidates_df)
    risk_adjusted = _frame(risk_adjusted_candidates_df)
    dominance = _frame(scenario_dominance_df)
    input_row_counts = {
        "decision_lane_enriched_candidates": int(len(enriched)),
        "risk_adjusted_candidates": int(len(risk_adjusted)),
        "scenario_dominance_summary": int(len(dominance)),
    }
    if enriched.empty:
        allocation = empty_allocation_frame()
        summary = _empty_summary(resolved, no_allocation_count=0, candidate_count=0)
        risk_summary = _build_risk_summary(allocation, summary, resolved)
        cards: list[dict[str, Any]] = []
        manifest = build_allocator_manifest(
            config=resolved,
            input_row_counts=input_row_counts,
            portfolio_allocation=allocation,
            portfolio_summary=summary,
            portfolio_risk_summary=risk_summary,
            decision_card_count=0,
            artifact_paths={},
            missing_enriched_candidates=False,
        )
        return PortfolioAllocatorResult(allocation, summary, risk_summary, cards, manifest)

    prepared = _prepare_candidates(
        enriched,
        risk_adjusted_candidates_df=risk_adjusted,
        scenario_dominance_df=dominance,
    )

    gated_rows: list[tuple[pd.Series, dict[str, Any], str, list[str]]] = []
    pass_rows: list[tuple[pd.Series, dict[str, Any]]] = []
    for index, row in prepared.iterrows():
        base = _base_output_row(row, _allocation_id(row, int(index) + 1))
        gate = evaluate_allocation_gate(row, resolved)
        if gate.passed:
            pass_rows.append((row, base))
        else:
            gated_rows.append((row, base, gate.no_allocation_reason, gate.reason_codes))

    pass_rows = sorted(
        pass_rows,
        key=lambda item: (
            -safe_float(item[1].get("allocation_priority_score"), default=0.0),
            safe_float(item[1].get("disagreement_score"), default=1.0),
            -safe_float(item[1].get("dominance_score"), default=0.0),
            -safe_float(item[1].get("risk_adjusted_confidence"), default=0.0),
            -safe_float(item[1].get("risk_adjusted_candidate_score"), default=0.0),
            safe_float(item[1].get("risk_score"), default=1.0),
            normalize_text(item[1].get("candidate_id")),
        ),
    )
    gated_rows = sorted(
        gated_rows,
        key=lambda item: (
            -safe_float(item[1].get("allocation_priority_score"), default=0.0),
            normalize_text(item[1].get("candidate_id")),
        ),
    )

    output_rows: list[dict[str, Any]] = []
    current_exposure = 0.0
    for source_row, base in pass_rows:
        raw_weight = calculate_raw_weight(source_row, resolved)
        remaining_room = max(0.0, resolved.effective_max_exposure - current_exposure)
        final_weight = min(raw_weight, float(resolved.max_position_weight), remaining_room)
        final_weight = round(max(0.0, final_weight), 10)
        exposure_before = round(current_exposure, 10)
        sizing_codes: list[str] = ["passed_allocator_gates"]
        if final_weight < raw_weight:
            sizing_codes.append("reduced_to_remaining_exposure_room")

        if final_weight < float(resolved.min_position_weight):
            reason = "insufficient_exposure_room" if remaining_room < float(resolved.min_position_weight) else "final_weight_below_min_position"
            output_rows.append(
                {
                    **base,
                    "allocation_status": "no_allocation",
                    "no_allocation_reason": reason,
                    "raw_weight": raw_weight,
                    "final_weight": 0.0,
                    "exposure_before_allocation": exposure_before,
                    "exposure_after_allocation": exposure_before,
                    "cash_buffer_after_allocation": round(1.0 - exposure_before, 10),
                    "allocation_reason_codes": combine_reason_codes(source_row.get("reason_codes"), sizing_codes, [reason]),
                }
            )
            continue

        current_exposure = round(current_exposure + final_weight, 10)
        output_rows.append(
            {
                **base,
                "allocation_status": "allocation_candidate",
                "no_allocation_reason": "",
                "raw_weight": raw_weight,
                "final_weight": final_weight,
                "exposure_before_allocation": exposure_before,
                "exposure_after_allocation": current_exposure,
                "cash_buffer_after_allocation": round(1.0 - current_exposure, 10),
                "allocation_reason_codes": combine_reason_codes(
                    source_row.get("reason_codes"),
                    sizing_codes,
                    ["sized_within_portfolio_constraints"],
                ),
            }
        )

    for source_row, base, reason, reason_codes in gated_rows:
        output_rows.append(
            {
                **base,
                "allocation_status": "no_allocation",
                "no_allocation_reason": reason,
                "raw_weight": 0.0,
                "final_weight": 0.0,
                "exposure_before_allocation": 0.0,
                "exposure_after_allocation": 0.0,
                "cash_buffer_after_allocation": 1.0,
                "allocation_reason_codes": combine_reason_codes(source_row.get("reason_codes"), reason_codes),
            }
        )

    allocation = pd.DataFrame(output_rows)
    if allocation.empty:
        allocation = empty_allocation_frame()
    else:
        allocation = allocation[list(ALLOCATION_OUTPUT_COLUMNS)]
    summary = _build_summary(allocation, resolved, candidate_count=len(enriched))
    risk_summary = _build_risk_summary(allocation, summary, resolved)
    cards = _decision_cards(allocation)
    manifest = build_allocator_manifest(
        config=resolved,
        input_row_counts=input_row_counts,
        portfolio_allocation=allocation,
        portfolio_summary=summary,
        portfolio_risk_summary=risk_summary,
        decision_card_count=len(cards),
        artifact_paths={},
        missing_enriched_candidates=False,
    )
    return PortfolioAllocatorResult(allocation, summary, risk_summary, cards, manifest)


allocate_portfolio = run_portfolio_allocator
