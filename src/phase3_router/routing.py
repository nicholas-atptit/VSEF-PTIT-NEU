"""Deterministic routing orchestration for Phase 3 Router v1."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.phase3_router.guards import evaluate_route_guard
from src.phase3_router.manifest import build_router_manifest
from src.phase3_router.schema import (
    ROUTE_DECISIONS,
    ROUTER_DECISION_COLUMNS,
    ROUTER_SUMMARY_COLUMNS,
    Phase3RouterConfig,
    Phase3RouterResult,
    PortfolioContext,
    bounded,
    normalize_text,
    safe_float,
)


def _frame(value: pd.DataFrame | None) -> pd.DataFrame:
    return value.copy() if value is not None else pd.DataFrame()


def _series(frame: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def _first_present(row: pd.Series | dict[str, Any], columns: tuple[str, ...], default: Any = None) -> Any:
    for column in columns:
        value = row.get(column) if isinstance(row, dict) else row.get(column)
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
        if text and text.lower() != "nan":
            return value
    return default


def _fill_missing_from_suffix(frame: pd.DataFrame, column: str, suffix_column: str) -> pd.DataFrame:
    if suffix_column not in frame.columns:
        return frame
    if column not in frame.columns:
        frame[column] = frame[suffix_column]
    else:
        current = frame[column]
        missing = current.isna() | current.astype(str).str.strip().isin(["", "nan", "None"])
        frame[column] = current.where(~missing, frame[suffix_column])
    return frame.drop(columns=[suffix_column])


def _merge_risk_summary(allocation: pd.DataFrame, portfolio_risk_summary: pd.DataFrame) -> pd.DataFrame:
    if allocation.empty or portfolio_risk_summary.empty or "ticker" not in allocation.columns:
        return allocation
    risk = portfolio_risk_summary.copy()
    if "ticker" not in risk.columns:
        return allocation
    keep = [
        "ticker",
        *[
            column
            for column in (
                "risk_level",
                "risk_score",
                "dominant_scenario",
                "scenario_alignment",
                "volatility_regime",
                "volatility_bucket",
            )
            if column in risk.columns
        ],
    ]
    right = risk[keep].drop_duplicates("ticker", keep="first")
    renamed = right.rename(columns={column: f"{column}__risk_summary" for column in keep if column != "ticker"})
    merged = allocation.merge(renamed, on="ticker", how="left")
    for column in keep:
        if column != "ticker":
            merged = _fill_missing_from_suffix(merged, column, f"{column}__risk_summary")
    return merged


def _standardize_allocation(
    portfolio_allocation: pd.DataFrame,
    portfolio_risk_summary: pd.DataFrame,
) -> pd.DataFrame:
    allocation = _merge_risk_summary(portfolio_allocation.copy(), portfolio_risk_summary)
    if allocation.empty:
        return allocation

    standardized = allocation.copy()
    if "allocation_status" not in standardized.columns and "decision_label" in standardized.columns:
        standardized["allocation_status"] = standardized["decision_label"]
    if "allocation_status" not in standardized.columns:
        standardized["allocation_status"] = "no_allocation"

    if "source_packet_id" not in standardized.columns and "packet_id" in standardized.columns:
        standardized["source_packet_id"] = standardized["packet_id"]
    if "source_packet_id" not in standardized.columns:
        standardized["source_packet_id"] = ""
    if "candidate_id" not in standardized.columns:
        standardized["candidate_id"] = [
            f"allocation_candidate|{packet_id}" if normalize_text(packet_id) else f"candidate_{index:04d}"
            for index, packet_id in enumerate(standardized["source_packet_id"], start=1)
        ]

    if "final_weight" not in standardized.columns:
        if "allocation_weight" in standardized.columns:
            standardized["final_weight"] = standardized["allocation_weight"]
        elif "invested_weight" in standardized.columns:
            standardized["final_weight"] = standardized["invested_weight"]
        else:
            standardized["final_weight"] = 0.0
    if "risk_adjusted_confidence" not in standardized.columns:
        if "model_agreement_score" in standardized.columns:
            standardized["risk_adjusted_confidence"] = standardized["model_agreement_score"]
        elif "agreement_score" in standardized.columns:
            standardized["risk_adjusted_confidence"] = standardized["agreement_score"]
        else:
            standardized["risk_adjusted_confidence"] = 0.0
    if "disagreement_score" not in standardized.columns:
        agreement = pd.to_numeric(
            _series(
                standardized,
                "model_agreement_score",
                _series(standardized, "agreement_score", 0.0),
            ),
            errors="coerce",
        ).fillna(0.0)
        standardized["disagreement_score"] = 1.0 - agreement
    if "dominance_score" not in standardized.columns:
        standardized["dominance_score"] = _series(standardized, "scenario_dominance_score", 1.0)
    if "dominant_scenario" not in standardized.columns:
        standardized["dominant_scenario"] = _series(standardized, "regime_label", "")

    defaults: dict[str, Any] = {
        "allocation_id": "",
        "timestamp": "",
        "ticker": "",
        "horizon": "",
        "risk_level": "level_1_soft_adjustment",
        "risk_score": 0.0,
        "scenario_alignment": "unknown",
    }
    for column, default in defaults.items():
        if column not in standardized.columns:
            standardized[column] = default

    for column, default in (
        ("final_weight", 0.0),
        ("risk_score", 0.0),
        ("risk_adjusted_confidence", 0.0),
        ("disagreement_score", 1.0),
        ("dominance_score", 1.0),
    ):
        standardized[column] = pd.to_numeric(_series(standardized, column, default), errors="coerce").fillna(default)

    sort_columns = [column for column in ("timestamp", "ticker", "allocation_id", "candidate_id") if column in standardized.columns]
    if sort_columns:
        standardized = standardized.sort_values(sort_columns).reset_index(drop=True)
    return standardized


def _summary_value(summary: pd.DataFrame, columns: tuple[str, ...], default: Any) -> Any:
    if summary.empty:
        return default
    row = summary.iloc[0]
    return _first_present(row, columns, default=default)


def _build_portfolio_context(
    allocation: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    config: Phase3RouterConfig,
) -> PortfolioContext:
    total_exposure_default = 0.0
    if not allocation.empty and "allocation_status" in allocation.columns and "final_weight" in allocation.columns:
        allocated = allocation[allocation["allocation_status"].astype(str) == "allocation_candidate"]
        total_exposure_default = float(pd.to_numeric(allocated["final_weight"], errors="coerce").fillna(0.0).sum())

    total_exposure = safe_float(
        _summary_value(portfolio_summary, ("total_exposure", "invested_exposure"), total_exposure_default),
        default=total_exposure_default,
    )
    cash_weight = safe_float(
        _summary_value(portfolio_summary, ("cash_weight",), 1.0 - total_exposure),
        default=1.0 - total_exposure,
    )
    min_cash_buffer = safe_float(
        _summary_value(portfolio_summary, ("min_cash_buffer", "cash_buffer"), 0.0),
        default=0.0,
    )
    max_total_exposure = safe_float(
        _summary_value(portfolio_summary, ("max_total_exposure",), 1.0),
        default=1.0,
    )
    effective_max_exposure = safe_float(
        _summary_value(portfolio_summary, ("effective_max_exposure",), min(max_total_exposure, 1.0 - min_cash_buffer)),
        default=min(max_total_exposure, 1.0 - min_cash_buffer),
    )
    return PortfolioContext(
        portfolio_status=normalize_text(
            _summary_value(portfolio_summary, ("portfolio_status", "portfolio_label"), "unknown"),
            default="unknown",
        ),
        total_exposure=round(max(0.0, total_exposure), 10),
        cash_weight=round(max(0.0, min(1.0, cash_weight)), 10),
        min_cash_buffer=round(max(0.0, min(1.0, min_cash_buffer)), 10),
        max_total_exposure=round(max(0.0, min(1.0, max_total_exposure)), 10),
        effective_max_exposure=round(max(0.0, min(1.0, effective_max_exposure)), 10),
    )


def _router_decision_id(row: pd.Series, index: int) -> str:
    allocation_id = normalize_text(row.get("allocation_id"), default=f"allocation_{index:04d}")
    return f"phase3_router_v1|{allocation_id}"


def _route_row(row: pd.Series, context: PortfolioContext, config: Phase3RouterConfig, index: int) -> dict[str, Any]:
    record = row.to_dict()
    route = evaluate_route_guard(record, context, config)
    return {
        "router_decision_id": _router_decision_id(row, index),
        "allocation_id": normalize_text(row.get("allocation_id")),
        "candidate_id": normalize_text(row.get("candidate_id")),
        "source_packet_id": normalize_text(row.get("source_packet_id")),
        "timestamp": row.get("timestamp", ""),
        "ticker": normalize_text(row.get("ticker")),
        "horizon": row.get("horizon", ""),
        "route_decision": route.route_decision,
        "route_reason": route.route_reason,
        "allocation_status": normalize_text(row.get("allocation_status"), default="no_allocation"),
        "final_weight": round(safe_float(row.get("final_weight"), default=0.0), 10),
        "risk_level": normalize_text(row.get("risk_level"), default="level_1_soft_adjustment"),
        "risk_score": round(bounded(row.get("risk_score"), default=0.0), 10),
        "risk_adjusted_confidence": round(bounded(row.get("risk_adjusted_confidence"), default=0.0), 10),
        "disagreement_score": round(bounded(row.get("disagreement_score"), default=1.0), 10),
        "dominance_score": round(bounded(row.get("dominance_score"), default=1.0), 10),
        "scenario_alignment": normalize_text(row.get("scenario_alignment"), default="unknown"),
        "dominant_scenario": normalize_text(row.get("dominant_scenario")),
        "portfolio_status": context.portfolio_status,
        "total_exposure": context.total_exposure,
        "cash_weight": context.cash_weight,
        "route_reason_codes": "|".join(route.reason_codes),
        "diagnostic_only_authority": True,
        "no_buy_sell_recommendation_authority": True,
    }


def _missing_allocator_row(reason: str, context: PortfolioContext) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "router_decision_id": "phase3_router_v1|missing_allocator_outputs",
                "allocation_id": "",
                "candidate_id": "",
                "source_packet_id": "",
                "timestamp": "",
                "ticker": "",
                "horizon": "",
                "route_decision": "no_candidate",
                "route_reason": reason,
                "allocation_status": "missing_allocator_outputs",
                "final_weight": 0.0,
                "risk_level": "",
                "risk_score": 0.0,
                "risk_adjusted_confidence": 0.0,
                "disagreement_score": 1.0,
                "dominance_score": 0.0,
                "scenario_alignment": "unknown",
                "dominant_scenario": "",
                "portfolio_status": context.portfolio_status,
                "total_exposure": context.total_exposure,
                "cash_weight": context.cash_weight,
                "route_reason_codes": reason,
                "diagnostic_only_authority": True,
                "no_buy_sell_recommendation_authority": True,
            }
        ],
        columns=list(ROUTER_DECISION_COLUMNS),
    )


def _build_router_decisions(
    allocation: pd.DataFrame,
    context: PortfolioContext,
    config: Phase3RouterConfig,
    *,
    missing_allocator_outputs: bool,
) -> pd.DataFrame:
    if missing_allocator_outputs:
        return _missing_allocator_row("missing_allocator_outputs", context)
    if allocation.empty:
        return _missing_allocator_row("empty_portfolio_allocation", context)

    rows = [_route_row(row, context, config, index) for index, (_, row) in enumerate(allocation.iterrows(), start=1)]
    decisions = pd.DataFrame(rows)
    return decisions[list(ROUTER_DECISION_COLUMNS)]


def _build_router_summary(router_decisions: pd.DataFrame, context: PortfolioContext) -> pd.DataFrame:
    counts = (
        router_decisions["route_decision"].astype(str).value_counts().to_dict()
        if not router_decisions.empty and "route_decision" in router_decisions.columns
        else {}
    )
    routed = router_decisions[
        router_decisions["route_decision"].astype(str) == "route_allocation_candidate"
    ] if not router_decisions.empty else pd.DataFrame()
    status = "routes_available" if counts.get("route_allocation_candidate", 0) else "no_routed_candidates"
    row = {
        "router_status": status,
        "source_allocation_count": int(len(router_decisions)),
        "route_allocation_candidate_count": int(counts.get("route_allocation_candidate", 0)),
        "hold_count": int(counts.get("hold", 0)),
        "reject_count": int(counts.get("reject", 0)),
        "no_candidate_count": int(counts.get("no_candidate", 0)),
        "routed_final_weight": round(
            float(pd.to_numeric(routed.get("final_weight"), errors="coerce").fillna(0.0).sum()) if not routed.empty else 0.0,
            10,
        ),
        "total_exposure": context.total_exposure,
        "cash_weight": context.cash_weight,
        "diagnostic_only_authority": True,
        "no_buy_sell_recommendation_authority": True,
    }
    return pd.DataFrame([row], columns=list(ROUTER_SUMMARY_COLUMNS))


def run_phase3_router(
    portfolio_allocation_df: pd.DataFrame | None = None,
    *,
    portfolio_summary_df: pd.DataFrame | None = None,
    portfolio_risk_summary_df: pd.DataFrame | None = None,
    allocator_manifest: dict[str, Any] | None = None,
    config: Phase3RouterConfig | None = None,
    missing_allocator_outputs: bool = False,
) -> Phase3RouterResult:
    """Convert Portfolio Allocator v1 outputs into diagnostic route decisions."""

    resolved = config or Phase3RouterConfig()
    raw_allocation = _frame(portfolio_allocation_df)
    portfolio_summary = _frame(portfolio_summary_df)
    portfolio_risk_summary = _frame(portfolio_risk_summary_df)
    allocation = _standardize_allocation(raw_allocation, portfolio_risk_summary)
    context = _build_portfolio_context(allocation, portfolio_summary, resolved)
    router_decisions = _build_router_decisions(
        allocation,
        context,
        resolved,
        missing_allocator_outputs=missing_allocator_outputs,
    )
    router_summary = _build_router_summary(router_decisions, context)
    manifest = build_router_manifest(
        config=resolved,
        input_row_counts={
            "portfolio_allocation": int(len(raw_allocation)),
            "portfolio_summary": int(len(portfolio_summary)),
            "portfolio_risk_summary": int(len(portfolio_risk_summary)),
        },
        router_decisions=router_decisions,
        router_summary=router_summary,
        artifact_paths={},
        allocator_manifest=allocator_manifest,
        missing_allocator_outputs=missing_allocator_outputs,
    )
    return Phase3RouterResult(
        router_decisions=router_decisions,
        router_summary=router_summary,
        manifest=manifest,
    )


route_portfolio_allocations = run_phase3_router
