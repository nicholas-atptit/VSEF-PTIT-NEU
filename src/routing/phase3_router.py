"""Legacy deterministic Phase 3 Router v1 for saved Quant Core diagnostics.

The router emits auditable route decisions only. It does not train a meta-model
and does not create trading recommendations.

Canonical Phase 3 Router v1 artifacts and route decisions live in
``src.phase3_router``. This module is retained for compatibility with older
tests and local artifacts that used the pre-canonical route-label schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_ALLOCATOR_INPUT_FILES: tuple[str, ...] = (
    "portfolio_allocation.csv",
    "portfolio_summary.csv",
    "portfolio_risk_summary.csv",
    "portfolio_decision_cards.jsonl",
    "allocator_manifest.json",
)

OPTIONAL_DIAGNOSTIC_INPUT_FILES: tuple[str, ...] = (
    "model_consensus_summary.csv",
    "model_health_summary.csv",
    "risk_summary.csv",
    "regime_summary.csv",
    "strategy_metrics.csv",
)

LEGACY_PHASE3_ROUTER_OUTPUT_FILES: tuple[str, ...] = (
    "route_decision.csv",
    "phase3_decision_cards.jsonl",
    "routing_summary.csv",
    "routing_manifest.json",
)

PHASE3_ROUTER_OUTPUT_FILES = LEGACY_PHASE3_ROUTER_OUTPUT_FILES

LEGACY_PHASE3_ROUTE_LABELS: tuple[str, ...] = (
    "route_allocation_candidate",
    "hold_for_review",
    "reject_low_confidence",
    "reject_high_risk",
    "reject_low_agreement",
    "reject_unhealthy_model",
    "reject_allocator_no_allocation",
    "no_candidate",
    "rejected_missing_required_data",
)

PHASE3_ROUTE_LABELS = LEGACY_PHASE3_ROUTE_LABELS

CORE_ALLOCATION_COLUMNS: tuple[str, ...] = ("ticker", "allocation_weight")


@dataclass(frozen=True)
class Phase3RouterConfig:
    """Conservative deterministic routing settings."""

    min_allocation_weight: float = 0.01
    min_candidate_score: float = 0.0
    min_model_agreement: float = 0.5
    max_risk_score: float = 1.0
    require_positive_allocation: bool = True
    allow_no_allocation: bool = True
    severe_drawdown_blocks: bool = True
    unhealthy_model_blocks: bool = True
    low_agreement_action: str = "hold_for_review"
    output_label: str = "route_allocation_candidate"

    def __post_init__(self) -> None:
        if self.output_label != "route_allocation_candidate":
            raise ValueError("Phase 3 Router v1 only supports route_allocation_candidate output labels")
        if self.low_agreement_action not in {"hold_for_review", "reject_low_agreement"}:
            raise ValueError("low_agreement_action must be hold_for_review or reject_low_agreement")
        for name in ("min_allocation_weight", "min_candidate_score", "min_model_agreement", "max_risk_score"):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase3RouterResult:
    route_decision: pd.DataFrame
    decision_cards: list[dict[str, Any]]
    routing_summary: pd.DataFrame
    manifest: dict[str, Any]
    output_paths: dict[str, Path]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(number):
        return float(default)
    return float(number)


def _safe_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _first_value(row: pd.Series, columns: list[str]) -> Any:
    for column in columns:
        if column in row and pd.notna(row.get(column)):
            return row.get(column)
    return None


def _first_float(row: pd.Series, columns: list[str], *, default: float = 0.0) -> float:
    value = _first_value(row, columns)
    return _safe_float(value, default=default)


def _input_status_for_csv(path: Path) -> dict[str, Any]:
    frame = _read_csv(path)
    return {
        "path": str(path),
        "exists": bool(path.exists()),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
    }


def _load_inputs(input_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    frames: dict[str, pd.DataFrame] = {}
    jsonl_rows: dict[str, list[dict[str, Any]]] = {}
    json_docs: dict[str, dict[str, Any]] = {}
    status: dict[str, dict[str, Any]] = {}

    for filename in REQUIRED_ALLOCATOR_INPUT_FILES + OPTIONAL_DIAGNOSTIC_INPUT_FILES:
        path = input_dir / filename
        if filename.endswith(".csv"):
            frame = _read_csv(path)
            frames[filename] = frame
            status[filename] = {
                "path": str(path),
                "exists": bool(path.exists()),
                "rows": int(len(frame)),
                "columns": list(frame.columns),
            }
        elif filename.endswith(".jsonl"):
            rows = _read_jsonl(path)
            jsonl_rows[filename] = rows
            status[filename] = {
                "path": str(path),
                "exists": bool(path.exists()),
                "rows": int(len(rows)),
                "columns": sorted({key for row in rows for key in row}),
            }
        elif filename.endswith(".json"):
            doc = _read_json(path)
            json_docs[filename] = doc
            status[filename] = {
                "path": str(path),
                "exists": bool(path.exists()),
                "rows": 1 if doc else 0,
                "columns": sorted(doc.keys()),
            }
    return frames, jsonl_rows, json_docs, status


def _missing_required_inputs(input_status: dict[str, dict[str, Any]]) -> list[str]:
    missing: list[str] = []
    for filename in REQUIRED_ALLOCATOR_INPUT_FILES:
        status = input_status.get(filename, {})
        if not status.get("exists"):
            missing.append(filename)
    return missing


def _shared_columns(left: pd.DataFrame, right: pd.DataFrame, preferred: list[str]) -> list[str]:
    return [column for column in preferred if column in left.columns and column in right.columns]


def _dedupe_for_merge(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty or not keys:
        return pd.DataFrame()
    return frame.sort_values(keys).drop_duplicates(keys, keep="first").reset_index(drop=True)


def _merge_optional(
    allocation: pd.DataFrame,
    frame: pd.DataFrame,
    preferred_keys: list[str],
    *,
    suffix: str,
) -> pd.DataFrame:
    if allocation.empty or frame.empty:
        return allocation
    keys = _shared_columns(allocation, frame, preferred_keys)
    if not keys:
        return allocation
    right = _dedupe_for_merge(frame, keys)
    return allocation.merge(right, on=keys, how="left", suffixes=("", suffix))


def _merge_health(allocation: pd.DataFrame, health: pd.DataFrame) -> pd.DataFrame:
    if allocation.empty or health.empty or "primary_model_name" not in allocation.columns or "model_name" not in health.columns:
        return allocation
    right = health.sort_values("model_name").drop_duplicates("model_name", keep="first").copy()
    right = right.rename(columns={column: f"{column}_health" for column in right.columns if column != "model_name"})
    return allocation.merge(right, left_on="primary_model_name", right_on="model_name", how="left")


def _merge_strategy(allocation: pd.DataFrame, strategy: pd.DataFrame) -> pd.DataFrame:
    if allocation.empty or strategy.empty:
        return allocation
    strategy_frame = strategy.copy()
    if "model_name" in strategy_frame.columns and "primary_model_name" in allocation.columns:
        strategy_frame = strategy_frame.rename(columns={"model_name": "primary_model_name"})
    keys = _shared_columns(strategy_frame, allocation, ["primary_model_name", "horizon", "target_type", "run_mode", "group_name"])
    if not keys:
        return allocation
    keep = [
        *keys,
        *[
            column
            for column in ("sharpe", "cagr", "max_drawdown", "total_return", "win_rate", "trade_count", "policy_variant")
            if column in strategy_frame.columns
        ],
    ]
    right = _dedupe_for_merge(strategy_frame[keep], keys)
    return allocation.merge(right, on=keys, how="left", suffixes=("", "_strategy"))


def _prepare_allocation(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    allocation = frames.get("portfolio_allocation.csv", pd.DataFrame()).copy()
    if allocation.empty:
        return allocation

    portfolio_risk = frames.get("portfolio_risk_summary.csv", pd.DataFrame())
    allocation = _merge_optional(allocation, portfolio_risk, ["ticker"], suffix="_portfolio_risk")
    allocation = _merge_optional(
        allocation,
        frames.get("model_consensus_summary.csv", pd.DataFrame()),
        ["timestamp", "ticker", "horizon", "target_type", "run_mode", "core_run_id", "group_name"],
        suffix="_consensus",
    )
    allocation = _merge_optional(
        allocation,
        frames.get("risk_summary.csv", pd.DataFrame()),
        ["timestamp", "ticker", "horizon", "target_type", "run_mode", "core_run_id", "group_name"],
        suffix="_risk",
    )
    allocation = _merge_optional(
        allocation,
        frames.get("regime_summary.csv", pd.DataFrame()),
        ["timestamp", "ticker", "horizon", "target_type", "run_mode", "core_run_id", "group_name"],
        suffix="_regime",
    )
    allocation = _merge_health(allocation, frames.get("model_health_summary.csv", pd.DataFrame()))
    allocation = _merge_strategy(allocation, frames.get("strategy_metrics.csv", pd.DataFrame()))
    sort_columns = [column for column in ["ticker", "allocation_id", "packet_id", "timestamp"] if column in allocation.columns]
    if sort_columns:
        allocation = allocation.sort_values(sort_columns)
    return allocation.reset_index(drop=True)


def _allocator_no_allocation(allocation: pd.DataFrame, summary: pd.DataFrame, manifest: dict[str, Any]) -> tuple[bool, str]:
    manifest_reason = _safe_str(manifest.get("no_allocation_reason"))
    if manifest_reason:
        return True, manifest_reason
    if not summary.empty:
        label = _safe_str(summary.iloc[0].get("portfolio_label")).lower()
        reason = _safe_str(summary.iloc[0].get("no_allocation_reason"))
        if label == "no_allocation":
            return True, reason or "allocator_no_allocation"
    if not allocation.empty:
        labels = allocation.get("decision_label", pd.Series(dtype="object")).astype(str).str.lower()
        tickers = allocation.get("ticker", pd.Series(dtype="object")).astype(str).str.upper()
        if labels.eq("no_allocation").any() or tickers.eq("CASH").all():
            reason = _safe_str(allocation.iloc[0].get("reason"))
            return True, reason or "allocator_no_allocation"
    return False, ""


def _agreement_score(row: pd.Series) -> float | None:
    for column in ("model_agreement_score", "agreement_score", "agreement_score_consensus"):
        if column in row and pd.notna(row.get(column)):
            return _safe_float(row.get(column))
    return None


def _candidate_score(row: pd.Series) -> float:
    return _first_float(row, ["candidate_score", "base_candidate_score", "adjusted_score", "route_score"], default=0.0)


def _risk_score(row: pd.Series) -> float:
    numeric_columns = [
        "risk_score",
        "risk_score_portfolio_risk",
        "risk_score_risk",
        "portfolio_risk_score",
        "vol_forecast",
        "volatility",
        "var_loss_95",
        "cvar_loss_95",
    ]
    values = [
        value
        for value in (
            _safe_float(row.get(column), default=np.nan)
            for column in numeric_columns
            if column in row and pd.notna(row.get(column))
        )
        if np.isfinite(value)
    ]
    drawdown_values = [
        abs(_safe_float(row.get(column), default=np.nan))
        for column in ("current_drawdown", "max_drawdown", "max_drawdown_strategy")
        if column in row and pd.notna(row.get(column))
    ]
    values.extend(value for value in drawdown_values if np.isfinite(value))
    return float(max(values)) if values else 0.0


def _severe_drawdown(row: pd.Series) -> bool:
    for column in ("drawdown_state", "drawdown_state_portfolio_risk", "drawdown_state_risk"):
        if _safe_str(row.get(column)).lower() == "severe":
            return True
    return False


def _health_status(row: pd.Series) -> str:
    return _safe_str(_first_value(row, ["health_status", "health_status_health"])).lower()


def _unhealthy_model(row: pd.Series) -> bool:
    status = _health_status(row)
    if status in {"failing", "weak"}:
        return True
    success_rate = _first_value(row, ["run_success_rate", "run_success_rate_health"])
    if success_rate is not None and _safe_float(success_rate, default=1.0) < 0.5:
        return True
    return False


def _route_score(row: pd.Series, *, candidate_score: float, agreement: float | None, risk_score: float, config: Phase3RouterConfig) -> float:
    allocation_weight = _safe_float(row.get("allocation_weight"), default=0.0)
    agreement_multiplier = agreement if agreement is not None else 1.0
    risk_multiplier = 1.0
    if float(config.max_risk_score) > 0.0:
        risk_multiplier = max(0.0, 1.0 - min(1.0, risk_score / float(config.max_risk_score)))
    return float(max(0.0, allocation_weight * max(0.0, candidate_score) * max(0.0, agreement_multiplier) * risk_multiplier))


def _route_allocation_rows(allocation: pd.DataFrame, config: Phase3RouterConfig) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    eligible = allocation.copy()
    if "decision_label" in eligible.columns:
        eligible = eligible[eligible["decision_label"].astype(str).str.lower().eq("allocation_candidate")].copy()
    if "ticker" in eligible.columns:
        eligible = eligible[~eligible["ticker"].astype(str).str.upper().eq("CASH")].copy()

    if eligible.empty:
        return _single_route(
            label="no_candidate",
            reason="no_allocator_candidates",
            config=config,
        )

    sort_columns = [column for column in ["ticker", "allocation_id", "packet_id", "timestamp"] if column in eligible.columns]
    if sort_columns:
        eligible = eligible.sort_values(sort_columns).reset_index(drop=True)

    for index, row in eligible.iterrows():
        allocation_weight = _safe_float(row.get("allocation_weight"), default=0.0)
        candidate_score = _candidate_score(row)
        agreement = _agreement_score(row)
        risk_score = _risk_score(row)
        health_status = _health_status(row)
        route_label = "route_allocation_candidate"
        reason = ""

        if "ticker" not in eligible.columns or not _safe_str(row.get("ticker")):
            route_label = "rejected_missing_required_data"
            reason = "missing_ticker"
        elif config.require_positive_allocation and allocation_weight <= 0.0:
            route_label = "hold_for_review"
            reason = "non_positive_allocation_weight"
        elif allocation_weight < float(config.min_allocation_weight):
            route_label = "hold_for_review"
            reason = "allocation_weight_below_threshold"
        elif candidate_score < float(config.min_candidate_score):
            route_label = "reject_low_confidence"
            reason = "candidate_score_below_threshold"
        elif agreement is not None and agreement < float(config.min_model_agreement):
            route_label = config.low_agreement_action
            reason = "model_agreement_below_threshold"
        elif risk_score > float(config.max_risk_score) or (config.severe_drawdown_blocks and _severe_drawdown(row)):
            route_label = "reject_high_risk"
            reason = "risk_score_or_drawdown_above_threshold"
        elif config.unhealthy_model_blocks and _unhealthy_model(row):
            route_label = "reject_unhealthy_model"
            reason = "model_health_not_eligible"

        route_score = _route_score(row, candidate_score=candidate_score, agreement=agreement, risk_score=risk_score, config=config)
        route_id = f"route_{index + 1:04d}_{_safe_str(row.get('ticker')).upper() or 'UNKNOWN'}"
        route_row = {
            "route_id": route_id,
            "route_label": route_label,
            "ticker": row.get("ticker"),
            "allocation_id": row.get("allocation_id"),
            "allocation_weight": allocation_weight,
            "candidate_score": candidate_score,
            "route_score": route_score,
            "model_agreement_score": agreement,
            "risk_score": risk_score,
            "health_status": health_status,
            "reason": reason,
            "primary_model_name": row.get("primary_model_name"),
            "horizon": row.get("horizon"),
            "target_type": row.get("target_type"),
            "run_mode": row.get("run_mode"),
            "timestamp": row.get("timestamp"),
            "packet_id": row.get("packet_id"),
            "regime_label": row.get("regime_label", row.get("regime_label_regime")),
            "volatility_bucket": row.get("volatility_bucket", row.get("volatility_bucket_risk")),
            "drawdown_state": row.get("drawdown_state", row.get("drawdown_state_portfolio_risk", row.get("drawdown_state_risk"))),
            "claim_boundary": "diagnostic route decision only; not a trading recommendation",
        }
        rows.append(route_row)
        cards.append(
            {
                "card_type": "Phase3RouteDecisionCard",
                "label": route_label,
                "ticker": row.get("ticker"),
                "allocation_weight": allocation_weight,
                "candidate_score": candidate_score,
                "route_score": route_score,
                "model_agreement_score": agreement,
                "risk_score": risk_score,
                "health_status": health_status,
                "reason": reason,
                "source_allocation_id": row.get("allocation_id"),
                "claim_boundary": "diagnostic route decision only; not a trading recommendation",
            }
        )

    return pd.DataFrame(rows).sort_values(["route_label", "ticker", "route_id"]).reset_index(drop=True), cards


def _single_route(
    *,
    label: str,
    reason: str,
    config: Phase3RouterConfig,
    missing_files: list[str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    row = {
        "route_id": f"route_{label}",
        "route_label": label,
        "ticker": "CASH" if label in {"no_candidate", "reject_allocator_no_allocation"} else "",
        "allocation_id": "",
        "allocation_weight": 0.0,
        "candidate_score": 0.0,
        "route_score": 0.0,
        "model_agreement_score": np.nan,
        "risk_score": 0.0,
        "health_status": "",
        "reason": reason,
        "primary_model_name": "",
        "horizon": np.nan,
        "target_type": "",
        "run_mode": "",
        "timestamp": "",
        "packet_id": "",
        "regime_label": "",
        "volatility_bucket": "",
        "drawdown_state": "",
        "claim_boundary": "diagnostic route decision only; not a trading recommendation",
    }
    if missing_files:
        row["missing_required_files"] = "|".join(sorted(missing_files))
    card = {
        "card_type": "Phase3RouteDecisionCard",
        "label": label,
        "ticker": row["ticker"],
        "allocation_weight": 0.0,
        "candidate_score": 0.0,
        "route_score": 0.0,
        "model_agreement_score": None,
        "risk_score": 0.0,
        "health_status": "",
        "reason": reason,
        "claim_boundary": "diagnostic route decision only; not a trading recommendation",
    }
    if missing_files:
        card["missing_required_files"] = sorted(missing_files)
    return pd.DataFrame([row]), [card]


def _build_summary(route_decision: pd.DataFrame) -> pd.DataFrame:
    if route_decision.empty:
        return pd.DataFrame(
            [
                {
                    "route_label": "no_candidate",
                    "route_count": 0,
                    "allocation_weight_sum": 0.0,
                    "mean_candidate_score": np.nan,
                    "mean_model_agreement_score": np.nan,
                    "mean_risk_score": np.nan,
                    "claim_boundary": "diagnostic route decision only; not a trading recommendation",
                }
            ]
        )
    grouped = (
        route_decision.groupby("route_label", sort=True, dropna=False)
        .agg(
            route_count=("route_id", "count"),
            allocation_weight_sum=("allocation_weight", "sum"),
            mean_candidate_score=("candidate_score", "mean"),
            mean_model_agreement_score=("model_agreement_score", "mean"),
            mean_risk_score=("risk_score", "mean"),
        )
        .reset_index()
    )
    grouped["claim_boundary"] = "diagnostic route decision only; not a trading recommendation"
    return grouped


def _build_manifest(
    *,
    config: Phase3RouterConfig,
    input_status: dict[str, dict[str, Any]],
    route_decision: pd.DataFrame,
    cards: list[dict[str, Any]],
    missing_required_files: list[str],
    reason: str,
) -> dict[str, Any]:
    label_counts = (
        route_decision["route_label"].astype(str).value_counts().sort_index().to_dict()
        if not route_decision.empty and "route_label" in route_decision.columns
        else {}
    )
    return {
        "manifest_type": "phase3_router_v1_manifest",
        "router_version": "v1",
        "deterministic": True,
        "decision_authority": "route_decisions_only",
        "config": config.to_dict(),
        "input_files": input_status,
        "output_files": {filename: filename for filename in PHASE3_ROUTER_OUTPUT_FILES},
        "route_label_counts": {key: int(value) for key, value in sorted(label_counts.items())},
        "source_allocation_count": int(len(route_decision)),
        "route_card_count": int(len(cards)),
        "missing_required_files": sorted(missing_required_files),
        "reason": reason,
        "claim_boundary": "diagnostic route decision only; not a trading recommendation",
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=_json_default, sort_keys=True))
            handle.write("\n")


def _write_outputs(
    output_dir: Path,
    *,
    route_decision: pd.DataFrame,
    cards: list[dict[str, Any]],
    routing_summary: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "route_decision.csv": output_dir / "route_decision.csv",
        "phase3_decision_cards.jsonl": output_dir / "phase3_decision_cards.jsonl",
        "routing_summary.csv": output_dir / "routing_summary.csv",
        "routing_manifest.json": output_dir / "routing_manifest.json",
    }
    route_decision.to_csv(output_paths["route_decision.csv"], index=False)
    _write_jsonl(output_paths["phase3_decision_cards.jsonl"], cards)
    routing_summary.to_csv(output_paths["routing_summary.csv"], index=False)
    output_paths["routing_manifest.json"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return output_paths


def run_phase3_router(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    config: Phase3RouterConfig | None = None,
) -> Phase3RouterResult:
    """Run deterministic Phase 3 routing against saved allocator outputs."""

    resolved_config = config or Phase3RouterConfig()
    source_dir = Path(input_dir)
    destination_dir = Path(output_dir) if output_dir is not None else source_dir
    frames, _jsonl_rows, json_docs, input_status = _load_inputs(source_dir)
    missing_required_files = _missing_required_inputs(input_status)

    if missing_required_files:
        reason = "missing_required_allocator_outputs"
        route_decision, cards = _single_route(
            label="rejected_missing_required_data",
            reason=reason,
            config=resolved_config,
            missing_files=missing_required_files,
        )
    else:
        allocation_raw = frames.get("portfolio_allocation.csv", pd.DataFrame())
        summary = frames.get("portfolio_summary.csv", pd.DataFrame())
        manifest = json_docs.get("allocator_manifest.json", {})

        if allocation_raw.empty or summary.empty or any(column not in allocation_raw.columns for column in CORE_ALLOCATION_COLUMNS):
            reason = "allocator_outputs_missing_required_columns"
            route_decision, cards = _single_route(
                label="rejected_missing_required_data",
                reason=reason,
                config=resolved_config,
            )
            missing_required_files = []
        else:
            no_allocation, no_allocation_reason = _allocator_no_allocation(allocation_raw, summary, manifest)
            if no_allocation:
                label = "no_candidate" if resolved_config.allow_no_allocation else "reject_allocator_no_allocation"
                reason = no_allocation_reason or "allocator_no_allocation"
                route_decision, cards = _single_route(label=label, reason=reason, config=resolved_config)
            else:
                prepared = _prepare_allocation(frames)
                route_decision, cards = _route_allocation_rows(prepared, resolved_config)
                reason = ""

    routing_summary = _build_summary(route_decision)
    manifest = _build_manifest(
        config=resolved_config,
        input_status=input_status,
        route_decision=route_decision,
        cards=cards,
        missing_required_files=missing_required_files,
        reason=reason,
    )
    output_paths = _write_outputs(
        destination_dir,
        route_decision=route_decision,
        cards=cards,
        routing_summary=routing_summary,
        manifest=manifest,
    )
    return Phase3RouterResult(
        route_decision=route_decision,
        decision_cards=cards,
        routing_summary=routing_summary,
        manifest=manifest,
        output_paths=output_paths,
    )
