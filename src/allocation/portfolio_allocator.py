"""Deterministic Portfolio Allocator v1 for Quant Core diagnostic outputs.

This allocator emits allocation candidates only. It does not emit final trading
recommendations and does not train or select models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INPUT_FILES: tuple[str, ...] = (
    "decision_lane_candidates.csv",
    "model_consensus_summary.csv",
    "model_health_summary.csv",
    "risk_summary.csv",
    "strategy_metrics.csv",
    "regime_summary.csv",
)

ALLOCATION_OUTPUT_FILES: tuple[str, ...] = (
    "portfolio_allocation.csv",
    "portfolio_summary.csv",
    "portfolio_risk_summary.csv",
    "portfolio_decision_cards.jsonl",
    "allocator_manifest.json",
)

ALLOCATION_LABELS: tuple[str, ...] = (
    "allocation_candidate",
    "no_allocation",
    "rejected_low_confidence",
    "rejected_high_risk",
    "rejected_low_agreement",
    "rejected_unhealthy_model",
    "rejected_missing_required_data",
)

CORE_CANDIDATE_COLUMNS: tuple[str, ...] = ("ticker",)


@dataclass(frozen=True)
class PortfolioAllocatorConfig:
    """Conservative deterministic allocation settings."""

    max_ticker_weight: float = 0.10
    max_total_exposure: float = 0.60
    cash_buffer: float = 0.40
    min_candidate_score: float = 0.0
    min_model_agreement: float = 0.5
    max_risk_score: float = 1.0
    risk_penalty_strength: float = 0.5
    agreement_penalty_strength: float = 0.5
    allow_short: bool = False
    label: str = "allocation_candidate"

    def __post_init__(self) -> None:
        if self.label != "allocation_candidate":
            raise ValueError("Portfolio Allocator v1 only supports allocation_candidate labels")
        for name in ("max_ticker_weight", "max_total_exposure", "cash_buffer"):
            value = float(getattr(self, name))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("risk_penalty_strength", "agreement_penalty_strength"):
            value = float(getattr(self, name))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

    @property
    def effective_max_exposure(self) -> float:
        return max(0.0, min(float(self.max_total_exposure), 1.0 - float(self.cash_buffer)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"effective_max_exposure": self.effective_max_exposure}


@dataclass(frozen=True)
class PortfolioAllocatorResult:
    allocation: pd.DataFrame
    summary: pd.DataFrame
    risk_summary: pd.DataFrame
    decision_cards: list[dict[str, Any]]
    manifest: dict[str, Any]
    output_paths: dict[str, Path]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
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


def _shared_columns(left: pd.DataFrame, right: pd.DataFrame, preferred: list[str]) -> list[str]:
    return [column for column in preferred if column in left.columns and column in right.columns]


def _dedupe_for_merge(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty or not keys:
        return pd.DataFrame()
    return frame.sort_values(keys).drop_duplicates(keys, keep="first").reset_index(drop=True)


def _merge_optional(
    candidates: pd.DataFrame,
    frame: pd.DataFrame,
    preferred_keys: list[str],
    *,
    suffix: str,
) -> pd.DataFrame:
    if candidates.empty or frame.empty:
        return candidates
    keys = _shared_columns(candidates, frame, preferred_keys)
    if not keys:
        return candidates
    right = _dedupe_for_merge(frame, keys)
    return candidates.merge(right, on=keys, how="left", suffixes=("", suffix))


def _merge_health(candidates: pd.DataFrame, health: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty or health.empty or "primary_model_name" not in candidates.columns or "model_name" not in health.columns:
        if "health_status" not in candidates.columns:
            candidates = candidates.copy()
            candidates["health_status"] = ""
        return candidates
    right = health.sort_values("model_name").drop_duplicates("model_name", keep="first").copy()
    right = right.rename(columns={column: f"{column}_health" for column in right.columns if column != "model_name"})
    merged = candidates.merge(right, left_on="primary_model_name", right_on="model_name", how="left")
    if "health_status" not in merged.columns:
        merged["health_status"] = merged.get("health_status_health", "")
    return merged


def _merge_strategy(candidates: pd.DataFrame, strategy: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty or strategy.empty:
        return candidates
    strategy_frame = strategy.copy()
    if "model_name" in strategy_frame.columns and "primary_model_name" in candidates.columns:
        strategy_frame = strategy_frame.rename(columns={"model_name": "primary_model_name"})
    keys = _shared_columns(
        candidates,
        strategy_frame,
        ["primary_model_name", "horizon", "target_type", "run_mode", "group_name"],
    )
    if not keys:
        return candidates
    keep = [
        *keys,
        *[
            column
            for column in ["sharpe", "cagr", "max_drawdown", "total_return", "win_rate", "trade_count", "policy_variant"]
            if column in strategy_frame.columns
        ],
    ]
    right = _dedupe_for_merge(strategy_frame[keep], keys)
    return candidates.merge(right, on=keys, how="left", suffixes=("", "_strategy"))


def _load_inputs(input_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    frames: dict[str, pd.DataFrame] = {}
    status: dict[str, dict[str, Any]] = {}
    for filename in INPUT_FILES:
        path = input_dir / filename
        frame = _read_csv(path)
        frames[filename] = frame
        status[filename] = {
            "path": str(path),
            "exists": bool(path.exists()),
            "rows": int(len(frame)),
            "columns": list(frame.columns),
        }
    return frames, status


def _prepare_candidates(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    candidates = frames.get("decision_lane_candidates.csv", pd.DataFrame()).copy()
    if candidates.empty:
        return candidates
    candidates = _merge_optional(
        candidates,
        frames.get("model_consensus_summary.csv", pd.DataFrame()),
        ["timestamp", "ticker", "horizon", "target_type", "run_mode", "core_run_id", "group_name"],
        suffix="_consensus",
    )
    candidates = _merge_optional(
        candidates,
        frames.get("risk_summary.csv", pd.DataFrame()),
        ["timestamp", "ticker", "horizon", "target_type", "run_mode", "core_run_id", "group_name"],
        suffix="_risk",
    )
    candidates = _merge_optional(
        candidates,
        frames.get("regime_summary.csv", pd.DataFrame()),
        ["timestamp", "ticker", "horizon", "target_type", "run_mode", "core_run_id", "group_name"],
        suffix="_regime",
    )
    candidates = _merge_health(candidates, frames.get("model_health_summary.csv", pd.DataFrame()))
    candidates = _merge_strategy(candidates, frames.get("strategy_metrics.csv", pd.DataFrame()))
    return candidates.sort_values([column for column in ["ticker", "timestamp", "packet_id"] if column in candidates.columns]).reset_index(drop=True)


def _candidate_score(row: pd.Series) -> float:
    if "candidate_score" in row and pd.notna(row.get("candidate_score")):
        return _safe_float(row.get("candidate_score"))
    prediction = _safe_float(row.get("primary_prediction"))
    agreement = _safe_float(row.get("model_agreement_score", row.get("agreement_score", 1.0)), default=1.0)
    return prediction * agreement


def _agreement_score(row: pd.Series) -> float | None:
    for column in ("model_agreement_score", "agreement_score"):
        if column in row and pd.notna(row.get(column)):
            return _safe_float(row.get(column))
    return None


def _risk_score(row: pd.Series) -> tuple[float, bool]:
    numeric_columns = ["risk_score", "vol_forecast", "volatility", "var_loss_95", "cvar_loss_95"]
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
        for column in ("current_drawdown", "max_drawdown")
        if column in row and pd.notna(row.get(column))
    ]
    values.extend(value for value in drawdown_values if np.isfinite(value))
    score = max(values) if values else 0.0

    drawdown_state = _safe_str(row.get("drawdown_state")).lower()
    if drawdown_state == "severe":
        score = max(score, 1.0)
    elif drawdown_state == "elevated":
        score = max(score, 0.5)

    volatility_bucket = _safe_str(row.get("volatility_bucket")).lower()
    if volatility_bucket == "high":
        score = max(score, 1.0)
    elif volatility_bucket == "medium":
        score = max(score, 0.5)

    risk_model_text = " ".join(_safe_str(row.get(column)).lower() for column in ("risk_model", "source_model"))
    fallback_used = "fallback" in risk_model_text
    return float(score), bool(fallback_used)


def _is_unhealthy(row: pd.Series) -> bool:
    status = _safe_str(row.get("health_status", row.get("health_status_health"))).lower()
    success_rate = row.get("run_success_rate", row.get("run_success_rate_health"))
    if status in {"failing", "weak"}:
        return True
    if pd.notna(success_rate) and _safe_float(success_rate, default=1.0) < 0.5:
        return True
    return False


def _evaluate_candidates(candidates: pd.DataFrame, config: PortfolioAllocatorConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    if candidates.empty:
        return pd.DataFrame(), {}

    rows: list[dict[str, Any]] = []
    rejection_counts = {label: 0 for label in ALLOCATION_LABELS if label.startswith("rejected_")}
    for row_index, row in candidates.iterrows():
        score = _candidate_score(row)
        agreement = _agreement_score(row)
        risk_score, risk_fallback_used = _risk_score(row)
        label = "allocation_candidate"
        reason = ""

        if "ticker" not in candidates.columns or not _safe_str(row.get("ticker")):
            label = "rejected_missing_required_data"
            reason = "missing_ticker"
        elif not config.allow_short and _safe_float(row.get("primary_prediction"), default=score) < 0.0:
            label = "rejected_low_confidence"
            reason = "short_candidate_blocked"
        elif score < float(config.min_candidate_score):
            label = "rejected_low_confidence"
            reason = "candidate_score_below_threshold"
        elif agreement is not None and agreement < float(config.min_model_agreement):
            label = "rejected_low_agreement"
            reason = "model_agreement_below_threshold"
        elif risk_score > float(config.max_risk_score) or _safe_str(row.get("drawdown_state")).lower() == "severe":
            label = "rejected_high_risk"
            reason = "risk_score_above_threshold"
        elif _is_unhealthy(row):
            label = "rejected_unhealthy_model"
            reason = "model_health_not_eligible"

        agreement_multiplier = 1.0
        if agreement is not None:
            agreement_multiplier = max(0.0, 1.0 - (float(config.agreement_penalty_strength) * max(0.0, 1.0 - agreement)))
        risk_penalty = 0.0
        if float(config.max_risk_score) > 0.0:
            risk_penalty = min(1.0, max(0.0, risk_score / float(config.max_risk_score))) * float(config.risk_penalty_strength)
        fallback_penalty = 0.25 * float(config.risk_penalty_strength) if risk_fallback_used else 0.0
        adjusted_score = max(0.0, score * agreement_multiplier * max(0.0, 1.0 - risk_penalty) * max(0.0, 1.0 - fallback_penalty))

        output = row.to_dict()
        output.update(
            {
                "candidate_row_id": int(row_index),
                "base_candidate_score": float(score),
                "agreement_for_allocation": agreement,
                "risk_score": float(risk_score),
                "risk_fallback_used": bool(risk_fallback_used),
                "risk_penalty": float(risk_penalty + fallback_penalty),
                "agreement_penalty": float(1.0 - agreement_multiplier),
                "adjusted_score": float(adjusted_score),
                "allocator_label": label,
                "rejection_reason": reason,
            }
        )
        if label != "allocation_candidate":
            rejection_counts[label] += 1
        rows.append(output)

    return pd.DataFrame(rows), rejection_counts


def _no_allocation_outputs(
    *,
    config: PortfolioAllocatorConfig,
    input_status: dict[str, dict[str, Any]],
    reason: str,
    rejection_counts: dict[str, int] | None = None,
    source_candidate_count: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    rejection_counts = dict(rejection_counts or {})
    if reason == "missing_required_candidate_file":
        rejection_counts["rejected_missing_required_data"] = max(1, rejection_counts.get("rejected_missing_required_data", 0))
    allocation = pd.DataFrame(
        [
            {
                "allocation_id": "cash_no_allocation",
                "decision_label": "no_allocation",
                "ticker": "CASH",
                "allocation_weight": 1.0,
                "invested_weight": 0.0,
                "cash_weight": 1.0,
                "source_candidate_count": int(source_candidate_count),
                "accepted_candidate_count": 0,
                "reason": reason,
            }
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "portfolio_label": "no_allocation",
                "allocation_count": 0,
                "source_candidate_count": int(source_candidate_count),
                "accepted_candidate_count": 0,
                "rejected_candidate_count": int(sum(rejection_counts.values())),
                "invested_exposure": 0.0,
                "cash_weight": 1.0,
                "max_ticker_weight": float(config.max_ticker_weight),
                "max_total_exposure": float(config.max_total_exposure),
                "cash_buffer": float(config.cash_buffer),
                "no_allocation_reason": reason,
            }
        ]
    )
    risk_summary = pd.DataFrame(
        [
            {
                "portfolio_label": "no_allocation",
                "ticker": "CASH",
                "allocation_weight": 1.0,
                "portfolio_risk_score": 0.0,
                "weighted_risk_contribution": 0.0,
                "reason": reason,
            }
        ]
    )
    cards = [
        {
            "card_type": "PortfolioAllocationCard",
            "label": "no_allocation",
            "ticker": "CASH",
            "allocation_weight": 1.0,
            "cash_weight": 1.0,
            "reason": reason,
            "claim_boundary": "allocation diagnostic only; not a buy recommendation",
        }
    ]
    manifest = _build_manifest(
        config=config,
        input_status=input_status,
        label_counts={"no_allocation": 1},
        rejection_counts=rejection_counts,
        source_candidate_count=source_candidate_count,
        accepted_candidate_count=0,
        no_allocation_reason=reason,
    )
    return allocation, summary, risk_summary, cards, manifest


def _build_allocations(
    evaluated: pd.DataFrame,
    *,
    config: PortfolioAllocatorConfig,
    input_status: dict[str, dict[str, Any]],
    rejection_counts: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    survivors = evaluated[evaluated["allocator_label"].astype(str) == "allocation_candidate"].copy()
    if survivors.empty:
        return _no_allocation_outputs(
            config=config,
            input_status=input_status,
            reason="all_candidates_rejected" if len(evaluated) else "no_candidates",
            rejection_counts=rejection_counts,
            source_candidate_count=int(len(evaluated)),
        )

    survivors = survivors.sort_values(
        ["adjusted_score", "ticker", *[column for column in ["packet_id", "timestamp"] if column in survivors.columns]],
        ascending=[False, True, *([True] * len([column for column in ["packet_id", "timestamp"] if column in survivors.columns]))],
    ).reset_index(drop=True)

    grouped_rows: list[dict[str, Any]] = []
    for ticker, group in survivors.groupby("ticker", sort=True, dropna=False):
        ordered = group.sort_values(["adjusted_score", "candidate_row_id"], ascending=[False, True]).reset_index(drop=True)
        top = ordered.iloc[0].to_dict()
        top["ticker"] = ticker
        top["ticker_adjusted_score"] = float(pd.to_numeric(group["adjusted_score"], errors="coerce").fillna(0.0).sum())
        top["accepted_candidate_count"] = int(len(group))
        grouped_rows.append(top)
    grouped = pd.DataFrame(grouped_rows).sort_values(["ticker_adjusted_score", "ticker"], ascending=[False, True]).reset_index(drop=True)

    score_sum = float(pd.to_numeric(grouped["ticker_adjusted_score"], errors="coerce").fillna(0.0).sum())
    if score_sum <= 0.0:
        return _no_allocation_outputs(
            config=config,
            input_status=input_status,
            reason="non_positive_adjusted_scores",
            rejection_counts=rejection_counts,
            source_candidate_count=int(len(evaluated)),
        )

    exposure_target = config.effective_max_exposure
    cap = min(float(config.max_ticker_weight), exposure_target) if exposure_target > 0.0 else 0.0
    grouped["raw_weight"] = grouped["ticker_adjusted_score"] / score_sum * exposure_target
    grouped["allocation_weight"] = grouped["raw_weight"].clip(lower=0.0, upper=cap)
    invested_exposure = float(grouped["allocation_weight"].sum())
    cash_weight = max(float(config.cash_buffer), 1.0 - invested_exposure)

    allocation_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    for index, row in grouped.iterrows():
        allocation_id = f"alloc_{index + 1:04d}_{_safe_str(row.get('ticker')).upper()}"
        allocation_row = {
            "allocation_id": allocation_id,
            "decision_label": "allocation_candidate",
            "ticker": row.get("ticker"),
            "allocation_weight": float(row.get("allocation_weight", 0.0)),
            "invested_weight": float(row.get("allocation_weight", 0.0)),
            "cash_weight": cash_weight,
            "raw_weight": float(row.get("raw_weight", 0.0)),
            "max_ticker_weight": float(config.max_ticker_weight),
            "max_total_exposure": float(config.max_total_exposure),
            "candidate_score": float(row.get("base_candidate_score", 0.0)),
            "adjusted_score": float(row.get("ticker_adjusted_score", 0.0)),
            "model_agreement_score": row.get("agreement_for_allocation"),
            "risk_score": float(row.get("risk_score", 0.0)),
            "risk_penalty": float(row.get("risk_penalty", 0.0)),
            "agreement_penalty": float(row.get("agreement_penalty", 0.0)),
            "primary_model_name": row.get("primary_model_name"),
            "health_status": row.get("health_status", row.get("health_status_health", "")),
            "horizon": row.get("horizon"),
            "target_type": row.get("target_type"),
            "run_mode": row.get("run_mode"),
            "timestamp": row.get("timestamp"),
            "packet_id": row.get("packet_id"),
            "regime_label": row.get("regime_label"),
            "volatility_bucket": row.get("volatility_bucket"),
            "accepted_candidate_count": int(row.get("accepted_candidate_count", 1)),
            "source_candidate_count": int(len(evaluated)),
            "claim_boundary": "allocation diagnostic only; not a buy recommendation",
        }
        allocation_rows.append(allocation_row)
        weighted_risk = float(allocation_row["allocation_weight"]) * float(allocation_row["risk_score"])
        risk_rows.append(
            {
                "portfolio_label": "allocation_candidate",
                "ticker": row.get("ticker"),
                "allocation_weight": float(allocation_row["allocation_weight"]),
                "risk_score": float(allocation_row["risk_score"]),
                "weighted_risk_contribution": weighted_risk,
                "risk_fallback_used": bool(row.get("risk_fallback_used", False)),
                "drawdown_state": row.get("drawdown_state"),
                "volatility_bucket": row.get("volatility_bucket"),
                "risk_model": row.get("risk_model", row.get("source_model")),
            }
        )
        cards.append(
            {
                "card_type": "PortfolioAllocationCard",
                "label": "allocation_candidate",
                "ticker": row.get("ticker"),
                "allocation_weight": float(allocation_row["allocation_weight"]),
                "cash_weight": cash_weight,
                "candidate_score": float(row.get("base_candidate_score", 0.0)),
                "adjusted_score": float(row.get("ticker_adjusted_score", 0.0)),
                "model_agreement_score": row.get("agreement_for_allocation"),
                "risk_score": float(row.get("risk_score", 0.0)),
                "primary_model_name": row.get("primary_model_name"),
                "horizon": row.get("horizon"),
                "target_type": row.get("target_type"),
                "run_mode": row.get("run_mode"),
                "packet_id": row.get("packet_id"),
                "claim_boundary": "allocation diagnostic only; not a buy recommendation",
            }
        )

    allocation = pd.DataFrame(allocation_rows)
    risk_summary = pd.DataFrame(risk_rows)
    portfolio_risk_score = float(risk_summary["weighted_risk_contribution"].sum()) if not risk_summary.empty else 0.0
    summary = pd.DataFrame(
        [
            {
                "portfolio_label": "allocation_candidate",
                "allocation_count": int(len(allocation)),
                "source_candidate_count": int(len(evaluated)),
                "accepted_candidate_count": int(len(survivors)),
                "rejected_candidate_count": int(sum(rejection_counts.values())),
                "invested_exposure": invested_exposure,
                "cash_weight": cash_weight,
                "max_ticker_weight": float(config.max_ticker_weight),
                "max_total_exposure": float(config.max_total_exposure),
                "cash_buffer": float(config.cash_buffer),
                "portfolio_risk_score": portfolio_risk_score,
                "no_allocation_reason": "",
            }
        ]
    )
    manifest = _build_manifest(
        config=config,
        input_status=input_status,
        label_counts={"allocation_candidate": int(len(allocation))},
        rejection_counts=rejection_counts,
        source_candidate_count=int(len(evaluated)),
        accepted_candidate_count=int(len(survivors)),
        no_allocation_reason="",
    )
    return allocation, summary, risk_summary, cards, manifest


def _build_manifest(
    *,
    config: PortfolioAllocatorConfig,
    input_status: dict[str, dict[str, Any]],
    label_counts: dict[str, int],
    rejection_counts: dict[str, int],
    source_candidate_count: int,
    accepted_candidate_count: int,
    no_allocation_reason: str,
) -> dict[str, Any]:
    return {
        "manifest_type": "portfolio_allocator_v1_manifest",
        "allocator_version": "v1",
        "deterministic": True,
        "decision_authority": "allocation_candidates_only",
        "config": config.to_dict(),
        "input_files": input_status,
        "output_files": {filename: filename for filename in ALLOCATION_OUTPUT_FILES},
        "label_counts": {key: int(value) for key, value in sorted(label_counts.items())},
        "rejection_counts": {key: int(value) for key, value in sorted(rejection_counts.items()) if int(value) > 0},
        "source_candidate_count": int(source_candidate_count),
        "accepted_candidate_count": int(accepted_candidate_count),
        "no_allocation_reason": no_allocation_reason,
        "claim_boundary": "allocation diagnostic only; not a buy recommendation",
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=_json_default, sort_keys=True))
            handle.write("\n")


def _write_outputs(
    output_dir: Path,
    *,
    allocation: pd.DataFrame,
    summary: pd.DataFrame,
    risk_summary: pd.DataFrame,
    cards: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "portfolio_allocation.csv": output_dir / "portfolio_allocation.csv",
        "portfolio_summary.csv": output_dir / "portfolio_summary.csv",
        "portfolio_risk_summary.csv": output_dir / "portfolio_risk_summary.csv",
        "portfolio_decision_cards.jsonl": output_dir / "portfolio_decision_cards.jsonl",
        "allocator_manifest.json": output_dir / "allocator_manifest.json",
    }
    allocation.to_csv(output_paths["portfolio_allocation.csv"], index=False)
    summary.to_csv(output_paths["portfolio_summary.csv"], index=False)
    risk_summary.to_csv(output_paths["portfolio_risk_summary.csv"], index=False)
    _write_jsonl(output_paths["portfolio_decision_cards.jsonl"], cards)
    output_paths["allocator_manifest.json"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return output_paths


def run_portfolio_allocator(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    config: PortfolioAllocatorConfig | None = None,
) -> PortfolioAllocatorResult:
    """Run Portfolio Allocator v1 against a Quant Core output directory."""

    resolved_config = config or PortfolioAllocatorConfig()
    source_dir = Path(input_dir)
    destination_dir = Path(output_dir) if output_dir is not None else source_dir
    frames, input_status = _load_inputs(source_dir)
    candidate_file = source_dir / "decision_lane_candidates.csv"
    candidates_raw = frames.get("decision_lane_candidates.csv", pd.DataFrame())

    if not candidate_file.exists():
        allocation, summary, risk_summary, cards, manifest = _no_allocation_outputs(
            config=resolved_config,
            input_status=input_status,
            reason="missing_required_candidate_file",
            source_candidate_count=0,
        )
    elif candidates_raw.empty:
        allocation, summary, risk_summary, cards, manifest = _no_allocation_outputs(
            config=resolved_config,
            input_status=input_status,
            reason="no_candidates",
            source_candidate_count=0,
        )
    elif any(column not in candidates_raw.columns for column in CORE_CANDIDATE_COLUMNS):
        allocation, summary, risk_summary, cards, manifest = _no_allocation_outputs(
            config=resolved_config,
            input_status=input_status,
            reason="candidate_file_missing_required_columns",
            source_candidate_count=int(len(candidates_raw)),
        )
    else:
        candidates = _prepare_candidates(frames)
        evaluated, rejection_counts = _evaluate_candidates(candidates, resolved_config)
        allocation, summary, risk_summary, cards, manifest = _build_allocations(
            evaluated,
            config=resolved_config,
            input_status=input_status,
            rejection_counts=rejection_counts,
        )

    output_paths = _write_outputs(
        destination_dir,
        allocation=allocation,
        summary=summary,
        risk_summary=risk_summary,
        cards=cards,
        manifest=manifest,
    )
    return PortfolioAllocatorResult(
        allocation=allocation,
        summary=summary,
        risk_summary=risk_summary,
        decision_cards=cards,
        manifest=manifest,
        output_paths=output_paths,
    )
