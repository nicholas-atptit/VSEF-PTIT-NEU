"""Governed benchmark acceptance decisions.

This module turns statistical and economic evidence into explicit benchmark
promotion status. It is intentionally independent of data providers and model
training so tests can exercise the policy directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal


AcceptanceStatus = Literal["accepted", "inconclusive", "exploratory_only", "rejected"]


@dataclass(frozen=True)
class AcceptancePolicy:
    """Conservative defaults for benchmark promotion governance."""

    policy_version: str = "phase6_statistical_acceptance_v1"
    min_effect_size: float = 0.0
    min_economic_metric_delta: float = 0.0
    min_cost_adjusted_delta: float = 0.0
    dm_p_value_threshold: float = 0.05
    min_sample_size: int = 30
    bootstrap_ci_lower_tolerance: float = 0.0
    max_bootstrap_ci_width_multiple: float = 10.0
    max_turnover_delta: float = 1.0
    multiple_comparison_warning_threshold: int = 20
    downgrade_many_comparisons: bool = True


@dataclass(frozen=True)
class AcceptanceResult:
    """Structured acceptance output for benchmark reports."""

    accepted: bool
    status: AcceptanceStatus
    effect_size: float
    bootstrap_ci: tuple[float | None, float | None]
    dm_p_value: float | None
    warnings: list[str] = field(default_factory=list)
    economic_metric_delta: float | None = None
    turnover_penalty: float | None = None
    cost_adjusted_delta: float | None = None
    sample_size: int | None = None
    comparison_count: int | None = None
    policy_version: str = "phase6_statistical_acceptance_v1"
    decision_reasons: list[str] = field(default_factory=list)
    interpretation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable acceptance fields."""
        return {
            "accepted": bool(self.accepted),
            "status": self.status,
            "effect_size": float(self.effect_size),
            "bootstrap_ci": [self.bootstrap_ci[0], self.bootstrap_ci[1]],
            "dm_p_value": self.dm_p_value,
            "warnings": list(self.warnings),
            "economic_metric_delta": self.economic_metric_delta,
            "turnover_penalty": self.turnover_penalty,
            "cost_adjusted_delta": self.cost_adjusted_delta,
            "sample_size": self.sample_size,
            "comparison_count": self.comparison_count,
            "policy_version": self.policy_version,
            "decision_reasons": list(self.decision_reasons),
            "interpretation": self.interpretation,
        }


def evaluate_benchmark_acceptance(
    *,
    prediction_metric_delta: float | None,
    economic_metric_delta: float | None,
    bootstrap_ci: tuple[float, float] | list[float] | None,
    dm_p_value: float | None,
    turnover_delta: float | None = None,
    cost_adjusted_delta: float | None = None,
    sample_size: int | None = None,
    comparison_count: int | None = None,
    policy: AcceptancePolicy | None = None,
) -> AcceptanceResult:
    """Classify whether benchmark evidence supports promotion.

    Missing evidence is never imputed. Any missing core statistical/economic
    evidence prevents acceptance and yields ``exploratory_only`` unless the
    available evidence already rejects the benchmark claim.
    """
    resolved = policy or AcceptancePolicy()
    effect_size = _safe_float(prediction_metric_delta)
    economic_delta = _safe_float(economic_metric_delta)
    ci = _normalize_ci(bootstrap_ci)
    p_value = _safe_float(dm_p_value)
    turnover_penalty = _safe_float(turnover_delta)
    cost_delta = _safe_float(cost_adjusted_delta)
    n = _safe_int(sample_size)
    comparisons = _safe_int(comparison_count)

    warnings: list[str] = []
    reasons: list[str] = []

    missing = []
    if effect_size is None:
        missing.append("prediction_metric_delta")
    if economic_delta is None:
        missing.append("economic_metric_delta")
    if ci[0] is None or ci[1] is None:
        missing.append("bootstrap_ci")
    if p_value is None:
        missing.append("dm_p_value")
    if n is None:
        missing.append("sample_size")
    if missing:
        warnings.append("missing_evidence:" + ",".join(missing))

    if comparisons is not None and comparisons > resolved.multiple_comparison_warning_threshold:
        warnings.append(
            "many_unadjusted_comparisons:"
            f"{comparisons}>{resolved.multiple_comparison_warning_threshold}"
        )

    if n is not None and n < resolved.min_sample_size:
        warnings.append(f"small_sample_size:{n}<{resolved.min_sample_size}")

    rejection_reasons = _rejection_reasons(
        effect_size=effect_size,
        economic_delta=economic_delta,
        turnover_penalty=turnover_penalty,
        cost_delta=cost_delta,
        policy=resolved,
    )
    if rejection_reasons:
        reasons.extend(rejection_reasons)
        return _result(
            status="rejected",
            effect_size=effect_size,
            bootstrap_ci=ci,
            dm_p_value=p_value,
            warnings=warnings,
            economic_metric_delta=economic_delta,
            turnover_penalty=turnover_penalty,
            cost_adjusted_delta=cost_delta,
            sample_size=n,
            comparison_count=comparisons,
            policy=resolved,
            decision_reasons=reasons,
        )

    if missing:
        reasons.append("core_statistical_or_economic_evidence_missing")
        return _result(
            status="exploratory_only",
            effect_size=effect_size,
            bootstrap_ci=ci,
            dm_p_value=p_value,
            warnings=warnings,
            economic_metric_delta=economic_delta,
            turnover_penalty=turnover_penalty,
            cost_adjusted_delta=cost_delta,
            sample_size=n,
            comparison_count=comparisons,
            policy=resolved,
            decision_reasons=reasons,
        )

    mixed_reasons = _mixed_evidence_reasons(
        ci=ci,
        p_value=p_value,
        sample_size=n,
        comparison_count=comparisons,
        policy=resolved,
    )
    if mixed_reasons:
        reasons.extend(mixed_reasons)
        return _result(
            status="inconclusive",
            effect_size=effect_size,
            bootstrap_ci=ci,
            dm_p_value=p_value,
            warnings=warnings,
            economic_metric_delta=economic_delta,
            turnover_penalty=turnover_penalty,
            cost_adjusted_delta=cost_delta,
            sample_size=n,
            comparison_count=comparisons,
            policy=resolved,
            decision_reasons=reasons,
        )

    reasons.append("all_acceptance_gates_passed")
    return _result(
        status="accepted",
        effect_size=effect_size,
        bootstrap_ci=ci,
        dm_p_value=p_value,
        warnings=warnings,
        economic_metric_delta=economic_delta,
        turnover_penalty=turnover_penalty,
        cost_adjusted_delta=cost_delta,
        sample_size=n,
        comparison_count=comparisons,
        policy=resolved,
        decision_reasons=reasons,
    )


def _rejection_reasons(
    *,
    effect_size: float | None,
    economic_delta: float | None,
    turnover_penalty: float | None,
    cost_delta: float | None,
    policy: AcceptancePolicy,
) -> list[str]:
    reasons: list[str] = []
    if effect_size is not None and effect_size <= policy.min_effect_size:
        reasons.append(f"effect_size_not_positive:{effect_size:g}<={policy.min_effect_size:g}")
    if economic_delta is not None and economic_delta < policy.min_economic_metric_delta:
        reasons.append(f"economic_metric_delta_below_threshold:{economic_delta:g}<{policy.min_economic_metric_delta:g}")
    if cost_delta is not None and cost_delta < policy.min_cost_adjusted_delta:
        reasons.append(f"cost_adjusted_delta_below_threshold:{cost_delta:g}<{policy.min_cost_adjusted_delta:g}")
    if turnover_penalty is not None and turnover_penalty > policy.max_turnover_delta:
        reasons.append(f"turnover_penalty_exceeds_threshold:{turnover_penalty:g}>{policy.max_turnover_delta:g}")
    return reasons


def _mixed_evidence_reasons(
    *,
    ci: tuple[float | None, float | None],
    p_value: float | None,
    sample_size: int | None,
    comparison_count: int | None,
    policy: AcceptancePolicy,
) -> list[str]:
    reasons: list[str] = []
    lower, upper = ci
    if upper is not None and upper <= 0.0:
        reasons.append(f"bootstrap_ci_does_not_support_positive_effect:{upper:g}<=0")
    if lower is not None and lower < -abs(policy.bootstrap_ci_lower_tolerance):
        reasons.append(
            "bootstrap_ci_lower_below_tolerance:"
            f"{lower:g}<-{abs(policy.bootstrap_ci_lower_tolerance):g}"
        )
    if lower is not None and upper is not None:
        width = upper - lower
        max_width = max(abs(upper), abs(lower), 1e-12) * policy.max_bootstrap_ci_width_multiple
        if width > max_width:
            reasons.append(f"bootstrap_ci_severely_wide:{width:g}>{max_width:g}")
    if p_value is not None and p_value >= policy.dm_p_value_threshold:
        reasons.append(f"dm_p_value_not_significant:{p_value:g}>={policy.dm_p_value_threshold:g}")
    if sample_size is not None and sample_size < policy.min_sample_size:
        reasons.append(f"sample_size_below_threshold:{sample_size}<{policy.min_sample_size}")
    if (
        policy.downgrade_many_comparisons
        and comparison_count is not None
        and comparison_count > policy.multiple_comparison_warning_threshold
    ):
        reasons.append(
            "multiple_comparison_inflation_unadjusted:"
            f"{comparison_count}>{policy.multiple_comparison_warning_threshold}"
        )
    return reasons


def _result(
    *,
    status: AcceptanceStatus,
    effect_size: float | None,
    bootstrap_ci: tuple[float | None, float | None],
    dm_p_value: float | None,
    warnings: list[str],
    economic_metric_delta: float | None,
    turnover_penalty: float | None,
    cost_adjusted_delta: float | None,
    sample_size: int | None,
    comparison_count: int | None,
    policy: AcceptancePolicy,
    decision_reasons: list[str],
) -> AcceptanceResult:
    accepted = status == "accepted"
    return AcceptanceResult(
        accepted=accepted,
        status=status,
        effect_size=0.0 if effect_size is None else float(effect_size),
        bootstrap_ci=bootstrap_ci,
        dm_p_value=dm_p_value,
        warnings=list(dict.fromkeys(warnings)),
        economic_metric_delta=economic_metric_delta,
        turnover_penalty=turnover_penalty,
        cost_adjusted_delta=cost_adjusted_delta,
        sample_size=sample_size,
        comparison_count=comparison_count,
        policy_version=policy.policy_version,
        decision_reasons=list(dict.fromkeys(decision_reasons)),
        interpretation=_interpret_status(status, decision_reasons),
    )


def _interpret_status(status: AcceptanceStatus, reasons: list[str]) -> str:
    if status == "accepted":
        return "Governed benchmark promotion is supported by the supplied evidence."
    if status == "rejected":
        return "Benchmark promotion is rejected because available evidence fails core acceptance gates."
    if status == "inconclusive":
        return "Benchmark promotion is not supported because evidence is mixed or unstable."
    if reasons and "core_statistical_or_economic_evidence_missing" in reasons:
        return "Benchmark output is exploratory only because required statistical or economic evidence is missing."
    return "Benchmark output is exploratory only and must not be promoted as a strong claim."


def _normalize_ci(value: tuple[float, float] | list[float] | None) -> tuple[float | None, float | None]:
    if value is None or len(value) != 2:
        return (None, None)
    lower = _safe_float(value[0])
    upper = _safe_float(value[1])
    if lower is not None and upper is not None and lower > upper:
        return (None, None)
    return (lower, upper)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None
