"""Risk-aware portfolio allocation with modular penalty controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AllocationResult:
    weights: pd.Series
    cash_weight: float
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": {str(k): float(v) for k, v in self.weights.items()},
            "cash_weight": float(self.cash_weight),
            "diagnostics": self.diagnostics,
        }


class RiskAwareAllocator:
    """Apply risk and regime penalties to a baseline allocation."""

    def __init__(
        self,
        *,
        risk_penalty_strength: float = 1.0,
        regime_multipliers: dict[str, float] | None = None,
        high_vol_exposure_cut: float | None = None,
        crisis_exposure_cut: float | None = None,
    ) -> None:
        self.risk_penalty_strength = max(float(risk_penalty_strength), 0.0)
        self.regime_multipliers = regime_multipliers or {
            "NORMAL": 1.0,
            "HIGH_VOL": float(0.6 if high_vol_exposure_cut is None else high_vol_exposure_cut),
            "CRISIS": float(0.25 if crisis_exposure_cut is None else crisis_exposure_cut),
        }

    @staticmethod
    def _normalize_base_weights(base_weights: pd.Series | None, index: pd.Index) -> pd.Series:
        if base_weights is None or base_weights.empty:
            values = pd.Series(1.0 / max(len(index), 1), index=index)
        else:
            values = pd.to_numeric(base_weights.reindex(index), errors="coerce").fillna(0.0)
            total = float(values.sum())
            if total > 0:
                values = values / total
        return values

    def allocate(
        self,
        *,
        risk_frame: pd.DataFrame,
        regime_labels: pd.Series | None = None,
        base_weights: pd.Series | None = None,
    ) -> AllocationResult:
        if risk_frame.empty:
            empty = pd.Series(dtype=float)
            return AllocationResult(weights=empty, cash_weight=1.0, diagnostics={})

        weights = self._normalize_base_weights(base_weights, risk_frame.index)
        metrics = risk_frame.copy()
        for column in ("var_q", "delta_covar", "rolling_drawdown"):
            if column not in metrics.columns:
                metrics[column] = 0.0
            metrics[column] = pd.to_numeric(metrics[column], errors="coerce").fillna(0.0)

        var_penalty = metrics["var_q"].clip(upper=0.0).abs()
        covar_penalty = metrics["delta_covar"].clip(lower=0.0)
        drawdown_penalty = metrics["rolling_drawdown"].clip(upper=0.0).abs()
        total_penalty = var_penalty + covar_penalty + drawdown_penalty
        scaled_penalty = total_penalty / max(float(total_penalty.max()), 1e-9)
        adjusted = weights * np.exp(-self.risk_penalty_strength * scaled_penalty)

        labels = regime_labels.reindex(metrics.index) if regime_labels is not None else pd.Series("NORMAL", index=metrics.index)
        regime_multiplier = labels.map(self.regime_multipliers).fillna(1.0).astype(float)
        adjusted = adjusted * regime_multiplier

        gross = float(adjusted.sum())
        if gross > 0:
            final_weights = adjusted / gross
        else:
            final_weights = adjusted

        exposure_cap = float(regime_multiplier.min()) if len(regime_multiplier) else 1.0
        final_weights = final_weights * min(exposure_cap, 1.0)
        cash_weight = max(0.0, 1.0 - float(final_weights.sum()))

        diagnostics = {
            "risk_penalty_strength": self.risk_penalty_strength,
            "gross_pre_cap": gross,
            "regime_multiplier": {str(k): float(v) for k, v in regime_multiplier.items()},
            "penalties": {
                str(idx): float(val) for idx, val in scaled_penalty.items()
            },
        }
        return AllocationResult(weights=final_weights, cash_weight=cash_weight, diagnostics=diagnostics)
