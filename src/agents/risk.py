from __future__ import annotations

from .contracts import AnalystDecision, MarketSignal, RiskDecision


class RiskAgent:
    """Hard risk overlay.

    This stays deterministic.
    Do not outsource this layer to an LLM.
    """

    def __init__(
        self,
        max_position_pct: float = 0.20,
        base_risk_pct: float = 0.01,
        max_volatility: float = 0.08,
    ) -> None:
        self.max_position_pct = max_position_pct
        self.base_risk_pct = base_risk_pct
        self.max_volatility = max_volatility

    def review(self, signal: MarketSignal, analyst: AnalystDecision) -> RiskDecision:
        veto_reasons: list[str] = []

        if analyst.action == "HOLD":
            return RiskDecision(
                ticker=signal.ticker,
                approved=False,
                action="HOLD",
                position_size_pct=0.0,
                stop_loss_pct=0.0,
                take_profit_pct=0.0,
                max_holding_days=0,
                veto_reasons=["analyst returned HOLD"],
            )

        if signal.volatility > self.max_volatility:
            veto_reasons.append(
                f"volatility {signal.volatility:.2%} exceeds cap {self.max_volatility:.2%}"
            )

        if analyst.confidence < 0.55:
            veto_reasons.append("confidence below execution threshold")

        approved = not veto_reasons

        volatility_scale = min(1.0, max(0.25, self.max_volatility / max(signal.volatility, 1e-6)))
        confidence_scale = min(1.0, max(0.25, analyst.confidence))
        position_size_pct = self.max_position_pct * volatility_scale * confidence_scale
        position_size_pct = round(min(self.max_position_pct, position_size_pct), 4)

        stop_loss_pct = 0.03 if signal.volatility < 0.03 else 0.05
        take_profit_pct = round(stop_loss_pct * 2.0, 4)

        return RiskDecision(
            ticker=signal.ticker,
            approved=approved,
            action=analyst.action if approved else "HOLD",
            position_size_pct=position_size_pct if approved else 0.0,
            stop_loss_pct=stop_loss_pct if approved else 0.0,
            take_profit_pct=take_profit_pct if approved else 0.0,
            max_holding_days=5 if approved else 0,
            veto_reasons=veto_reasons,
        )
