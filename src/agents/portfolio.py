from __future__ import annotations

from .contracts import PortfolioProposal, PositionProposal, RiskDecision


class PortfolioAgent:
    """Converts approved single-name decisions into portfolio weights."""

    def __init__(self, max_gross_exposure: float = 1.0, cash_buffer: float = 0.10) -> None:
        self.max_gross_exposure = max_gross_exposure
        self.cash_buffer = cash_buffer

    def build(self, risk_decisions: list[RiskDecision]) -> PortfolioProposal:
        approved = [d for d in risk_decisions if d.approved and d.position_size_pct > 0]
        if not approved:
            return PortfolioProposal(
                positions=[],
                gross_exposure=0.0,
                cash_buffer=1.0,
                notes=["no approved positions"],
            )

        total_requested = sum(d.position_size_pct for d in approved)
        target_budget = max(0.0, self.max_gross_exposure - self.cash_buffer)
        scale = min(1.0, target_budget / total_requested) if total_requested > 0 else 0.0

        positions: list[PositionProposal] = []
        for d in approved:
            weight = round(d.position_size_pct * scale, 4)
            positions.append(
                PositionProposal(
                    ticker=d.ticker,
                    action=d.action,
                    weight=weight,
                    confidence=min(0.95, 0.50 + weight),
                    rationale=f"approved by risk overlay; stop={d.stop_loss_pct:.2%}; tp={d.take_profit_pct:.2%}",
                )
            )

        gross_exposure = round(sum(p.weight for p in positions), 4)
        cash_buffer = round(max(0.0, 1.0 - gross_exposure), 4)
        return PortfolioProposal(
            positions=positions,
            gross_exposure=gross_exposure,
            cash_buffer=cash_buffer,
            notes=["weights scaled to portfolio budget"],
        )
