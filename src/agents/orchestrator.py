from __future__ import annotations

from dataclasses import asdict

from .analyst import AnalystAgent
from .contracts import MarketSignal
from .explainer import ExplainerAgent
from .portfolio import PortfolioAgent
from .risk import RiskAgent


class AgentOrchestrator:
    """End-to-end deterministic decision flow.

    Intended placement:
    current prediction -> normalized MarketSignal -> orchestrator.run()
    """

    def __init__(
        self,
        analyst: AnalystAgent | None = None,
        risk: RiskAgent | None = None,
        portfolio: PortfolioAgent | None = None,
        explainer: ExplainerAgent | None = None,
    ) -> None:
        self.analyst = analyst or AnalystAgent()
        self.risk = risk or RiskAgent()
        self.portfolio = portfolio or PortfolioAgent()
        self.explainer = explainer or ExplainerAgent()

    async def run(self, signals: list[MarketSignal]) -> dict:
        analyst_decisions = [self.analyst.decide(s) for s in signals]
        risk_decisions = [
            self.risk.review(signal=s, analyst=a)
            for s, a in zip(signals, analyst_decisions)
        ]
        portfolio = self.portfolio.build(risk_decisions)

        # Generate explanations if enabled
        explanations = []
        if self.explainer.settings.enable_llm_explainer:
            explanations = await self.explainer.explain_batch(
                signals=signals,
                analyst_decisions=analyst_decisions,
                risk_decisions=risk_decisions,
            )

        return {
            "signals": [asdict(s) for s in signals],
            "analyst_decisions": [asdict(x) for x in analyst_decisions],
            "risk_decisions": [asdict(x) for x in risk_decisions],
            "portfolio": asdict(portfolio),
            "explanations": explanations,
        }
