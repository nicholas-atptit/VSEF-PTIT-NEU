from __future__ import annotations

import asyncio
from dataclasses import asdict

from .analyst import AnalystAgent
from .contracts import MarketSignal
from .explainer_local import LocalExplainerAgent
from .portfolio import PortfolioAgent
from .risk import RiskAgent


class AgentOrchestrator:
    """VN100 Low-Resource Orchestrator. 
    
    Deterministic Flow:
    Analyst -> Risk -> Portfolio -> (Local Explanation)
    """

    def __init__(
        self,
        analyst: AnalystAgent | None = None,
        risk: RiskAgent | None = None,
        portfolio: PortfolioAgent | None = None,
        explainer: LocalExplainerAgent | None = None,
    ) -> None:
        self.analyst = analyst or AnalystAgent()
        self.risk = risk or RiskAgent()
        self.portfolio = portfolio or PortfolioAgent()
        self.explainer = explainer or LocalExplainerAgent()

    async def run(self, signals: list[MarketSignal]) -> dict:
        """Execute the one-way deterministic pipeline for low-resource deployment."""
        
        # 1. Analyst Level (Directional conviction)
        analyst_decisions = [self.analyst.decide(s) for s in signals]
        
        # 2. Risk Level (Boundaries & Constraints)
        risk_decisions = [
            self.risk.review(signal=s, analyst=a) 
            for s, a in zip(signals, analyst_decisions)
        ]
        
        # 3. Portfolio Level (Allocation & Aggregation)
        # This is the "Trading Decision" point - final and deterministic.
        portfolio_proposal = self.portfolio.build(risk_decisions)

        # 4. Explanation Level (Optional Post-Process)
        # LLM failure must NOT impact the previous steps.
        explanations = []
        try:
            explanations = await asyncio.wait_for(
                self.explainer.explain_batch(
                    signals=signals,
                    risk_decisions=risk_decisions,
                    portfolio=portfolio_proposal
                ),
                timeout=1.0,
            )
        except Exception as e:
            from src.utils.logging import get_logger
            logger = get_logger(__name__)
            logger.error("orchestrator_explanation_fallback", error=str(e))
            explanations = [f"Fallback: Local Explainer failed: {str(e)}"] * len(signals)

        return {
            "signals": [asdict(s) for s in signals],
            "analyst_decisions": [asdict(x) for x in analyst_decisions],
            "risk_decisions": [asdict(x) for x in risk_decisions],
            "portfolio": asdict(portfolio_proposal),
            "explanations": explanations,
        }
