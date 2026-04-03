"""Local Explainer Agent.

Generates natural language explanations using the local LLM.
Optimized for the low-resource branch.
"""

from __future__ import annotations

import asyncio
from typing import Any

from config.settings import get_settings
from src.llm.local_client import get_local_client, LocalLLMError
from .contracts import MarketSignal, RiskDecision, PortfolioProposal
from .prompts_explainer import build_explainer_prompt


class LocalExplainerAgent:
    """Agent specialized in generating local LLM explanations for trading decisions."""

    def __init__(self, model_name: str | None = None) -> None:
        self.settings = get_settings()
        self._client = get_local_client()

    async def explain(
        self, 
        signal: MarketSignal, 
        risk: RiskDecision, 
        portfolio: PortfolioProposal
    ) -> str:
        """Generate a structured, Vietnamese explanation for a trading decision.
        
        Inputs are strictly from the deterministic core.
        """
        if not self.settings.enable_llm_explainer:
            return "Local LLM Explainer is disabled in settings."

        prompt = build_explainer_prompt(signal, risk, portfolio)
        
        # Execute explanation with fallback
        try:
            explanation = await self._client.generate(prompt)
            return explanation
        except LocalLLMError as e:
            from src.utils.logging import get_logger
            logger = get_logger(__name__)
            logger.error("local_explainer_llm_failure", ticker=signal.ticker, error=str(e))
            return f"Explain-Only Fallback: Trading system decision made, but explanation generation failed. ({str(e)})"
        except Exception as e:
            from src.utils.logging import get_logger
            logger = get_logger(__name__)
            logger.error("local_explainer_unexpected_error", ticker=signal.ticker, error=str(e))
            return f"Fallback: Explanation service unavailable. [{str(e)}]"

    async def explain_batch(
        self, 
        signals: list[MarketSignal], 
        risk_decisions: list[RiskDecision], 
        portfolio: PortfolioProposal
    ) -> list[str]:
        """Process local explanations in parallel."""
        tasks = [
            self.explain(s, r, portfolio) 
            for s, r in zip(signals, risk_decisions)
        ]
        return await asyncio.gather(*tasks)
