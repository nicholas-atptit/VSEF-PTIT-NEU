from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

from config.settings import get_settings
from src.ml.llm.client import get_llm_client
from .contracts import MarketSignal, AnalystDecision, RiskDecision


class ExplainerAgent:
    """Agent that generates natural language explanations for trading decisions.
    
    Uses an LLM (Ollama/OpenAI/Gemini) to synthesize technical signals and 
    risk constraints into a human-readable narrative.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.settings = get_settings()
        self.model_name = model_name or self.settings.llm_model_explainer
        self._client = get_llm_client()

    async def explain(
        self, 
        signal: MarketSignal, 
        analyst: AnalystDecision, 
        risk: RiskDecision
    ) -> str:
        """Generate a markdown explanation for a single ticker decision."""
        if not self.settings.enable_llm_explainer:
            return "LLM Explainer is disabled in settings."

        prompt = self._build_prompt(signal, analyst, risk)
        
        try:
            response = await self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a senior financial analyst and quantitative researcher. Your goal is to explain trading signals clearly and concisely."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            return response.choices[0].message.content or "No explanation generated."
        except Exception as e:
            from src.utils.logging import get_logger
            logger = get_logger(__name__)
            logger.error("explainer_agent_error", ticker=signal.ticker, error=str(e))
            return f"Failed to generate explanation: {str(e)}"

    def _build_prompt(
        self, 
        signal: MarketSignal, 
        analyst: AnalystDecision, 
        risk: RiskDecision
    ) -> str:
        """Construct the prompt from the quantitative context."""
        return f"""
Explain the following trading decision for {signal.ticker}:

## Quantitative Context:
- Current Price: {signal.current_price:.2f}
- Prediction: Up={signal.trend_up_prob:.2f}, Down={signal.trend_down_prob:.2f}, Side={signal.trend_sideways_prob:.2f}
- Confidence: {signal.confidence:.2f}
- Volatility: {signal.volatility:.4f}
- Technical Indicators: RSI={signal.rsi_14 or 'N/A'}, SMA20={signal.sma_20 or 'N/A'}

## Agent Decisions:
- Analyst Action: {analyst.action}
- Analyst Logic: {', '.join(analyst.reasons)}
- Risk Status: {"Approved" if risk.approved else "REJECTED"}
- Risk Veto Reasons: {', '.join(risk.veto_reasons) if risk.veto_reasons else "None"}

Generate a brief (2-3 paragraph) explanation for a human trader in Vietnamese. 
Highlight why the decision was made, what the key technical drivers are, and any risks identified.
Use a professional, objective tone. Output in Markdown.
"""

    async def explain_batch(self, signals: list[MarketSignal], analyst_decisions: list[AnalystDecision], risk_decisions: list[RiskDecision]) -> list[str]:
        """Process multiple signals in parallel."""
        tasks = [
            self.explain(s, a, r) 
            for s, a, r in zip(signals, analyst_decisions, risk_decisions)
        ]
        return await asyncio.gather(*tasks)
