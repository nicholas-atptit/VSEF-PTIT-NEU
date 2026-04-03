"""Tests for the Low-Resource branch logic.

Verifies that the LLM explainer acts only as a decorator and does not 
interfere with the core trading decisions.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from src.agents.orchestrator import AgentOrchestrator
from src.agents.contracts import MarketSignal, PortfolioProposal
from src.agents.explainer_local import LocalExplainerAgent
from config.settings import get_settings

@pytest.fixture
def mock_signal():
    return MarketSignal(
        ticker="SSI",
        current_price=35.0,
        pred_return=0.05,
        confidence=0.8,
        volatility=0.02,
        trend_up_prob=0.7,
        trend_down_prob=0.1,
        trend_sideways_prob=0.2
    )

@pytest.mark.asyncio
async def test_explainer_fallback(mock_signal):
    """Verify that if the LLM client fails, the orchestrator still returns portfolio decisions."""
    
    # 1. Setup Orchestrator with a failing explainer
    orchestrator = AgentOrchestrator()
    orchestrator.explainer = LocalExplainerAgent()
    
    # Mock the internal client to raise an exception
    orchestrator.explainer._client.generate = AsyncMock(side_effect=Exception("Ollama Offline"))
    
    # 2. Run
    result = await orchestrator.run([mock_signal])
    
    # 3. Verify
    assert "portfolio" in result
    assert result["portfolio"]["positions"] is not None
    assert "explanations" in result
    assert "Fallback" in result["explanations"][0]
    print("Fallback test passed: Portfolio returned despite LLM failure.")

@pytest.mark.asyncio
async def test_orchestrator_integrity(mock_signal):
    """Verify that enabling/disabling the explainer does not change the portfolio proposal."""
    
    settings = get_settings()
    
    # 1. Run with explainer ENABLED
    settings.enable_local_explainer = True
    orch_enabled = AgentOrchestrator()
    # Mock to avoid real network call
    orch_enabled.explainer.explain_batch = AsyncMock(return_value=["Mock explanation"])
    result_enabled = await orch_enabled.run([mock_signal])
    
    # 2. Run with explainer DISABLED
    settings.enable_local_explainer = False
    orch_disabled = AgentOrchestrator()
    result_disabled = await orch_disabled.run([mock_signal])
    
    # 3. Compare the core logic outputs
    assert result_enabled["portfolio"]["positions"] == result_disabled["portfolio"]["positions"]
    assert result_enabled["risk_decisions"] == result_disabled["risk_decisions"]
    
    # But explanations should differ
    assert result_enabled["explanations"] == ["Mock explanation"]
    assert "disabled" in result_disabled["explanations"][0].lower()
    
    print("Integrity test passed: Trading decisions are identical regardless of LLM state.")

if __name__ == "__main__":
    # Simple manual run if pytest is not available
    async def run_tests():
        signal = MarketSignal(ticker="SSI", current_price=35.0, pred_return=0.05, confidence=0.8, volatility=0.02)
        await test_explainer_fallback(signal)
        await test_orchestrator_integrity(signal)
    
    asyncio.run(run_tests())
