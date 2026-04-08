import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from src.agents.orchestrator import AgentOrchestrator
from src.agents.contracts import MarketSignal
from src.agents.explainer_local import LocalExplainerAgent
from src.llm.local_client import LocalLLMError
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
async def test_local_client_failure_path():
    """Verify that the LocalLLMClient raises LocalLLMError on failure."""
    from src.llm.local_client import LocalLLMClient
    
    client = LocalLLMClient(base_url="http://invalid-url:11434")
    
    with pytest.raises(LocalLLMError) as excinfo:
        await client.generate("test prompt")
    
    assert "Failed to connect" in str(excinfo.value) or "timed out" in str(excinfo.value)
    print("Local client failure path test passed: Exception raised as expected.")

@pytest.mark.asyncio
async def test_explainer_fallback_when_ollama_unavailable(mock_signal):
    """Verify that LocalExplainerAgent returns a fallback string on LocalLLMError."""
    from src.agents.contracts import RiskDecision, PortfolioProposal, PositionProposal
    
    agent = LocalExplainerAgent()
    # Mock client to raise LocalLLMError
    agent._client.generate = AsyncMock(side_effect=LocalLLMError("Ollama connection refused"))
    
    risk = RiskDecision(
        ticker="SSI", 
        approved=True, 
        action="BUY", 
        position_size_pct=0.1, 
        stop_loss_pct=0.05, 
        take_profit_pct=0.1, 
        max_holding_days=10
    )
    portfolio = PortfolioProposal(
        positions=[PositionProposal(ticker="SSI", action="BUY", weight=0.1, confidence=0.8, rationale="Test")],
        gross_exposure=0.1,
        cash_buffer=0.9,
        notes=["Test"]
    )
    
    explanation = await agent.explain(mock_signal, risk, portfolio)
    
    assert "Explain-Only Fallback" in explanation
    assert "Ollama connection refused" in explanation
    print("Explainer fallback test passed: Fallback string returned.")

@pytest.mark.asyncio
async def test_orchestrator_still_returns_portfolio_when_explainer_fails(mock_signal):
    """Verify that the orchestrator protects the portfolio decision from LLM failures."""
    
    orchestrator = AgentOrchestrator()
    # Mock explainer to raise any exception during batch processing
    orchestrator.explainer.explain_batch = AsyncMock(side_effect=RuntimeError("Batch failure"))
    
    result = await orchestrator.run([mock_signal])
    
    # Core trading decisions MUST exist
    assert "portfolio" in result
    assert result["portfolio"]["positions"] is not None
    assert len(result["portfolio"]["positions"]) > 0
    
    # Explanations should contain fallback message
    assert "explanations" in result
    assert "Fallback" in result["explanations"][0]
    print("Orchestrator protection test passed: Portfolio returned despite explainer failure.")

@pytest.mark.asyncio
async def test_orchestrator_integrity(mock_signal):
    """Verify that enabling/disabling the explainer does not change the deterministic portfolio proposal."""
    
    settings = get_settings()
    
    # 1. Run with explainer ENABLED (Mocked)
    settings.enable_llm_explainer = True
    orch_enabled = AgentOrchestrator()
    orch_enabled.explainer.explain_batch = AsyncMock(return_value=["Mock explanation"])
    result_enabled = await orch_enabled.run([mock_signal])
    
    # 2. Run with explainer DISABLED
    settings.enable_llm_explainer = False
    orch_disabled = AgentOrchestrator()
    result_disabled = await orch_disabled.run([mock_signal])
    
    # 3. Compare the core logic outputs (MUST be identical)
    assert result_enabled["portfolio"]["positions"] == result_disabled["portfolio"]["positions"]
    assert result_enabled["risk_decisions"] == result_disabled["risk_decisions"]
    
    print("Integrity test passed: Trading decisions are identical regardless of LLM state.")

if __name__ == "__main__":
    # Simple manual run if pytest is not available
    async def run_tests():
        signal = MarketSignal(ticker="SSI", current_price=35.0, pred_return=0.05, confidence=0.8, volatility=0.02)
        await test_explainer_fallback(signal)
        await test_orchestrator_integrity(signal)
    
    asyncio.run(run_tests())
