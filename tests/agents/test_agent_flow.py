import pytest
from src.agents.orchestrator import AgentOrchestrator
from src.signals.builder import build_market_signal


@pytest.mark.asyncio
async def test_agent_flow_buy_case():
    signal = build_market_signal(
        ticker="SSI",
        current_price=35.0,
        model_output={
            "trend_probabilities": {"up": 0.70, "down": 0.10, "sideways": 0.20},
            "expected_range": {"bottom_10th": 34.8, "median_50th": 36.0, "ceiling_90th": 37.0},
            "confidence": 0.74,
            "volatility": 0.025,
        },
        feature_snapshot={"rsi_14": 55, "sma_20": 35.0, "sma_50": 34.0},
        sentiment_payload={"sentiment_score": 0.18},
    )

    orchestrator = AgentOrchestrator()
    result = await orchestrator.run([signal])

    assert result["analyst_decisions"][0]["action"] == "BUY"
    assert result["risk_decisions"][0]["approved"] is True
    assert result["portfolio"]["gross_exposure"] > 0


@pytest.mark.asyncio
async def test_agent_flow_blocks_high_volatility():
    signal = build_market_signal(
        ticker="VGI",
        current_price=40.0,
        model_output={
            "trend_probabilities": {"up": 0.72, "down": 0.08, "sideways": 0.20},
            "expected_range": {"bottom_10th": 39.0, "median_50th": 41.0, "ceiling_90th": 42.0},
            "confidence": 0.76,
            "volatility": 0.20,
        },
        feature_snapshot={"rsi_14": 61},
        sentiment_payload={"sentiment_score": 0.25},
    )

    orchestrator = AgentOrchestrator()
    result = await orchestrator.run([signal])

    assert result["risk_decisions"][0]["approved"] is False
    assert result["risk_decisions"][0]["action"] == "HOLD"
