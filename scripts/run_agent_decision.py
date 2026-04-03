from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path for local run
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.orchestrator import AgentOrchestrator
from src.signals.builder import build_market_signal


async def main() -> None:
    # 1. Mock inputs that represent current repo's typical ML outputs
    mock_model_output = {
        "trend_probabilities": {"up": 0.68, "down": 0.12, "sideways": 0.20},
        "expected_range": {"bottom_10th": 34.5, "median_50th": 35.8, "ceiling_90th": 36.9},
        "confidence": 0.72,
        "volatility": 0.028,
    }
    feature_snapshot = {
        "rsi_14": 57.0,
        "sma_20": 35.0,
        "sma_50": 34.2,
        "sma_200": 31.5,
        "atr_14": 0.9,
    }
    sentiment_payload = {"sentiment_score": 0.20}

    # 2. Normalize to stable MarketSignal contract
    signal = build_market_signal(
        ticker="SSI",
        current_price=35.0,
        model_output=mock_model_output,
        feature_snapshot=feature_snapshot,
        sentiment_payload=sentiment_payload,
    )

    # 3. Execute multi-agent rules
    orchestrator = AgentOrchestrator()
    result = await orchestrator.run([signal])
    
    print("\n=== Multi-Agent Upgrade Decision Output ===")
    print(json.dumps(result, indent=2))
    print("\nResult summary:")
    print(f"Ticker: {result['signals'][0]['ticker']}")
    print(f"Action: {result['analyst_decisions'][0]['action']}")
    print(f"Risk Approved: {result['risk_decisions'][0]['approved']}")
    print(f"Position Size: {result['portfolio']['positions'][0]['weight'] if result['portfolio']['positions'] else 0.0}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
