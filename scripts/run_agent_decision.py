import asyncio
import json
import sys
from pathlib import Path

# Add project root to path for local run
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.orchestrator import AgentOrchestrator
from src.agents.contracts import MarketSignal
from src.utils.logging import get_logger

logger = get_logger(__name__)


async def main() -> None:
    """Demo Script for VN100 Low-Resource Deployment Branch.
    
    Demonstrates the Signal -> Risk -> Portfolio -> (Local Explanation) flow.
    """
    
    # 1. Prepare a mock MarketSignal
    # In production, this comes from src.ml.signal_generator.SignalGenerator
    signal = MarketSignal(
        ticker="SSI",
        current_price=35000.0,
        pred_return=0.045,
        confidence=0.85,
        volatility=0.02,
        trend_up_prob=0.68,
        trend_down_prob=0.12,
        trend_sideways_prob=0.20,
        rsi_14=58.5,
        sma_20=34500.0,
        sma_50=33800.0,
        sma_200=31200.0,
        atr_14=950.0,
        regime="trend"
    )

    # 2. Execute Orchestrator (Multi-Agent Deterministic Core)
    orchestrator = AgentOrchestrator()
    
    print("\n--- Running VN100 Agent Orchestrator ---")
    print(f"Ticker: {signal.ticker} | Current Price: {signal.current_price:,.0f} VND")
    
    result = await orchestrator.run([signal])
    
    # 3. Clean JSON Output for User Report
    output = {
        "ticker": signal.ticker,
        "trading_decision": {
            "action": result["analyst_decisions"][0]["action"],
            "risk_approved": result["risk_decisions"][0]["approved"],
            "position_weight": result["portfolio"]["positions"][0]["weight"] if result["portfolio"]["positions"] else 0.0,
            "risk_veto_reasons": result["risk_decisions"][0]["veto_reasons"]
        },
        "portfolio": result["portfolio"],
        "explanations": result["explanations"],
        "local_llm_explanation": result["explanations"][0]
    }
    
    print("\n=== FINAL AGENT OUTPUT (VN100-LOCAL) ===")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    
    # --- Visual Console Summary ---
    print("\n--- Summary Narrative ---")
    print(f"Action Recommendation: {output['trading_decision']['action']}")
    print(f"Local LLM Explanation: \n{output['local_llm_explanation']}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logger.error("demo_script_failed", error=str(e))
        sys.exit(1)
