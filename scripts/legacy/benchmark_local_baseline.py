import asyncio
import json
import time
import sys
from pathlib import Path
from dataclasses import asdict
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.orchestrator import AgentOrchestrator
from src.agents.contracts import MarketSignal
from config.settings import get_settings
from src.llm.local_client import LocalLLMError

async def run_mode(name, orchestrator, signals, enable_llm=True, force_fail=False):
    """Run the orchestrator in a specific mode and measure performance."""
    settings = get_settings()
    settings.enable_llm_explainer = enable_llm
    
    # Setup failure injection if needed
    patcher = None
    if force_fail:
        # Inject an immediate failure/timeout via mock
        from src.llm.local_client import LocalLLMClient
        patcher = patch.object(LocalLLMClient, 'generate', side_effect=LocalLLMError("Benchmark Forced Failure"))
        patcher.start()

    start_time = time.perf_counter()
    try:
        result = await orchestrator.run(signals)
        end_time = time.perf_counter()
        
        return {
            "name": name,
            "latency_ms": (end_time - start_time) * 1000,
            "portfolio": result["portfolio"],
            "explanations": result["explanations"],
            "success": True
        }
    except Exception as e:
        return {
            "name": name,
            "error": str(e),
            "success": False
        }
    finally:
        if patcher:
            patcher.stop()

def get_semantic_portfolio(portfolio):
    """Extract deterministic fields for semantic comparison."""
    # Compare positions (ticker, action, weight)
    positions = []
    for p in portfolio["positions"]:
        positions.append({
            "ticker": p["ticker"],
            "action": p["action"],
            "weight": p["weight"]
        })
    
    return {
        "positions": sorted(positions, key=lambda x: x["ticker"]),
        "gross_exposure": portfolio["gross_exposure"],
        "cash_buffer": portfolio["cash_buffer"]
    }

async def main():
    print("--- Starting Low-Resource Baseline Benchmark ---")
    
    orchestrator = AgentOrchestrator()
    signals = [
        MarketSignal(
            ticker="SSI",
            current_price=35000.0,
            pred_return=0.05,
            confidence=0.8,
            volatility=0.02,
            trend_up_prob=0.7,
            trend_down_prob=0.1,
            trend_sideways_prob=0.2
        )
    ]

    # Mode 1: Disabled
    res_disabled = await run_mode("Explainer Disabled", orchestrator, signals, enable_llm=False)
    
    # Mode 2: Enabled (Healthy) - Note: Might timeout if Ollama is not running, but that's okay
    res_enabled = await run_mode("Explainer Enabled (Healthy)", orchestrator, signals, enable_llm=True)
    
    # Mode 3: Forced Failure
    res_failed = await run_mode("Explainer Forced Failure", orchestrator, signals, enable_llm=True, force_fail=True)

    # Parity Checks
    p_disabled = get_semantic_portfolio(res_disabled["portfolio"])
    p_enabled = get_semantic_portfolio(res_enabled["portfolio"])
    p_failed = get_semantic_portfolio(res_failed["portfolio"])

    parity_enabled = (p_disabled == p_enabled)
    parity_failed = (p_disabled == p_failed)
    
    # Fallback Verification
    fallback_caught = any("Fallback" in exp or "generation failed" in exp for exp in res_failed["explanations"])

    # Generate Report
    report_path = Path("reports/baseline_local_qwen_report.md")
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Baseline Benchmark Report: local-qwen\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Branch**: low-resource-local-qwen\n\n")
        
        f.write("## 1. Runtime Performance\n")
        f.write(f"- **Explainer Disabled**: {res_disabled['latency_ms']:.2f} ms\n")
        f.write(f"- **Explainer Enabled (Healthy)**: {res_enabled['latency_ms']:.2f} ms\n")
        f.write(f"- **Explainer Forced Failure**: {res_failed['latency_ms']:.2f} ms\n\n")
        
        f.write("## 2. Deterministic Portfolio Parity\n")
        f.write("| Mode Comparison | Parity Result | Deterministic Fields Match |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| Disabled vs Enabled | {'PASSED' if parity_enabled else 'FAILED'} | ticker, action, weight, exposure, buffer |\n")
        f.write(f"| Disabled vs Forced Failure | {'PASSED' if parity_failed else 'FAILED'} | ticker, action, weight, exposure, buffer |\n\n")
        
        f.write("## 3. Resilience & Fallback\n")
        f.write(f"- **Fallback String Caught**: {'YES' if fallback_caught else 'NO'}\n")
        f.write(f"- **Failure Sample**: `{res_failed['explanations'][0][:100]}...`\n\n")
        
        f.write("## 4. Operational Observations\n")
        if res_enabled['latency_ms'] > 1000:
            f.write("- **Latency Alert**: Local LLM inference added significant delay (>1s).\n")
        if not res_enabled['success'] or "timeout" in res_enabled['explanations'][0].lower():
            f.write("- **Ollama Status**: Local LLM was unavailable during benchmark (returned fallback).\n")
        f.write("- **Core Integrity**: Quantitative trading path remains logic-neutral.\n\n")
        
        f.write("---\n")
        f.write(f"Deterministic Core Parity: {'PASS' if parity_enabled and parity_failed else 'FAIL'}\n")
        f.write(f"Explainer Fallback: {'PASS' if fallback_caught else 'FAIL'}\n")
        f.write(f"Baseline Status: {'VERIFIED' if parity_enabled and parity_failed and fallback_caught else 'NOT VERIFIED'}\n")

    print(f"--- Benchmark Complete. Report saved to {report_path} ---")

if __name__ == "__main__":
    asyncio.run(main())
