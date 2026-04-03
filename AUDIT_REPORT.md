# Repository Audit Report - Low-Resource Deployment Branch

## 1. Current Architecture Analysis

### Entry Points & Orchestration
- **Main Entry**: `src/ml/signal_generator.py` acts as the primary interface for generating trading signals.
- **Orchestrator**: `src/agents/orchestrator.py` (`AgentOrchestrator`) coordinates the multi-agent flow.
- **Deterministic Core**: `AnalystAgent`, `RiskAgent`, and `PortfolioAgent` perform cascading logic based on quantitative data.

### Data Flow (End-to-End)
1. **ML Pipeline**: Produces raw predictions (trend probabilities, expected ranges).
2. **Signal Builder**: Normalizes raw data into a `MarketSignal` dataclass (`src/signals/builder.py`).
3. **Analyst Agent**: Makes a raw directional decision based on the signal.
4. **Risk Agent**: Reviews the analyst's decision against hard constraints (volatility, max position).
5. **Portfolio Agent**: Consolidates approved risk decisions into a portfolio proposal.
6. **Explainer Agent**: (Optional) Generates a natural language narrative for the final decision.

### LLM / Explainer Integration
- **Current Logic**: `src/agents/explainer.py` calls an LLM client.
- **Client Implementation**: `src/ml/llm/client.py` uses `AsyncOpenAI` pointing to either cloud (OpenAI/Gemini/Groq) or local (Ollama).
- **Placement**: The explainer is placed **LAST** in the orchestrator, ensuring it has no feedback loop into the trading logic.

## 2. Completed vs. Incomplete Components

| Component | Status | Location |
| :--- | :--- | :--- |
| ML Prediction Pipeline | Completed | `src/ml/` |
| Risk & Portfolio Logic | Completed | `src/engine/risk.py`, `src/engine/portfolio/` |
| Multi-Agent Orchestration | Completed | `src/agents/orchestrator.py` |
| Local LLM (Ollama) Client | Implemented | `src/ml/llm/client.py` |
| Explainer Logic | Implemented | `src/agents/explainer.py` |
| Local Explainer Optimization | **Incomplete** | Needs branch-specific tuning |

## 3. Technical Debt & Risks

- **LLM Latency**: Local models (Ollama) can be slow on CPU/Low-GPU systems. Needs asynchronous handling (already handled by `asyncio.gather` in orchestrator).
- **Hardcoded Configs**: Some default models in `settings.py` point to `qwen2.5:7b` instead of the requested `qwen3:8b`.
- **Prompt Complexity**: Current prompts might be too verbose for an 8B model to handle quickly while maintaining Vietnamese output quality.

## 4. Code Smells & Architectural Weaknesses
- **Tight coupling in SignalGenerator**: The `SignalGenerator` directly instantiates `AgentOrchestrator`. While acceptable for now, it makes testing harder. 
- **Explain-only enforcement**: It is currently enforced by code ordering, but there is no structural interface preventing a "Controller Agent" from being added.

## 5. Major Risks for Low-Resource Branch
- **Memory Exhaustion**: Running an 8B model alongside the ML pipeline might hit RAM limits on 16GB machines.
- **Hallucinations**: Qwen models can sometimes deviate from the quantitative facts if the prompt isn't strict.

---
*Signed: Antigravity - Senior Software Architect*
