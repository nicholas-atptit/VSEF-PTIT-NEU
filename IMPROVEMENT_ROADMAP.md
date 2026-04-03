# IMPROVEMENT ROADMAP: VN100 Multi-Agent Trading System

Date: 2026-04-03
Priority Status: High

## Recommended Next Steps

### P0: Intelligence Realism (Critical Structural Risk)
- **RAG Integration**: Fully connect `ZonedRAGService` with `ExplainerAgent`. Currently, the explainer uses a quantitative signal snapshot. Realism requires "context-in-loop" where the agent can search for recent (24H) news on a ticker before rendering a Vietnamese rationale.
- **Error Sensitivity**: Refine exception handling in the `AgentOrchestrator`. If an LLM call fails, the system should fallback to a rule-based "fallback rationale" rather than returning an error string.

### P1: Portfolio Optimization (Maintainability & Scalability)
- **Position Scaling Logic**: Implement a "Portfolio Optimizer" agent (e.g., CVaR or Kelly Criterion based). The current `PortfolioAgent` uses simple budget-scaling, which does not account for ticker covariance or volatility targeting.
- **Consolidate Backtesting**: Merge `src/ml/backtest/paper.py` (event-driven mode) with `scripts/backtest_portfolio_multi_agent.py`. The two logic chains share rebalancing and slippage code but live in different places.

### P2: UI & Observability (Quality Upgrade)
- **Feedback Loop**: Integrate the generated `backtest_portfolio_result.json` into the existing dashboard. Visualization of the equity curve vs individual ticker signals is the next logical step. 
- **Explainability Logging**: Create a dedicated `logs/agent_narratives.jsonl` to store every LLM response for audit and human-in-the-loop verification.

---

## Recommended Order of Execution
1. **RAG Hooking**: 2-3 hours (Focus on `ExplainerAgent` context injection).
2. **Backtest Consolidation**: 4-6 hours (Refactor `paper.py` to be a pure Simulation engine).
3. **Portfolio Optimizer**: 8-12 hours (Add a `PortfolioRiskAgent` specializing in cross-ticker correlations).
