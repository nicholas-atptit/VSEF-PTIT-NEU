# AUDIT REPORT: Multi-Agent Trading System Skeleton Refactor

Date: 2026-04-03
Status: Phase 3 Completed (Intelligence & Backtesting)

## Current Architecture Overview
The repository has been transitioned from a sequential, rule-heavy script structure to a modern, asynchronous multi-agent architecture.

### Entry Points
- `src/main.py`: Primary application entry.
- `scripts/run_agent_decision.py`: Direct CLI for multi-agent signal generation.
- `scripts/backtest_portfolio_multi_agent.py`: New daily-simulated portfolio backtester.

### Data Flow
1. **Extraction**: `VN100DataLoader` fetches OHLCV from local CSV fallbacks or TimescaleDB.
2. **Features**: `FeatureEngineer` transforms raw data into normalized vectors (SMA, RSI, returns).
3. **Signal Builder**: `build_market_signal`アダプター layers current market context into a strict `MarketSignal` contract.
4. **Agent Graph (Async)**: 
   - `AnalystAgent`: Rule-based/ML-based trading signals.
   - `RiskAgent`: Compliance checks and position sizing.
   - `ExplainerAgent`: Natural language rationale generation (Vietnamese).
5. **Portfolio Decision**: `AgentOrchestrator` synthesizes agent outputs into a `PortfolioProposal`.
6. **Execution/Simulation**: `PaperTradingEngine` or `PortfolioBacktester` executes trades with commission/slippage.

## Completed Components
- [x] **Asynchronous Refactor**: Unified `asyncio` execution across the agent lifecycle.
- [x] **LLM Intel Integration**: `ExplainerAgent` fully operational with Vietnamese prompt localization.
- [x] **Portfolio Backtesting**: Day-by-day vectorized simulation for multi-ticker universes.
- [x] **Import Correctness**: Fixed critical broken imports in `src/data/context` and `src/ml/backtest`.
- [x] **Date-Type Safety**: Standardized `pd.Timestamp` handling to resolve comparison errors.

## Technical Debt & Remaining Risks
### Risk 1: RAG Connectivity (P0)
- The RAG/context package (`src/data/context/`) depends on an external ChromaDB instance. If not connected, the `ExplainerAgent` may lack deep contextual narrative.

### Risk 2: Portfolio Logic Complexity (P1)
- `PortfolioAgent` currently uses a simple budget-scaling heuristic. Advanced risk-parity or covariance-based optimization is needed for production.

### Risk 3: Test Coverage (P2)
- Current tests verify the *flow* and *orchestration*. Predictive accuracy and edge-case handling for the ML models need more unit-level "gold set" tests.

## Code Smells & Inconsistencies
- **Redundancy**: `src/ml/backtest/event_driven.py` and `src/ml/backtest/paper.py` share simulation logic that should be consolidated.
- **Naming**: `PortfolioProposal` vs `PortfolioAllocation` used inconsistently in the codebase (Standardized to `Proposal` in this refactor).
- **Hardcoding**: Some LLM model names and API settings still rely on `settings.py` defaults; should use environment-specific overrides in `.env`.
