# STRUCTURE FINAL: VN100 Multi-Agent Project

Date: 2026-04-03
Status: Phase 3 Completed (Full Integration)

## Final Repository Tree (Key Folders)
```
/
├── config/                # Centralized Settings & Secrets
├── scripts/               # Operational Scripts (Backtest, Sync, Demo)
├── src/
│   ├── agents/            # Multi-Agent Intelligent Layer (Async)
│   │   ├── AnalystAgent   # Technical/Quantitative Decision Maker
│   │   ├── RiskAgent      # Protective Filter & Position Sizing
│   │   ├── ExplainerAgent # LLM-based Narrative Generator (Vietnamese)
│   │   └── Orchestrator   # Async Graph Coordinator
│   ├── data/              # Data Management & Context (RAG) layer
│   │   ├── context/       # ChromaDB, Embedding, and News Ingestion
│   │   ├── historical/    # Backdate processing & split handling
│   │   └── universe.py    # VN100 constituent managers
│   ├── ml/                # Machine Learning Pipeline
│   │   ├── backtest/      # Trading Simulation Engines
│   │   ├── inference/     # Predictor Wrappers
│   │   ├── signals/       # Quantitative Signal Generation
│   │   └── trainer.py     # Dual-Model (Trend/Range) Trainer
│   ├── reporting/         # Reports & Insight Generators
│   ├── utils/             # Cross-cutting logging & time utilities
│   └── main.py            # API / System Entry Point
├── reports/               # Auto-generated signal/backtest outputs
└── docs/                  # Technical notes & Architectural decisions
```

## Guidance for Future Development

### 1. Where do new Agents go?
- New agent logic should reside in `src/agents/`. 
- Ensure every agent inherits from a common async interface and returns a defined dataclass from `src/agents/contracts.py`.

### 2. Where do new Simulation logics go?
- Simulation and rebalancing logic should be added to `src/ml/backtest/`. 
- Prefer modifying the `PortfolioBacktester` script for multi-ticker logic or `paper.py` for event-driven execution.

### 3. How to add new ML features?
- Add new feature calculation logic to `src/ml/feature_engineering.py`.
- Ensure parity by using the `FeatureEngineer` class across both training and inference paths.

### 4. How to scale LLM capabilities?
- Add new LLM prompt templates and processing logic to `src/agents/explainer.py`.
- Use `src/ml/llm/client.py` for a unified model provider abstraction.
