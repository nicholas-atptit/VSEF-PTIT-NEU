# Repository structure (Final)

This document provides a definitive guide to the new production-grade directory layout of the `AI-ML-LLM in Stock` repository.

## Folder Tree

```text
├── archive/        # Legacy scripts, scratch files, and intermediate outputs
├── config/         # System settings and environment configuration (Pydantic)
├── data/           # Persistent storage for market data (CSV/DB)
├── docs/           # Documentation and task-specific reports (Preserved)
├── infra/          # Infrastructure configurations (Docker)
├── logs/           # Application and diagnostic logs
├── models/         # Trained model artifacts and evaluation reports
├── scripts/        # Production-grade entry points for core workflows
│   ├── core/       # High-priority production scripts
│   └── utils/      # Operational helper scripts
├── src/            # Core system source code
│   ├── api/        # Interaction layers (Web/Streaming)
│   ├── data/       # Data ingestion and universe management
│   ├── engine/      # Core logic and multi-agent system
│   ├── ml/         # Machine Learning pipeline and trackers
│   ├── reporting/  # Post-prediction insight generation
│   ├── utils/      # Common utilities (Logging/Time)
│   └── validators/ # Data quality and validation
├── tests/          # Unit and integration test suites
└── web/            # Web dashboard and UI frontend
```

## Folder Descriptions

### `src/` (Source Code)
*   **`api/`**: Contains API definitions and real-time communication modules.
*   **`data/`**: Handles all interactions with raw data sources and local persistence.
*   **`engine/`**: The "brain" of the system, where multi-agent logic and algorithms reside.
*   **`ml/`**: Manages the lifecycle of machine learning models, from feature engineering to tracking.
*   **`reporting/`**: Transforms model outputs into human-readable insights.

### `scripts/` (Workflows)
All operational scripts should reside here. Use `core/` for daily-run scripts and `utils/` for one-off maintenance tasks.

### `archive/` (Safety)
Anything that is not part of the active production path but may still be useful for reference should be moved here.

## Developer Guidance

When adding new functionality:
1.  **Logic**: Place core logic in the appropriate `src/` sub-package.
2.  **Inversion of Control**: Avoid hardcoded configuration; use `get_settings()` from `config.settings`.
3.  **Imports**: Always use absolute project imports (e.g., `from src.data.universe import ...`).
4.  **Logging**: Use the centralized logger from `src.utils.logging`.
5.  **Paths**: Use `Path` objects relative to `PROJECT_ROOT` when defining new file dependencies.
