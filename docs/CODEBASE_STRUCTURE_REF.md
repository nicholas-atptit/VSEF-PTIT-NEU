# Codebase Structure Reference (Post-Refactor)

This document provides a map of the repository structure following the three-phase cleanup and refactor pass.

## Architecture Overview

The system is organized into decoupled layers to ensure auditability and statistical governance.

### Core Layers
- **`src/forecast/`**: **Governed Forecasting Interface**. Provides a unified contract for statistical (`statistical/`) and machine learning models.
- **`src/ml/`**: **Internal ML Engine**. Contains heavy modeling logic, feature engineering, and training pipelines.
- **`src/risk/`**: Conditionality layer specializing in volatility (GARCH) and localized risk metrics (VaR).
- **`src/regime/`**: Contextual layer identifying market regimes (Markov Switching).
- **`src/retrieval/`**: RAG-oriented layer for document ingestion, prep, and retrieval stores.
- **`src/reporting/`**: Synthesis layer that converges model outputs into Analayst Packets and Manifests.

### Repository Map

| Directory | Role | Status |
| :--- | :--- | :--- |
| `scripts/` | Active entry points (Quant Core, Feed, Retrieval). | **ACTIVE** |
| `scripts/legacy/` | Phase runners (1, 2, Hardening) and research archive. | **LEGACY** |
| `tests/quant_core/` | Unit and integration tests for the prediction core. | **ACTIVE** |
| `tests/retrieval/` | Validation for document stores and ingestion. | **ACTIVE** |
| `archive/root_history/`| Historical root-level artifacts (Upgrade traces, etc). | **ARCHIVE** |
| `docs/prompt_runs/archive/` | Clustered historical implementation logs by phase. | **ARCHIVE** |

## Runner Reference

### Primary Execution
- `scripts/run_quant_core.py`: Main forecast/risk/regime orchestrator.
- `scripts/run_analysis_feed.py`: Synthesis and analyst memo generator.
- `scripts/run_retrieval_query.py`: Search and retrieval query interface.

### Legacy / Background
- `scripts/legacy/run_phase1_benchmark.py`: Historical benchmark runner.
- `scripts/legacy/research/`: Exploratory and one-off research scripts.

## Boundary Definitions

> [!NOTE]
> **Governed vs Heavy**: Use `src/forecast` when adding a model that should be consumed by the standard Quant Core orchestrators. Use `src/ml` for developing new feature sets or experiment pipelines that require manual parameter tuning before being "governed" in the core.
