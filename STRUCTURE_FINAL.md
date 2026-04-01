# 🗺️ Proposed Target Repository Structure

## 1. PROJECT ROOT
- `configs/`: Centralized settings (YAML, Python).
- `data/`: Ingested, staged, and featured data.
- `src/`: Core logic modules (Data, Features, Labels, Training, Inference, Backtest).
- `scripts/`: Final operational entry points.
- `models/`: Curated model artifacts (VN100 focus).
- `reports/`: Audit logs, experiment results, and performance analysis.
- `docs/`: Technical documentation and design notes.
- `archive/`: Redundant/legacy scripts and data.

## 2. FOLDER RESPONSIBILITIES

| Path | Responsibility |
| :--- | :--- |
| **configs/** | Unified source of truth for ML params, API keys, and paths. |
| **data/raw/** | Original data downloads (immutable). |
| **data/staging/** | Cleaned and partitioned OHLCV (Parquet preferred). |
| **data/features/** | Computed feature matrices (versioned per registry). |
| **src/data/** | Data source adapters and universe logic. |
| **src/features/** | Central Factor Registry and indicator logic. |
| **src/labels/** | Causal label generation logic. |
| **src/training/** | Orchestrated training and cross-validation loops. |
| **src/inference/** | Prediction engines for online/batch modes. |
| **src/backtest/** | Event-driven and vectorized testing frameworks. |

## 3. GUIDANCE FOR FUTURE CODING
- **New Indicators**: Add to `src/features/registry.py` first.
- **New Labels**: Add to `src/labels/` and unit-test for causality.
- **New Ingestion**: Add adapter to `src/data/adapters/`.
- **New Scripts**: Keep thin, import core logic from `src/`.
- **Path Handling**: Always use `config/settings.py` paths, never hardcoded strings.

---
*This structure is designed for production reliability and automated scaling.*
