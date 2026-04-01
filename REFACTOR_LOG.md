# Refactor Log

This document records all structural changes made to the repository during the cleanup and reorganization phase.

## Moves and Renames

### Archive Phase
*   **Root Cleanup**: Moved the following to `archive/root/`:
    *   `tmp_*.py`
    *   `pytest*.txt`
    *   `out*.log`
    *   `algo_trading.db`
    *   `fix_unicode.py`
*   **Script Rationalization**: Moved the following to `archive/scripts/`:
    *   `discover_vn100_v1.py` to `discover_vn100_v4.py`
    *   `explore_vnstock.py`
    *   `check_vn100_final.py`
*   **Legacy Source**: Moved `src/labels/` (root-level source) to `archive/src/`.

### Source Consolidation (`src/`)
*   **Data Package (`src/data/`)**: Consolidated the following:
    *   `src/adapters/` -> `src/data/adapters/`
    *   `src/database/` -> `src/data/database/`
    *   `src/datasets/` -> `src/data/datasets/`
    *   `src/historical/` -> `src/data/historical/`
    *   `src/context/` -> `src/data/context/`
*   **ML Package (`src/ml/`)**: Consolidated the following:
    *   `src/accuracy/` -> `src/ml/accuracy/`
    *   `src/backtest/` -> `src/ml/backtest/`
    *   `src/benchmark/` -> `src/ml/benchmark/`
    *   `src/features/` -> `src/ml/features/`
    *   `src/inference/` -> `src/ml/inference/`
    *   `src/models/` -> `src/ml/models/`
    *   `src/pipelines/` -> `src/ml/pipelines/`
    *   `src/training/` -> `src/ml/training/`
    *   `src/training_pipeline/` -> `src/ml/training_pipeline/`
    *   `src/llm/` -> `src/ml/llm/`
*   **Reporting Package (`src/reporting/`)**: Moved `src/reports/` to `src/reporting/reports/`.
*   **Utils Package (`src/utils/`)**: Moved `src/monitoring/` to `src/utils/monitoring/`.
*   **API/App Package (`src/api/`)**: Moved `src/streaming/` and `src/ui/` into `src/api/`.
*   **Engine Package (`src/engine/`)**: Moved `src/agents/`, `src/filtering/`, and `src/portfolio/` into `src/engine/`.

## Import Updates
A global search and replace was performed across all `.py` files to update imports to the new namespaces.

Example replacements:
*   `from src.adapters` -> `from src.data.adapters`
*   `from src.agents` -> `from src.engine.agents`
*   `from src.accuracy` -> `from src.ml.accuracy`
*   ... (and 20+ other namespace mappings)

## Verification
*   Checked `archive/` directory creation: **Success**
*   Checked `src/data/`, `src/ml/`, `src/engine/` contents: **Success**
*   Smoke test run: `python scripts/sync_all_data.py --help`: **Success (Exit Code 0)**
*   Smoke test run: `python scripts/train_ml_tickers.py --help`: **Success (Exit Code 0)**

> [!NOTE]
> All documentation in `docs/` remains untouched and preserved.
