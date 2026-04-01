# 🚀 Improvement Roadmap

## 1. P0: CRITICAL VALIDITY & SAFETY (Immediate)
**Reasoning**: Ensure the integrity of prediction signals before capital risk.

| Item | Action | Reasoning |
| :--- | :--- | :--- |
| **Fix Label Smoothing** | Update `src/ml/labels/*.py` to use raw market prices only. | Prevents "artificially" high accuracy from smoothed data. |
| **Sanitize Features** | Update `FeatureEngineer` to preserve `close_raw`. | Ensures causal inputs for all downstream indicators. |
| **Artifact Pruning** | Archive non-VN100 models in `models/`. | Improves repo maintainability and disk I/O. |

## 2. P1: MAINTAINABILITY & ROBUSTNESS (Secondary)
**Reasoning**: Standardize the environment for team collaboration and production scaling.

| Item | Action | Reasoning |
| :--- | :--- | :--- |
| **Feature Registry** | Implement `src/features/registry.py`. | Standardizes alpha factor computation and dependency management. |
| **Config Centralization** | Move hardcoded settings to `configs/ml_params.yaml`. | Reduces "magic numbers" and improves configurability. |
| **Data Ingestion** | Unified `sync_all_data.py` as the master orchestrator. | Prevents database conflicts across different sync scripts. |

## 3. P2: PERFORMANCE & QUALITY UPGRADES
**Reasoning**: Optional enhancements for long-term scalability.

| Item | Action | Reasoning |
| :--- | :--- | :--- |
| **Parquet Transition** | Migrating from ticker-split CSVs to partitioned Parquet files. | Significant improvement in data loading speed. |
| **Benchmark Baseline** | Integrated `VNINDEX` relative performance in all backtests. | Provides better context for alpha generation. |
| **Walk-Forward Loop** | Retraining models periodically during backtests. | Accounts for market regime changes dynamically. |

---

## EXECUTION ORDER RECOMMENDATION
1. **Validity Check (F-01, L-01)**: Fix Kalman smoothing vs Label interaction.
2. **Environment Cleanup (S-01)**: Prune models and mass CSV files.
3. **Registry Pattern (F-02)**: Formalize factor management.
4. **Data Infrastructure (D-02)**: Optimize I/O logic.
