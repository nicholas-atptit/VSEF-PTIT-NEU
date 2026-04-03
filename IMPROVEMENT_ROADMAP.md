# 🚀 Improvement Roadmap

## 1. P0: CRITICAL VALIDITY & SAFETY (Immediate)
**Reasoning**: Ensure the integrity of prediction signals and data security.

| Item | Action | Reasoning |
| :--- | :--- | :--- |
| **Fix Mock Benchmarks** | Replace `np.random` in `src/ml/benchmark/evaluator.py`. | Current CAGR/Sharpe are fake; prevents false signal confidence. |
| **Remove Hardcoded Key** | Move `vnstock_api_key` to `.env`. | Fixes critical security risk in `config/settings.py`. |
| **Fix Label Smoothing** | Update `src/ml/labels/*.py` to use raw market prices only. | Prevents "artificially" high accuracy from smoothed data. |
| **Sanitize Features** | Update `FeatureEngineer` to preserve `close_raw`. | Ensures causal inputs for all downstream indicators. |

## 2. P1: METHODOLOGY & ROBUSTNESS (Secondary)
**Reasoning**: Align training pipeline with financial machine learning best practices.

| Item | Action | Reasoning |
| :--- | :--- | :--- |
| **Walk-Forward Split** | Use `TimeSeriesSplit` in `train_ml_tickers.py`. | Replaces static 80/20 split; reflects real model drift. |
| **Returns-Based Metrics** | Replaces `Accuracy` with `Sharpe/Sortino`. | Accuracy is misleading for non-stationary price data. |
| **Feature Registry** | Implement `src/features/registry.py`. | Standardizes alpha factor computation and dependency management. |
| **Config Centralization** | Move hardcoded settings to `configs/ml_params.yaml`. | Reduces "magic numbers" and improves configurability. |

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
