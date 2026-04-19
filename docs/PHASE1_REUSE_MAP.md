# Phase 1 Reuse Map

This document captures the repo audit completed before the Phase 1 forecasting
architecture work started on branch `rebuild-regime-risk-aware-forecasting`.

## Reuse Decisions

| Existing module | Purpose | Action | Reason |
| --- | --- | --- | --- |
| `src/data/adapters/vnstock_adapter.py` | Canonical real OHLCV and index fetch + normalization | keep as-is | Already the repo's data boundary and used by real-data workflows |
| `src/ml/data_loader.py` | File-backed daily/context loaders and artifact metadata | wrap | Useful for prepared/context assets, but too broad to be the new forecasting contract |
| `src/ml/feature_engineering.py` | Leakage-safe shared feature generation | keep as-is | Strongest reusable backbone piece, already used by trainer/tests |
| `src/ml/features/registry.py` | Approved feature sets and feature governance | keep as-is | Directly reusable for manifests and explicit feature selection |
| `data/processed/ml_5y/*.csv` | Prepared per-ticker feature datasets | keep as-is | Real prepared pipeline already exists and should be reused |
| `src/ml/trainer.py` | Current prep, target creation, training, inference, manifests | wrap / selective refactor | Valuable logic exists, but the module is too monolithic for the Phase 1 layered architecture |
| `src/ml/models/base.py` | Low-level estimator contract | replace | Current contract is estimator-centric, not forecast-frame-centric |
| `src/ml/models/sarimax.py`, `src/ml/models/ets.py` | Statistical estimator wrappers | wrap / selective refactor | Algorithms exist, but they return arrays and fit the legacy trainer path |
| `src/ml/models/xgboost_model.py`, `src/ml/models/lightgbm_model.py` | Booster estimator wrappers | wrap / selective refactor | Reusable estimators, but need forecast-model adapters |
| `src/ml/models/cart.py` | Tree baseline | replace for Phase 1 scope | Not in the requested Phase 1 model list |
| `src/ml/backtest/real_data.py` | Fixed-window real-data evaluation | replace | Useful reference for fetch/eval semantics, but not unified walk-forward |
| `src/ml/backtest/forward_return.py` | Multi-horizon forward-return benchmarking | refactor / mine for logic | Contains useful metrics and artifact conventions, but still tied to the legacy trainer outputs |
| `src/ml/backtest/strategy_backtest.py` | Threshold-based, cost-aware strategy backtest | refactor / mine for logic | Strong reusable cost logic, but too coupled to old artifact layouts |
| `src/ml/backtest/walk_forward_all_models_stacking.py` | Leakage-aware rolling forecasts plus stacking | replace | Too specialized and already includes out-of-scope meta-model logic |
| `src/ml/risk.py` | Monte Carlo scenario simulator | wrap | Directly useful deterministic semantics for the Phase 1 risk layer |
| `src/ml/risk/core_risk.py`, `src/ml/risk/risk_engine.py` | Rolling VaR/CVaR/drawdown analytics | wrap / refactor | Practical reusable formulas that match Phase 1 well |
| `src/ml/artifacts.py` | Manifest and artifact helpers | wrap / refactor | Existing manifest conventions are useful, but Phase 1 needs run-level manifests |
| `src/reporting/*` | Daily brief / ranked-prediction reporting | replace for Phase 1 reporting | Current reporting layer is not benchmark/manifests oriented |

## Canonical Entry Points Found During Audit

- Data preparation: `scripts/train_ml_tickers.py --prepare-only`, backed by `DualModelTrainer.prepare_ticker_data(...)`
- Model training: `scripts/train_ml_tickers.py`, backed by `DualModelTrainer.train_explicit_split(...)`
- Evaluation: `scripts/run_backtest_forward_return.py`, `scripts/run_strategy_backtest.py`, `scripts/run_walkforward_all_models_stacking_eval.py`
- Reporting: workflow-specific writers inside `src/ml/backtest/*` and `scripts/run_walkforward_all_models_stacking_eval.py`

## Phase 1 Architectural Direction

- Keep the current data adapter, feature engineering, feature registry, context loaders, and core risk formulas.
- Add a contract-first layer under `src/core`, `src/forecast`, `src/risk`, `src/ensemble`, `src/strategy`, `src/evaluation`, and `src/reporting`.
- Bridge into stable existing modules where they already solve the data/features problem well.
- Avoid routing the new Phase 1 architecture through the full legacy `DualModelTrainer` stack when a smaller wrapper is safer.
