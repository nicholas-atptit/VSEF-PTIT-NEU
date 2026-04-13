# Prompt Run: Quantitative Architecture Expansion (Phases 1-6)

## Prompt Intent
Implement a next-generation quantitative forecasting architecture expanding the existing codebase. The expansion integrates statistical models (SARIMAX, ETS), boosting models (XGBoost, LightGBM) with optional Optuna tuning, and a conservative time-series safe stacking layer. A Monte Carlo Risk Layer is also constructed out-of-band to generate VaR and CVaR post-inference without polluting predictive ML capabilities. Strict back-compatibility with the ML pipeline along with zero look-ahead bias is demanded.

## Actual Outcome
Successfully executed across 6 distinct phases while retaining total backwards compatibility for existing manifest-driven workflows and tests. 
The system now safely supports explicit capabilities evaluation (`get_model_capabilities()`) for 5 new algorithm frameworks avoiding blind initialization failures. Stacking has been integrated safely via out-of-fold `TimeSeriesSplit` strategies strictly preventing future data leakage, degrading gracefully if optional packages like LightGBM are missing locally. The Monte Carlo script guarantees deterministic reproducibility using deterministic NumPy generation. This initial version (v1) provides a robust foundation for risk-aware forecasting.

## Files Created
- `src/ml/models/sarimax.py`
- `src/ml/models/ets.py`
- `src/ml/models/xgboost_model.py`
- `src/ml/models/lightgbm_model.py`
- `src/ml/models/stacking.py`
- `src/ml/tuning.py`
- `src/ml/risk.py`
- `tests/ml/test_statistical_models.py`
- `tests/ml/test_boosting_models.py`
- `tests/ml/test_tuning.py`
- `tests/ml/test_deep_learning_models.py`
- `tests/ml/test_stacking.py`
- `tests/ml/test_risk.py`

## Files Modified
- `src/ml/models/base.py`
- `src/ml/models/cart.py`
- `src/ml/models/lstm.py`
- `src/ml/models/factory.py`
- `src/ml/artifacts.py`
- `tests/ml/test_model_registry.py`

## Key Code Changes
- Implemented `@classmethod get_model_capabilities()` on `BaseModel` to safely expose algorithm structure and requirement constraints upstream to `DualModelTrainer` and factory logic.
- Constructed generalized wrappers spanning StatsModels (SARIMAX/ETS) tracking regression targets and explicitly extracting probability metrics for classification via threshold constraints.
- Generated native XGBoost and LightGBM models exposing optional dependency fallbacks tracking `HAS_XGBOOST` natively without breaking core application paths locally.
- Injected `tuning_backend` metadata tracking into boosting parameters supporting fully decoupled `test_tuning.py` architecture.
- Added array slicing fallback safeguards when `predict_proba()` attempts evaluation on fully homogenious class blocks during CV folding inside Stacking meta-feature aggregation.

## Functions / Classes / Scripts Added or Updated
- `BaseModel.get_model_capabilities`
- `SarimaxModel`
- `EtsModel`
- `XgboostModel`
- `LightgbmModel`
- `StackingModel`
- `optimize_hyperparameters` (in `tuning.py`)
- `MonteCarloRiskSimulator`
- `factory.create_model` (registry updated)

## Algorithms / Methods / Logic Introduced
- Implemented: SARIMAX forecasting
- Implemented: ETS forecasting
- Implemented: XGBoost
- Implemented: LightGBM
- Implemented: TimeSeries Stacking Ensemble (Meta-model: LogisticRegression/Ridge)
- Implemented: TimeSeriesSplit Optuna hyperparameter tuning
- Implemented: Expected Shortfall (CVaR) and Value-At-Risk Monte Carlo simulations
- Referenced only: Geometric Brownian Motion assumptions (used structurally in risk sampling via NumPy)
- Planned only: Deep learning inclusion in stacking ensemble

## Config / CLI / Environment Changes
- Optuna, XGBoost, and LightGBM treated rigorously as optional packages. Run-time `ImportError` protections catch unavailability natively routing executions safely to fallback modes without halting pipelines.
- No direct `.env` overrides added during this phase.

## Storage / Persistence Impact
- `joblib` artifacts formalized for `.joblib` inclusion into manifest generation explicitly mapping `tuning_backend` and fallback methodologies.
- The `bilstm`/`lstm` arrays persist unchanged as `.pt` bundles retaining legacy execution compatibility.

## Compatibility Notes
- Stacking heavily demands minimum evaluation splits (`n_samples > n_splits * 2`). Automatic fallback mapping `max(2, n_samples // 10)` guarantees minimal continuity execution during edge-cases, meaning short series won't crash the stack logic entirely, just truncate CV length safely. Tests mapped in `tests/ml/test_stacking.py`.

## Future Improvements
- Refactor `src/ml/risk.py` into a specialized package.
- Support Student-t or GARCH-based risk distributions.
- Portfolio sizing logic integration.

## Tests / Validation
- Rigorous integration coverage extending from `BaseModel` contract enforcement down scaling up to CLI simulations successfully predicting outputs locally across `FPT` samples using multi-model definitions `(algorithms="xgboost,lightgbm")`.
- `test_risk.py` enforces deterministic output constraints testing simulation parity via constant seeds (`42`).

## Remaining Gaps
- Expanding CLI parameters directly in `scripts/train_ml_tickers.py` to unlock configurable hyperparameter strings directly out-of-the-box remains mapped to Phase 7 scheduling.
- Deep Learning ensembles remain locked out of V1 stack.
