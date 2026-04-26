# VSEF Model Gap Audit

Audit date: 2026-04-26

This audit was prepared before adding any new model implementation. It separates implemented and governed model surfaces from legacy, placeholder, or roadmap-only references.

## Scope Checked

- `src/ml/models/`
- `src/forecast/`
- `src/ml/sequence_dataset.py`
- `src/ml/trainer.py`
- `src/ml/backtest/`
- `src/regime/`
- `src/ml/regime/`
- `src/risk/`
- `src/ml/risk/`
- `tests/`

## Registry Snapshot

The active local registry import reported:

- `src.ml.models.factory.supported_algorithms()`: `bilstm`, `cart`, `ets`, `lightgbm`, `lstm`, `sarimax`, `stacking`, `xgboost`
- `src.forecast.registry.supported_forecast_models(run_mode="full_forecast")`: `lightgbm`, `xgboost`, `random_forest`, `ets`, `sarimax`, `naive`, `moving_average`, `linear`, `ridge`, `lasso`
- `src.forecast.registry.supported_forecast_models(run_mode="research_core")`: `lightgbm`, `xgboost`, `random_forest`, `ets`, `sarimax`
- `src.forecast.registry.supported_forecast_models(run_mode="decision_core")`: `lightgbm`, `xgboost`, `random_forest`

## Models Actually Implemented

### `src/ml/models/`

Implemented trainable model classes in the ML workflow registry:

- CART: `src/ml/models/cart.py`
- LSTM: `src/ml/models/lstm.py`
- BiLSTM: `src/ml/models/bilstm.py`
- SARIMAX: `src/ml/models/sarimax.py`
- ETS: `src/ml/models/ets.py`
- XGBoost: `src/ml/models/xgboost_model.py`
- LightGBM: `src/ml/models/lightgbm_model.py`
- Stacking: `src/ml/models/stacking.py`

Notes:

- `src/ml/models/factory.py` registers the algorithms above.
- `src/ml/models/` also contains ORM/domain entities such as agent, company, price, signal, and watchlist models. Those are not forecasting algorithms.
- Linear Regression, Ridge, Lasso, and Random Forest are not registered in `src/ml/models/factory.py`.

### `src/forecast/`

Implemented governed forecast models:

- Naive: `src/forecast/statistical/naive.py`
- Moving Average: `src/forecast/statistical/moving_average.py`
- SARIMAX: `src/forecast/statistical/sarimax.py`
- ETS: `src/forecast/statistical/ets.py`
- Linear Regression: `src/forecast/ml/linear.py`
- Ridge: `src/forecast/ml/ridge.py`
- Lasso: `src/forecast/ml/lasso.py`
- Random Forest: `src/forecast/ml/random_forest.py`
- XGBoost: `src/forecast/ml/xgboost_model.py`
- LightGBM: `src/forecast/ml/lightgbm_model.py`

Notes:

- `src/forecast/registry.py` governs run-mode selection and model roles.
- Linear, Ridge, and Lasso are present in the full forecast surface but marked as shadow-only rather than active research-core models.
- Naive and Moving Average are baseline-only models.

### `src/ml/sequence_dataset.py`

Implemented sequence utilities:

- rolling sequence construction
- explicit target row alignment
- range selection by target index
- latest-window construction for inference

Notes:

- These utilities support LSTM and BiLSTM through `src/ml/trainer.py`.
- They do not yet imply GRU, TCN, Transformer, TFT, N-BEATS, or N-HiTS support.

### `src/ml/trainer.py`

Implemented trainer behavior:

- manifest-driven training and inference through `DualModelTrainer`
- target generation for multiple horizons
- model creation through `src.ml.models.factory`
- sequence handling for `lstm` and `bilstm`
- risk and regime feature integration
- artifact metadata and reload paths

Notes:

- `SEQUENCE_ALGORITHMS` is currently `{"lstm", "bilstm"}`.
- Adding GRU would require registry, artifact-extension, trainer sequence inclusion, and tests.

### `src/ml/backtest/`

Implemented research/evaluation workflows include:

- fixed-window real-data backtest
- model comparison
- forward-return backtest
- dual-task backtest
- strategy backtest
- combined-signal analysis
- regime-aware analysis
- walk-forward regime robustness
- walk-forward all-model stacking
- regime-conditioned and context-conditioned selector layers

Notes:

- Default workflow algorithm lists generally emphasize `cart`, `xgboost`, `lightgbm`, `sarimax`, and `ets`.
- LSTM and BiLSTM are supported by the model factory and trainer but are not the default path for most current research workflows.

### `src/regime/` and `src/ml/regime/`

Implemented regime logic:

- Markov-switching bull, bear, and sideway regime model with threshold fallback in `src/regime/markov_switching.py`
- regime probability and label helpers in `src/regime/labels.py`
- rule-based NORMAL, HIGH_VOL, and CRISIS detector in `src/ml/regime/regime_detector.py`

Notes:

- The regime layer is implemented but still needs stability analysis across folds and more explicit handling of sparse bear-regime samples.

### `src/risk/` and `src/ml/risk/`

Implemented risk logic:

- historical VaR/CVaR
- Monte Carlo VaR/CVaR
- drawdown metrics
- GARCH volatility and tail-risk layer
- rolling CoVaR and Delta-CoVaR estimators
- risk engine feature construction

Notes:

- EGARCH and GJR-GARCH are not implemented in the current governed risk surface.
- VaR/CVaR outputs should remain conservative research metrics, not guaranteed loss bounds.

## Models Only Mentioned, Legacy, or Not Governed

The following should not be claimed as implemented VSEF model coverage yet:

- GRU: no current model implementation, registry entry, or tests
- TCN: no current model implementation, registry entry, or tests
- Transformer: no current governed forecasting implementation, registry entry, or tests
- TFT: legacy experimental script exists under `src/ml/training_pipeline/train_tft.py`, but the directory README says these scripts are not maintained or used by the active application endpoints
- N-BEATS / N-HiTS: no current model implementation, registry entry, or tests
- CatBoost: only a commented reference appears in `src/ml/training/baseline_model.py`
- EGARCH / GJR-GARCH: no current risk model implementation
- DQN: no current implementation
- PPO: legacy/experimental scripts and allocator placeholders exist, but PPO is not part of the governed core forecast, risk, regime, or backtest model surface
- A2C: no current implementation

## Registered But Not Used in Active Default Workflows

- `linear`, `ridge`, and `lasso` are registered in `src/forecast/registry.py` for the full forecast surface, but they are shadow-only and excluded from the current research-core and decision-core run modes.
- `naive` and `moving_average` are registered in `src/forecast/registry.py` as baseline-only models.
- `random_forest` is registered in the governed `src/forecast` surface but not in `src/ml/models/factory.py`.
- `lstm` and `bilstm` are registered in `src/ml/models/factory.py` and supported by `DualModelTrainer`, but most default backtest configurations do not include them.
- `stacking` is registered in `src/ml/models/factory.py`; current workflow usage is primarily as a research/ensemble layer rather than a default standalone production-style forecaster.
- `src/ml/training_pipeline/` contains legacy experimental TFT, CNN, PPO, and ONNX scripts that are not registered in the active ML model factory or governed forecast registry.

## Tests That Should Come Before New Implementations

Before adding new model families:

- Linear/Ridge/Lasso should receive stronger tests for shared feature sets, fold-wise diagnostics, coefficient stability, and comparison against boosting baselines.
- LSTM/BiLSTM should receive stronger trainer-level tests for sequence preparation, prediction shape, persistence, classification/regression parity, and evaluation consistency across folds.
- GRU should be added only with capability, save/load, predict-shape, sequence-dataset, and trainer integration tests.
- TCN should wait until GRU and sequence evaluation tests are stable.
- Transformer, TFT, N-BEATS, and N-HiTS should wait until the sequence-evaluation surface is demonstrably robust.
- EGARCH/GJR-GARCH should wait for tests that isolate the current `arch` dependency, volatility forecast outputs, and conservative tail-risk interpretation.
- Regime tests should cover Markov-switching versus threshold benchmark labels, stability across walk-forward folds, and sparse bear-regime handling.
- RL tests should cover environment leakage, transaction costs, slippage, position limits, turnover penalties, reward construction, and comparisons against threshold and buy-and-hold baselines.

## Audit Conclusion

With the current codebase, **VSEF – Vietnam Stock Evaluation and Forecasting Framework** is a defensible project name. The codebase supports real-data evaluation, multiple forecast families, risk-aware analysis, regime-aware analysis, walk-forward validation, selector layers, and strategy backtesting.

The current codebase does not justify claims such as **state-of-the-art deep forecasting platform** or **finished live-trading execution system**. The next branch should improve evaluation depth and model governance before adding large new algorithms.
