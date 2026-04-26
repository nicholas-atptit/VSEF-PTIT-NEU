# VSEF ML Test Hardening Notes

Date: 2026-04-26

Branch scope: stabilize the existing ML workflow and add limited Linear/Ridge/Lasso diagnostics. This branch does not add GRU, TCN, Transformer, TFT, N-BEATS, N-HiTS, CatBoost, EGARCH, GJR-GARCH, DQN, PPO, or A2C.

## Initial Failure Summary

The initial `python -m pytest tests/ml -q` run failed 11 tests, with 90 passed and 2 skipped.

Failure groups:

| Failure group | Affected tests | Root cause |
| --- | --- | --- |
| No model bundles trained for `TEST` | Phase 2 manifest and inference metadata tests | The synthetic OHLCV fixture was too short and too monotonic after the current indicator warmup and split requirements. Training skipped usable bundles because the remaining target distribution was not valid enough for the short-horizon classification task. |
| Missing context features | Walk-forward all-model stacking test | Training used explicit synthetic market context, but inference feature rebuilding did not pass the same `context_sources`. The prediction frame therefore missed context features such as `m_ret_20d`, `market_return_60d`, and `m_ret_5d`. |
| Feature-count mismatch during benchmark evaluation | Risk tuning, stress test, and system benchmark tests | The benchmark evaluators rebuilt feature matrices with the algorithm-level feature union, then fed those matrices into task-specific models trained on `feature_columns_by_task`. Trend models received more columns than they were trained on. |
| Metadata expectation drift | Phase 2 hardening tests after fixture repair | The active trainer feature engine advertises deterministic local pandas/numpy technical-indicator computation. The test still expected the older external dependency behavior string. |
| Pytest cache warning | All local pytest runs | `.pytest_cache` could not be written because the repository cache path is access denied in this environment. This is not an application-code failure. |

## Fixes Applied

- Preserved context-source parity in walk-forward inference by passing the same `context_sources` into `compute_features_for_ticker`.
- Added backward-compatible market context aliases `m_ret_5d` and `m_ret_20d` in the context feature layer. These are trailing rolling means and use only data available at or before time `t`.
- Updated system benchmark, stress test, and risk tuning evaluation paths to rebuild trend and return problems with task-specific feature columns from `feature_columns_by_task`.
- Updated affected alignment tests to calculate expected strategy metrics from task-specific trend and return feature matrices.
- Strengthened the phase 2 synthetic OHLCV fixture so it has enough rows and nontrivial target variation after warmup and time-series splits.
- Updated phase 2 metadata assertions to expect the current `local_deterministic_numpy_pandas_computation` feature-generation behavior.

## Linear Model Diagnostics

Linear/Ridge/Lasso remain shadow or interpretable baseline-style forecast models. They were not promoted to research core or decision core.

Added a small coefficient diagnostics surface for the forecast `LinearForecastModel`, `RidgeForecastModel`, and `LassoForecastModel`:

- selected feature names
- intercept
- per-feature coefficient
- coefficient sign
- coefficient magnitude
- coefficient count and nonzero count
- explicit note that fold-level coefficient stability is not available in the current single-fit forecast model contract

This is diagnostic metadata only. It is not evidence that linear models outperform boosting models.

## Validation

Commands run:

```bash
python -m pytest tests/ml -q
python -m pytest tests/quant_core -q
python -m pytest tests/phase1/test_forecast_contracts.py -q
```

Observed results:

- `tests/ml`: 101 passed, 2 skipped
- `tests/quant_core`: 15 passed
- `tests/phase1/test_forecast_contracts.py`: 3 passed

The `.pytest_cache` access warning remains environment-level and was not patched in application code.

## Remaining Risks

- The benchmark and stress-test workflows still rely on synthetic fixtures for some assertions; passing tests should not be interpreted as trading-performance evidence.
- Linear/Ridge/Lasso coefficient diagnostics are single-fit diagnostics. Fold-level coefficient stability still needs an explicit walk-forward diagnostics workflow before it can be used for stronger conclusions.
- Context feature aliases are backward-compatible support columns. Longer-term feature governance should continue to prefer canonical feature names where possible.
- Sequence model hardening, GRU/TCN/Transformer-family work, GARCH-family extensions, and RL policy evaluation remain future tasks.

## Recommended Next Task

Add fold-level Linear/Ridge/Lasso evaluation diagnostics in the walk-forward evaluation path:

- train linear models on the same governed feature sets used by boosting comparators
- record coefficient stability across folds
- report feature sign consistency and magnitude dispersion
- keep linear models as interpretable baselines and sanity checks
- avoid any claim of production trading performance

## Follow-Up Completed: Linear Fold Diagnostics

The follow-up branch `vsef-linear-fold-diagnostics` added fold-level coefficient diagnostics for Linear Regression, Ridge, and Lasso in the walk-forward all-model evaluation path.

New outputs:

- `csv/linear_coefficient_diagnostics.csv`
- `csv/linear_coefficient_stability_summary.csv`

These outputs summarize coefficient signs, magnitudes, intercepts, nonzero counts, and sign consistency across rolling folds. Linear/Ridge/Lasso remain interpretable shadow/baseline models. The diagnostics are governance and feature-stability evidence only; they are not trading-performance evidence and do not prove causality.

## Follow-Up Completed: Feature Importance Diagnostics

The follow-up branch `vsef-feature-importance-diagnostics` added fold-level feature-importance diagnostics for supported tree and boosting models in the walk-forward all-model evaluation path.

New outputs:

- `csv/feature_importance_diagnostics.csv`
- `csv/feature_importance_stability_summary.csv`
- `csv/linear_vs_importance_feature_comparison.csv`

These outputs compare supported CART, XGBoost, and LightGBM feature-importance stability against the existing Linear/Ridge/Lasso coefficient stability diagnostics. The comparison is for model governance and interpretability only. It does not prove causality, does not establish trading performance, and does not change Linear/Ridge/Lasso governance status.

## Follow-Up Completed: Feature Governance Review

The follow-up branch `vsef-feature-leakage-governance` added a conservative rule-based review layer for features appearing in linear coefficient stability and tree/boosting importance diagnostics.

New output:

- `csv/feature_governance_review.csv`

The review flags features for timing, redundancy, target-derived, potential-leakage, or unknown-metadata review. It does not remove features automatically, does not prove leakage unless the implementation clearly uses future information, and does not establish trading performance.

## Follow-Up Completed: Context Availability Metadata

The follow-up branch `vsef-context-availability-metadata` added support metadata for breadth and foreign-flow context joins:

- `breadth_context_available`
- `breadth_context_source_date`
- `breadth_context_missing`
- `foreign_flow_context_available`
- `foreign_flow_context_source_date`
- `foreign_flow_context_missing`

These columns distinguish measured zero values from missing-context fallback values and are excluded from active model feature columns. They improve governance transparency only; they do not prove trading performance or remove leakage-review responsibility.

## Follow-Up Completed: Context Coverage Diagnostics

The follow-up branch `vsef-context-coverage-diagnostics` added walk-forward CSV summaries for context availability:

- `csv/context_coverage_diagnostics.csv`
- `csv/context_coverage_summary.csv`

These outputs quantify breadth and foreign-flow missing-context rates by ticker, fold, and horizon. They are diagnostic governance outputs only and do not change model governance status or trading-performance claims.
