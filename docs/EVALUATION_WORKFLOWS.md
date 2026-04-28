# Evaluation Workflows

## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Workflow guide |
| Created / authored | Wednesday, 2026-04-15 11:19:48 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-04-28 21:37:10 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | not specified |
| Commit | `1bbb75a793c5ea30a26ffef7860aadd06306a640` |
| Timestamp source | Git history |

This document describes the major evaluation workflows implemented in the repository.

Each section covers:

- purpose
- input data
- main outputs
- key metrics
- artifact location

## 1. Fixed-Window Real-Data Backtest

Purpose:

- Evaluate a single close-level forecasting path on real vnstock data.
- Compare it against a naive previous-close baseline.

Input data:

- daily OHLCV from `vnstock`
- tickers supplied on the CLI, typically `DGC`, `ACB`, `MWG`, `HPG`
- fixed train/eval split

Main outputs:

- `predicted_vs_actual.csv`
- `metrics_summary.csv`
- `fetch_summary.csv`
- `training_summary.csv`
- `run_config.json`
- per-ticker charts
- trained model manifests

Key metrics:

- `MAE`
- `RMSE`
- `MAPE`
- `directional_accuracy`
- baseline win flags against naive previous close

Artifact location:

- `artifacts/backtest/`

## 2. Fixed-Window Model Comparison

Purpose:

- Compare model families on the same fixed-window real-data backtest.

Input data:

- same OHLCV inputs as the fixed-window backtest
- same train/eval split
- algorithm list such as `cart,xgboost,lightgbm,sarimax,ets`

Main outputs:

- `predicted_vs_actual.csv`
- `model_comparison.csv`
- `overall_model_ranking.csv`
- `fetch_summary.csv`
- `training_summary.csv`
- `run_config.json`

Key metrics:

- `MAE`
- `RMSE`
- `MAPE`
- `directional_accuracy`
- `beats_naive_baseline`
- aggregate rank columns

Artifact location:

- `artifacts/backtest_model_comparison/`

## 3. Forward-Return Regression Evaluation

Purpose:

- Evaluate multi-horizon forward-return regression for `3d`, `5d`, and `20d`.

Input data:

- daily OHLCV from `vnstock`
- train/eval windows enforced on realized `target_date`
- horizon list
- algorithm list

Main outputs per horizon:

- `predicted_vs_actual.csv`
- `metrics_summary.csv`
- `model_comparison.csv`
- `overall_model_ranking.csv`
- `fetch_summary.csv`
- `training_summary.csv`
- `run_config.json`
- charts

Global outputs:

- `overall_horizon_summary.csv`
- `overall_horizon_ranking.csv`

Key metrics:

- `MAE`
- `RMSE`
- `MAPE`
- `directional_accuracy`
- `beats_naive_baseline`

Artifact location:

- `artifacts/backtest_forward_return/`

## 4. Strategy Evaluation

Purpose:

- Evaluate whether forward-return forecasts can produce usable threshold-based paper strategies after costs.

Input data:

- saved forward-return artifacts
- execution prices from the same vnstock data path
- thresholds, transaction fees, and slippage assumptions

Main outputs per horizon:

- `trades.csv`
- `strategy_metrics.csv`
- `equity_curve.csv`
- `portfolio_metrics.csv`
- `fetch_summary.csv`
- `run_config.json`

Global outputs:

- `overall_strategy_ranking.csv`
- `model_horizon_threshold_summary.csv`
- cross-model charts

Key metrics:

- `total_return`
- `annualized_return`
- `win_rate`
- `profit_factor`
- `max_drawdown`
- `sharpe_ratio`
- `sortino_ratio`
- `number_of_trades`
- `average_trade_return`
- `exposure_ratio`
- `turnover`
- benchmark comparison flags versus buy-and-hold and flat strategy

Artifact location:

- `artifacts/strategy_backtest/`

Important note:

- This remains an analysis-only backtest layer, not an execution engine.

## 4A. Signal-Effectiveness Backtest

Purpose:

- Convert saved prediction outputs into diagnostic `BUY`, `HOLD`, and `AVOID` labels.
- Evaluate whether strict BUY rules have useful precision after explicit cost and slippage assumptions.

Input data:

- saved fixed forward-return or walk-forward prediction CSVs
- no live provider access
- threshold, cost, slippage, and success-definition grids

Main outputs:

- `signal_rows.csv`
- `buy_precision_by_model_horizon.csv`
- `precision_coverage_frontier.csv`
- `signal_effectiveness_summary.csv`
- `strategy_proxy_metrics.csv`
- `benchmark_comparison.csv`
- `run_metadata.json`

Key metrics:

- BUY precision
- BUY recall, when computable
- average and median realized return after BUY
- win rate after BUY
- net average return after estimated cost/slippage
- cumulative simple signal return
- profit factor
- max drawdown on the signal-only proxy curve
- turnover proxy

Artifact location:

- caller-specified `--output-dir`

Important note:

- This layer is a forecast-to-signal diagnostic bridge, not trading-performance proof.

## 4B. Held-Out Threshold Selection

Purpose:

- Select signal thresholds on an earlier prediction-date window.
- Apply selected thresholds unchanged to a later held-out window.
- Test whether BUY precision targets such as 60%, 65%, and 70% survive out of selection.

Input data:

- saved prediction CSVs accepted by the signal-effectiveness layer
- explicit selection and held-out test date windows
- threshold, cost, slippage, success-definition, and minimum-count grids

Main outputs:

- `selected_thresholds.csv`
- `heldout_buy_precision.csv`
- `threshold_selection_trace.csv`
- `heldout_signal_rows.csv`
- `precision_target_pass_fail.csv`
- `heldout_strategy_proxy_metrics.csv`
- `run_metadata.json`

Key metrics:

- selection-period BUY precision and BUY count
- held-out BUY precision and BUY count
- held-out BUY recall, when computable
- held-out net average return after BUY
- held-out profit factor
- held-out max drawdown proxy
- precision target pass/fail flags

Artifact location:

- caller-specified `--output-dir`

Important note:

- This is required before treating descriptive BUY precision frontiers as policy candidates.

## 4C. Rolling Held-Out Threshold Selection

Purpose:

- Repeat held-out threshold selection across multiple chronological folds.
- Test whether BUY precision targets such as 60%, 65%, and 70% survive beyond one split.
- Optionally report regime-conditioned BUY precision when prediction rows already contain safe regime labels.

Input data:

- saved prediction CSVs accepted by the signal-effectiveness layer
- inline `--rolling-splits` definitions or a JSON/CSV `--rolling-splits-file`
- existing regime columns only: `regime`, `market_regime`, `regime_label`, or `market_state`

Main outputs:

- `rolling_selected_thresholds.csv`
- `rolling_heldout_buy_precision.csv`
- `rolling_threshold_selection_trace.csv`
- `rolling_heldout_signal_rows.csv`
- `rolling_precision_target_pass_fail.csv`
- `rolling_strategy_proxy_metrics.csv`
- `threshold_stability_summary.csv`
- `regime_buy_precision_summary.csv`, when regime labels exist
- `regime_precision_stability_summary.csv`, when regime labels exist
- `run_metadata.json`

Key metrics:

- selected threshold values by fold
- threshold stability level
- held-out BUY precision mean, min, max, and standard deviation across folds
- precision target pass rates across folds
- regime-specific BUY precision and 70% pass rate, when labels exist

Artifact location:

- caller-specified `--output-dir`

Important note:

- Rolling held-out testing is required before treating a BUY precision target as stable. Regime-conditioned evaluation is needed before deciding whether BUY rules should be active in all market states or only in favorable regimes.

## 4D. Signal Regime Join

Purpose:

- Attach precomputed safe regime labels to saved prediction rows before signal-effectiveness diagnostics.
- Record join coverage and source-review flags.
- Avoid fabricating regimes inside the signal layer.

Input data:

- saved prediction CSVs with `prediction_date`
- precomputed regime label CSVs with a configurable date column
- optional ticker column when using ticker/date joins

Main outputs:

- enriched prediction CSV with `regime`
- join coverage summary JSON or CSV

Supported join modes:

- `date`
- `ticker_date`

Governance checks:

- matched and unmatched prediction rows
- duplicate regime keys
- suspicious future-looking source column names
- regime label distribution
- classification as `safe_if_regime_source_is_trailing`, `requires_source_review`, or `schema_invalid`

Important note:

- Regime-conditioned BUY precision requires safe precomputed regime labels joined to prediction rows. The join layer does not infer regimes and does not prove leakage absence by itself; it records coverage and source-review flags.

## 5. Dual-Task Evaluation

Purpose:

- Evaluate return regression and profit/loss classification in parallel.

Input data:

- same daily OHLCV source
- same fixed split
- same feature pipeline
- cost assumptions for classification labels

Main outputs:

- regression branch outputs under `regression/{horizon}/`
- classification branch outputs under `classification/{horizon}/`
- joined cross-task summary tables
- model artifacts under `models/`

Key metrics:

Regression:

- `MAE`
- `RMSE`
- `MAPE`
- `directional_accuracy`

Classification:

- `accuracy`
- `precision`
- `recall`
- `f1`
- `roc_auc`
- `positive_class_precision`
- confusion-matrix counts

Artifact location:

- `artifacts/dual_task/`

## 6. Combined-Signal Evaluation

Purpose:

- Evaluate whether combining `predicted_return` and `predicted_profit_probability` produces better research signals than either alone.

Input data:

- `artifacts/dual_task/summary/joined_regression_classification_evaluation.csv`
- threshold grids
- weights for the weighted-linear score

Main outputs per horizon:

- `combined_signal_table.csv`
- `combined_bucket_summary.csv`
- `combined_ranking_summary.csv`
- `probability_calibration_summary.csv`
- `run_config.json`
- charts

Global outputs:

- `overall_combined_signal_summary.csv`
- `cross_horizon_combined_ranking.csv`

Key metrics:

- average actual return by bucket
- profit rate by bucket
- top-k average actual return
- top-k profit rate
- lift versus base rate
- strong-positive precision
- profitable-case recall from selected buckets
- calibration error summaries

Artifact location:

- `artifacts/combined_signal/`

## 7. Regime-Aware Evaluation

Purpose:

- Re-evaluate saved regression, classification, and combined-signal outputs under explicit `bull`, `bear`, and `sideway` market states.

Input data:

- saved dual-task outputs
- saved combined-signal outputs
- benchmark history, usually `VNINDEX`

Main outputs per horizon:

- `regime_labeled_signal_table.csv`
- `regression_by_regime.csv`
- `classification_by_regime.csv`
- `combined_signal_by_regime.csv`
- `ranking_by_regime.csv`
- `calibration_by_regime.csv`
- `run_config.json`
- regime charts

Global outputs:

- `overall_regime_summary.csv`
- `regime_model_horizon_ranking.csv`
- `regime_combined_method_ranking.csv`

Key metrics:

Regression by regime:

- `MAE`
- `RMSE`
- `directional_accuracy`
- average actual and predicted return

Classification by regime:

- `accuracy`
- `precision`
- `recall`
- `f1`
- `positive_class_precision`
- realized profit rate

Combined signals by regime:

- bucket average return
- bucket hit rate
- strong-positive precision
- top-k ranking quality
- regime-specific calibration

Artifact location:

- `artifacts/regime_aware_analysis/`

## 8. Walk-Forward Robustness Evaluation

Purpose:

- Test whether findings repeat across multiple chronological folds instead of a single holdout window.

Input data:

- repeated train/eval windows generated from the same ticker universe and data path
- same forward-return, dual-task, combined-signal, and regime-aware components reused inside each fold

Main fold outputs:

- `fold_config.json`
- `fold_summary.csv`
- `regime_summary.csv`
- `model_ranking.csv`
- `combined_method_ranking.csv`
- `joined_evaluation_sample.csv`

Global outputs:

- `fold_overview.csv`
- `model_stability_summary.csv`
- `horizon_stability_summary.csv`
- `regime_stability_summary.csv`
- `combined_method_stability_summary.csv`
- `overall_robustness_report.csv`
- summary charts

Key metrics:

- fold win count
- win rate
- average rank
- rank standard deviation
- regime consistency
- stability level labels

Artifact location:

- `artifacts/walk_forward_regime_robustness/`

## 9. Meta-Selector Evaluation

Purpose:

- Evaluate whether regime-conditioned adaptive candidate selection can beat fixed setups using prior-fold information only.

Input data:

- walk-forward fold summaries
- per-fold model and combined-method rankings
- regime labels from the regime-aware layer

Main fold outputs:

- `selected_candidates.csv`
- `selector_performance.csv`
- `selector_vs_baselines.csv`
- `regime_selection_summary.csv`
- `fold_config.json`

Global outputs:

- `meta_selector_overview.csv`
- `selector_stability_summary.csv`
- `selector_regime_summary.csv`
- `selector_vs_baselines_summary.csv`
- `overall_meta_selector_report.csv`
- summary charts

Key metrics:

- average selected-row actual return
- profit-label hit rate
- top-k average return
- top-k profit rate
- selector stability across folds
- fallback usage

Artifact location:

- `artifacts/meta_selector/`

## 10. Benchmark Audit and Context-Conditioned Selector Evaluation

Purpose:

- Audit whether selector baselines and summary rows are being computed correctly.
- Evaluate richer context-conditioned selectors against both fixed baselines and regime-based selectors.

Input data:

- walk-forward artifacts
- meta-selector artifacts
- saved prediction-state and market-state context

Audit outputs:

- `benchmark_audit_report.csv`
- `baseline_definition_check.csv`
- `entity_comparison_trace.csv`
- `suspicious_equal_summary_rows.csv`
- `run_config.json`

Context-selector fold outputs:

- `selected_candidates.csv`
- `context_features_used.csv`
- `selector_performance.csv`
- `selector_vs_baselines.csv`
- `selector_vs_regime_selector.csv`
- `fold_config.json`

Context-selector global outputs:

- `context_selector_overview.csv`
- `context_selector_stability_summary.csv`
- `context_selector_vs_baselines_summary.csv`
- `context_selector_vs_regime_summary.csv`
- `context_feature_importance_summary.csv`
- `overall_context_selector_report.csv`
- summary charts

Key metrics:

- benchmark recomputation deltas
- context selector average actual return
- context selector top-k ranking quality
- comparison deltas versus regime selectors
- fallback rate
- feature-importance summaries for the meta-score mode

Artifact locations:

- `artifacts/meta_selector_audit/`
- `artifacts/context_meta_selector/`

## Summary

The evaluation stack now answers different questions at different layers:

- `backtest` and `backtest_model_comparison`: can a model fit the fixed holdout at all?
- `backtest_forward_return`: which horizon and model family look strongest on returns?
- `strategy_backtest`: do those forecasts survive a simple cost-aware paper strategy?
- `signal_effectiveness`: can saved forecasts support high-precision BUY diagnostics under strict transparent rules?
- `heldout_threshold_selection`: do selected BUY thresholds survive on a later held-out period?
- `rolling_heldout_threshold_selection`: do selected BUY thresholds and precision targets remain stable across chronological folds?
- `signal_regime_join`: can safe precomputed regime labels be attached to prediction rows for regime-conditioned signal diagnostics?
- `dual_task`: can the system forecast both return and tradability?
- `combined_signal`: does combining both outputs help signal quality?
- `regime_aware_analysis`: does market state change what works?
- `walk_forward_regime_robustness`: do those findings repeat over time?
- `meta_selector` and `context_meta_selector`: can adaptive selection outperform fixed setups without leakage?
