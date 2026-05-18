# VSEF 1000-Seed Smoke Stability Report

## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Repeated-seed smoke stability report |
| Created / authored | Thursday, 2026-04-30 20:05:31 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Source output root | `artifacts\quant_core_repeated_seed_1000_smoke` |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `4e317fba24f836209488b0b6e23729a04bbe45dd` |
| Timestamp source | Local report generation from saved 1000-seed smoke artifacts |
| Run mode | `research_core` |
| Preset | `smoke` |
| Seed count | `1000` |
| Status | Complete report from saved artifacts |

## Executive Summary

This report summarizes the saved 1000-seed Quant Core smoke output under `artifacts\quant_core_repeated_seed_1000_smoke`. The output root contains 1000 seed folders, and 1000 seed folders contain the minimum completion files: `run_manifest.json`, `forecast_summary.csv`, and `model_execution_log.csv`. The run should therefore be interpreted as complete for the smoke diagnostic surface.

Repeated-seed training is a stability diagnostic. It must not be used to select the single best random seed based on test-period performance.

The primary finding is stability, not improved predictive accuracy. The aggregate stability files show identical or numerically indistinguishable values across the 1000 saved seeds for this bounded `smoke` / `research_core` setup. All standard deviations in the saved stability tables are effectively zero at normal reporting precision; the largest observed standard deviation is 0, which is numerical floating-point noise rather than economically meaningful seed dispersion. This supports an orchestration-level conclusion that the smoke workflow was reproducible across seeds in the saved outputs. It does not establish market edge, production readiness, or future trading profitability.

## Purpose of the Diagnostic

Repeated-seed evaluation measures whether a workflow's outputs are sensitive to random seeds. It is useful for detecting instability from randomized model training, stochastic initialization, randomized sampling, or non-deterministic policy behavior. In governance terms, seed stability should be read as an uncertainty diagnostic: if dispersion is large, claims should be weakened; if dispersion is small, the workflow is more reproducible under the tested setup.

The diagnostic must not be used to select the best seed. Selecting a single seed because it performs best on a test period is a form of test-period cherry-picking and would invalidate the empirical interpretation.

## Experiment Configuration

| Field | Value |
| --- | --- |
| Source output root | artifacts\quant_core_repeated_seed_1000_smoke |
| Seed count requested | 1000 |
| Seed range | 1 to 1000 |
| Preset | smoke |
| Run mode | research_core |
| Max workers | 1 |
| Resume | True |
| Stop on failure | False |
| Dry run | False |
| Real Quant Core execution path | True |


The manifest records `scripts\run_quant_core.py` as the per-seed Quant Core execution path. No heavy job was rerun for this report; the analysis reads existing saved artifacts only.

## Completion and Resume Verification

| Metric | Value |
| --- | --- |
| Seed folders found | 1000 |
| Completed seed folders by minimum files | 1000 |
| Seed execution log rows | 1000 |
| Completed execution rows | 985 |
| Skipped-complete execution rows | 15 |
| Failed execution rows | 0 |


Execution status counts from `seed_execution_log.csv` and the repeated-seed manifest:

| Status | Count |
| --- | --- |
| completed | 985 |
| skipped_complete | 15 |


The 15 `skipped_complete` rows indicate resume behavior: those seed directories already had the required completion files when the repeated-seed runner inspected them. This is consistent with resume semantics and does not indicate failed training.

## Artifact Inventory

Root aggregate files found:

| File | Exists | Rows |
| --- | --- | --- |
| summary.md | yes | file |
| repeated_seed_manifest.json | yes | file |
| seed_execution_log.csv | yes | 1000 |
| seed_artifact_inventory.csv | yes | 21000 |
| seed_forecast_summary.csv | yes | 16000 |
| model_seed_stability.csv | yes | 4 |
| model_horizon_seed_stability.csv | yes | 4 |
| strategy_seed_stability.csv | yes | 4 |
| model_health_seed_stability.csv | yes | 6 |
| consensus_seed_stability.csv | yes | 1 |
| decision_candidate_seed_stability.csv | yes | 1 |
| failure_summary.csv | yes | 2 |


Seed-level artifact coverage from `seed_artifact_inventory.csv`:

| Artifact | Expected seeds | Found | Missing |
| --- | --- | --- | --- |
| analysis_packets.jsonl | 1000 | 1000 | 0 |
| decision_lane_candidates.csv | 1000 | 1000 | 0 |
| equity_curve.csv | 1000 | 1000 | 0 |
| forecast_summary.csv | 1000 | 1000 | 0 |
| forecast_summary_by_horizon.csv | 1000 | 1000 | 0 |
| full_model_predictions.csv | 1000 | 1000 | 0 |
| model_consensus_summary.csv | 1000 | 1000 | 0 |
| model_execution_log.csv | 1000 | 1000 | 0 |
| model_governance.csv | 1000 | 1000 | 0 |
| model_health_summary.csv | 1000 | 1000 | 0 |
| policy_summary.csv | 1000 | 1000 | 0 |
| positions.csv | 1000 | 1000 | 0 |
| regime_summary.csv | 1000 | 1000 | 0 |
| risk_summary.csv | 1000 | 1000 | 0 |
| run_manifest.json | 1000 | 1000 | 0 |
| scenario_matrix.csv | 1000 | 1000 | 0 |
| signals.csv | 1000 | 1000 | 0 |
| strategy_metrics.csv | 1000 | 1000 | 0 |
| summary.md | 1000 | 1000 | 0 |
| trades.csv | 1000 | 1000 | 0 |
| window_summary.csv | 1000 | 1000 | 0 |


Every expected seed-level artifact listed in the inventory was present for all 1000 seed directories. This verifies artifact generation and retention for the smoke workflow, not trading readiness.

## Model-Level Stability Results

The model-level table is aggregated by `model_name` and `target_type` from `model_seed_stability.csv`. Reported dispersion should be interpreted across seeds, not as a basis for selecting a best seed.

| Model | Target | Seeds | RMSE mean | RMSE std | RMSE min | RMSE max | RMSE p05 | RMSE p50 | RMSE p95 | MAE mean | MAE std | MAE p05 | MAE p50 | MAE p95 | Directional accuracy mean | Directional accuracy std | Directional accuracy p05 | Directional accuracy p50 | Directional accuracy p95 | Obs. mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | forward_return | 1000 | 0.036183 | 0 | 0.036183 | 0.036183 | 0.036183 | 0.036183 | 0.036183 | 0.030635 | 0 | 0.030635 | 0.030635 | 0.030635 | 0.488095 | 0 | 0.488095 | 0.488095 | 0.488095 | 21 |
| random_forest | forward_return | 1000 | 0.03131 | 0 | 0.03131 | 0.03131 | 0.03131 | 0.03131 | 0.03131 | 0.026201 | 0 | 0.026201 | 0.026201 | 0.026201 | 0.488095 | 0 | 0.488095 | 0.488095 | 0.488095 | 21 |
| weighted_ensemble | forward_return | 1000 | 0.032346 | 0 | 0.032346 | 0.032346 | 0.032346 | 0.032346 | 0.032346 | 0.027306 | 0 | 0.027306 | 0.027306 | 0.027306 | 0.428571 | 0 | 0.428571 | 0.428571 | 0.428571 | 21 |
| xgboost | forward_return | 1000 | 0.032206 | 0 | 0.032206 | 0.032206 | 0.032206 | 0.032206 | 0.032206 | 0.026957 | 0 | 0.026957 | 0.026957 | 0.026957 | 0.464286 | 0 | 0.464286 | 0.464286 | 0.464286 | 21 |


At displayed precision, RMSE, MAE, directional accuracy, and observation counts are stable across all 1000 seeds for each model row. Min, p05, p50, p95, and max are identical for the displayed metrics. The most and least stable metrics cannot be meaningfully separated in practical terms because the observed standard deviations are near zero throughout the saved stability outputs.

Highest standard-deviation entries across available stability tables:

| Table | Metric | Identifier | Std |
| --- | --- | --- | --- |
| decision_candidate_seed_stability.csv | active_signal_count | horizon=5.0, target_type=forward_return, run_mode=research_core | 0 |
| consensus_seed_stability.csv | agreement_score | horizon=5.0, target_type=forward_return, run_mode=research_core | 0 |
| decision_candidate_seed_stability.csv | model_agreement_score | horizon=5.0, target_type=forward_return, run_mode=research_core | 0 |
| model_seed_stability.csv | directional_accuracy | model_name=xgboost, target_type=forward_return | 0 |
| model_horizon_seed_stability.csv | directional_accuracy | model_name=xgboost, horizon=5.0, target_type=forward_return | 0 |
| strategy_seed_stability.csv | win_rate | model_name=lightgbm, horizon=5.0, target_type=forward_return, run_mode=research_core, policy_variant=regime_threshold_adaptive_drawdown | 0 |
| strategy_seed_stability.csv | win_rate | model_name=xgboost, horizon=5.0, target_type=forward_return, run_mode=research_core, policy_variant=regime_threshold_adaptive_drawdown | 0 |
| strategy_seed_stability.csv | win_rate | model_name=weighted_ensemble, horizon=5.0, target_type=forward_return, run_mode=research_core, policy_variant=regime_threshold_adaptive_drawdown | 0 |


These values are at floating-point precision scale. They should be treated as numerical noise, not substantive seed-driven dispersion.

## Model-Horizon Stability Results

The smoke run contains one horizon in the model-horizon aggregate: horizon `5` for `forward_return`. Because only one horizon is present, this report cannot compare stability across multiple horizons.

| Model | Horizon | Target | Seeds | RMSE mean | RMSE std | RMSE p05 | RMSE p50 | RMSE p95 | MAE mean | MAE std | MAE p05 | MAE p50 | MAE p95 | Directional accuracy mean | Directional accuracy std | Obs. mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | 5 | forward_return | 1000 | 0.03813 | 0 | 0.03813 | 0.03813 | 0.03813 | 0.030635 | 0 | 0.030635 | 0.030635 | 0.030635 | 0.488095 | 0 | 84 |
| random_forest | 5 | forward_return | 1000 | 0.031776 | 0 | 0.031776 | 0.031776 | 0.031776 | 0.026201 | 0 | 0.026201 | 0.026201 | 0.026201 | 0.488095 | 0 | 84 |
| weighted_ensemble | 5 | forward_return | 1000 | 0.033183 | 0 | 0.033183 | 0.033183 | 0.033183 | 0.027306 | 0 | 0.027306 | 0.027306 | 0.027306 | 0.428571 | 0 | 84 |
| xgboost | 5 | forward_return | 1000 | 0.032745 | 0 | 0.032745 | 0.032745 | 0.032745 | 0.026957 | 0 | 0.026957 | 0.026957 | 0.026957 | 0.464286 | 0 | 84 |


Within the available horizon, the stability pattern matches the model-level table: values are stable across all 1000 seeds at displayed precision.

## Strategy, Health, Consensus, and Decision Diagnostics

Strategy stability from `strategy_seed_stability.csv`:

| Model | Horizon | Target | Run mode | Policy variant | Seeds | Sharpe mean | Sharpe std | CAGR mean | Max drawdown mean | Total return mean | Win rate mean | Trade count mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | 5 | forward_return | research_core | regime_threshold_adaptive_drawdown | 1000 | 0.63564 | 0 | 0.03934 | -0.010739 | 0.00276 | 0.666667 | 6 |
| random_forest | 5 | forward_return | research_core | regime_threshold_adaptive_drawdown | 1000 | -0.173171 | 0 | -0.012725 | -0.012397 | -0.00061 | 0.666667 | 3 |
| weighted_ensemble | 5 | forward_return | research_core | regime_threshold_adaptive_drawdown | 1000 | 0.61458 | 0 | 0.039319 | -0.013672 | 0.002758 | 0.666667 | 6 |
| xgboost | 5 | forward_return | research_core | regime_threshold_adaptive_drawdown | 1000 | 0.593702 | 0 | 0.038876 | -0.013672 | 0.002576 | 0.666667 | 6 |


Model-health stability from `model_health_seed_stability.csv`:

| Model | Seeds | Run success mean | Run success std | Forecast obs. mean | Mean RMSE | Mean RMSE std | Mean dir. acc. | Mean dir. acc. std | Policy eval count | Mean Sharpe | Mean Sharpe std | Positive policy freq. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ets | 1000 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | 0 | n/a | n/a | n/a |
| lightgbm | 1000 | 1 | 0 | 84 | 0.03813 | 0 | 0.488095 | 0 | 1 | 0.63564 | 0 | 1 |
| random_forest | 1000 | 1 | 0 | 84 | 0.031776 | 0 | 0.488095 | 0 | 1 | -0.173171 | 0 | 0 |
| sarimax | 1000 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | 0 | n/a | n/a | n/a |
| weighted_ensemble | 1000 | 1 | 0 | 84 | 0.033183 | 0 | 0.428571 | 0 | 1 | 0.61458 | 0 | 1 |
| xgboost | 1000 | 1 | 0 | 84 | 0.032745 | 0 | 0.464286 | 0 | 1 | 0.593702 | 0 | 1 |


Consensus stability from `consensus_seed_stability.csv`:

| Horizon | Target | Run mode | Seeds | Agreement mean | Agreement std | Disagreement mean | Dispersion mean | Sign conflict mean | Active signal share mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | forward_return | research_core | 1000 | 0.907738 | 0 | 0.092262 | 0.00602 | 0.309524 | 0.196429 |


Decision-candidate stability from `decision_candidate_seed_stability.csv`:

| Horizon | Target | Run mode | Seeds | Candidate count mean | Candidate count std | Candidate score mean | Model agreement mean | Active signal count mean | Top policy Sharpe mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | forward_return | research_core | 1000 | 22 | 0 | 0.027052 | 0.988636 | 2.954545 | 0.63564 |


Decision-candidate rows are diagnostics only. They are not final buy recommendations and should not be converted into trading actions without separate validation.

## Failure Summary

No failed seed rows were found in `failure_summary.csv`; the file contains status-summary rows for completed and skipped-complete seeds.

Failure summary contents:

| status | count |
| --- | --- |
| completed | 985 |
| skipped_complete | 15 |


No failed seed execution rows were present in the final saved execution log. Provider, cache, or environment issues can still affect future reruns, so the absence of failures in this saved output should be treated as evidence about this run only.

## Interpretation

The saved 1000-seed smoke outputs show high reproducibility across the recorded seed range. In this configuration, seed-to-seed dispersion is effectively zero for model-level forecast metrics, model-horizon metrics, strategy summaries, model-health summaries, consensus diagnostics, and decision-candidate summaries.

This is a stability finding. It does not mean the models are accurate enough for production trading, does not prove market edge, and does not identify a seed to use in future experiments. The correct interpretation is that the bounded smoke setup appears deterministic or seed-insensitive under the saved configuration and data state.

Large dispersion in future broader runs should be treated as uncertainty. Mean, standard deviation, and quantiles should be reported together; single-seed maxima should not be used as headline evidence.

## Limitations

- The `smoke` preset is a bounded orchestration and artifact-surface validation. It is not a full-market, long-window, production validation.
- The 1000 seeds test stability of saved smoke outputs, not market edge or future profitability.
- Provider availability, cached data, local environment state, and dependency behavior can affect future runs.
- The available model-horizon aggregate contains only horizon `5`, so cross-horizon stability cannot be evaluated here.
- Repeated-seed stability does not prove robustness to different market regimes, ticker universes, forecast windows, transaction-cost assumptions, or portfolio constraints.

## Recommended Next Steps

1. Review any future high-dispersion models or horizons using mean, standard deviation, and p05/p50/p95 bands rather than best-seed values.
2. If compute budget allows, run a later `medium` / `decision_core` repeated-seed diagnostic as a separate experiment with the same no-cherry-picking rule.
3. Keep generated artifacts under ignored artifact/output locations and do not commit them.
4. Prepare a separate presentation summary only after deciding which stability tables are most relevant for the advisor or lecturer.

## Appendix A - Files Used

This report used the following saved files when present:

| File | Exists | Rows |
| --- | --- | --- |
| summary.md | yes | file |
| repeated_seed_manifest.json | yes | file |
| seed_execution_log.csv | yes | 1000 |
| seed_artifact_inventory.csv | yes | 21000 |
| seed_forecast_summary.csv | yes | 16000 |
| model_seed_stability.csv | yes | 4 |
| model_horizon_seed_stability.csv | yes | 4 |
| strategy_seed_stability.csv | yes | 4 |
| model_health_seed_stability.csv | yes | 6 |
| consensus_seed_stability.csv | yes | 1 |
| decision_candidate_seed_stability.csv | yes | 1 |
| failure_summary.csv | yes | 2 |


## Appendix B - Reproducibility Command

The source artifact root corresponds to the following repeated-seed smoke command pattern:

```powershell
C:\Users\luong\.venv\Scripts\python.exe scripts/run_quant_core_repeated_seeds.py ^
  --seed-start 1 ^
  --seed-count 1000 ^
  --preset smoke ^
  --run-mode research_core ^
  --output-dir artifacts\quant_core_repeated_seed_1000_smoke ^
  --max-workers 1 ^
  --resume
```

This command is provided for reproducibility only. It was not rerun during report generation.
