# Quant Core Output Schema

Date: 2026-04-19
Branch: `rebuild-regime-risk-aware-forecasting`

## Artifact Set

`scripts/run_quant_core.py` now writes the following core artifacts under the
 requested output directory:

- `run_manifest.json`
- `summary.md`
- `scenario_matrix.csv`
- `model_governance.csv`
- `full_model_predictions.csv`
- `forecast_summary.csv`
- `forecast_summary_by_horizon.csv`
- `window_summary.csv`
- `risk_summary.csv`
- `regime_summary.csv`
- `signals.csv`
- `positions.csv`
- `trades.csv`
- `strategy_metrics.csv`
- `equity_curve.csv`
- `policy_summary.csv`
- `model_execution_log.csv`
- `model_consensus_summary.csv`
- `model_health_summary.csv`
- `analysis_packets.jsonl`
- `decision_lane_candidates.csv`

## Manifest

`run_manifest.json` records:

- git metadata
- runtime and dependency metadata
- requested and evaluated models
- skipped models with reasons
- run mode and run-mode specification
- requested role filters
- scenario matrix configuration
- artifact paths
- row counts for major output tables

Important additions for governance:

- `run_mode`
- `run_mode_spec`
- `requested_model_roles`
- `governance_output`

## Full Model Predictions

`full_model_predictions.csv` extends the shared forecast contract with quant-core
governance and scenario metadata.

Base forecast contract:

- `timestamp`
- `ticker`
- `y_true`
- `y_pred`
- `model_name`
- `target_type`
- `horizon`
- `window_id`
- optional `target_timestamp`

Added governance and scenario fields:

- `model_family`
- `model_role`
- `model_status`
- `research_priority`
- `supports_policy_eval`
- `model_notes`
- `core_run_id`
- `preset`
- `group_name`
- `target_name`
- `target_column`
- `target_family`
- `target_tradable`
- `ticker_group_members`
- `run_mode`

## Consensus Summary

`model_consensus_summary.csv` is one row per
`timestamp x ticker x horizon x target_type x run_mode x scenario`.

Core fields:

- `model_count`
- `primary_model_count`
- `comparator_model_count`
- `baseline_model_count`
- `shadow_model_count`
- `ensemble_model_count`
- `agreement_score`
- `disagreement_score`
- `agreement_bucket`
- `sign_conflict`
- `dispersion_score`
- `prediction_range`
- `rank_spread`
- `primary_mean_prediction`
- `comparator_mean_prediction`
- `baseline_mean_prediction`
- `ensemble_prediction`
- `primary_vs_comparator_gap`
- `primary_vs_baseline_gap`
- `ensemble_vs_primary_gap`
- `policy_gate_disagreement_share`
- `active_signal_share`

Notes:

- `agreement_score` is sign-based agreement within the packet.
- `policy_gate_disagreement_share` measures how often policy gating disagrees
  with the raw forecast sign after thresholding and conditioning.
- `agreement_bucket` is a retrieval-friendly summary:
  `high`, `medium`, `low`, `unknown`.

## Analysis Packets

`analysis_packets.jsonl` contains one JSON object per
`ticker x timestamp x horizon x target_type x run_mode x scenario`.

Each packet includes:

- packet identity:
  `packet_id`, `packet_generated_at`
- core dimensions:
  `timestamp`, `ticker`, `horizon`, `target_type`, `target_name`, `run_mode`,
  `core_run_id`, `group_name`
- primary prediction context:
  `primary_model_name`, `primary_model_role`, `primary_prediction`,
  `primary_prediction_summary`
- model-level detail:
  `model_by_model_predictions`, `model_ranks`
- agreement and disagreement:
  `model_agreement_score`, `model_disagreement_score`, `dispersion_score`,
  `sign_conflict`, `rank_spread`, `agreement_bucket`
- cross-role deltas:
  `primary_vs_comparator_gap`, `primary_vs_baseline_gap`,
  `ensemble_vs_primary_gap`
- conditioned execution context:
  `policy_gate_disagreement_share`, `active_signal_count`,
  `long_signal_count`, `short_signal_count`, `mean_position_size`
- policy performance context:
  `top_policy_model`, `top_policy_sharpe`, `top_policy_cagr`
- risk context:
  `risk_summary`, `vol_forecast`, `volatility_bucket`
- regime context:
  `regime_summary`, `regime_label`
- realized outcome context:
  `realized_y_true`, `realized_available`, `target_timestamp`
- retrieval metadata:
  `signal_strength_bucket`, `model_role_context`, `retrieval_metadata`

JSON-encoded nested fields:

- `primary_prediction_summary`
- `model_by_model_predictions`
- `model_ranks`
- `ensemble_summary`
- `regime_summary`
- `risk_summary`
- `policy_summary`
- `retrieval_metadata`

## Model Health Summary

`model_health_summary.csv` is one row per model aggregated across the full run.

Core fields:

- execution health:
  `run_success_count`, `run_failure_count`, `run_success_rate`,
  `warning_count_total`, `missing_output_count_total`, `failure_reasons`
- forecast quality:
  `forecast_observations_total`, `mean_rmse`, `median_rmse`,
  `rmse_dispersion`, `mean_directional_accuracy`,
  `median_directional_accuracy`, `directional_accuracy_dispersion`
- slice stability:
  `positive_slice_frequency`, `strong_slice_frequency`,
  `window_count`, `window_rmse_dispersion`,
  `window_directional_accuracy_dispersion`
- drift-style summaries:
  `directional_accuracy_drift`, `rmse_drift`, `drift_flag`
- policy summaries:
  `policy_eval_count`, `mean_sharpe`, `median_sharpe`,
  `positive_policy_frequency`, `mean_cagr`, `mean_max_drawdown`
- final label:
  `health_status`

Current `health_status` categories:

- `healthy`
- `brittle`
- `weak`
- `failing`

## Decision Lane Candidates

`decision_lane_candidates.csv` is intentionally conservative. It surfaces packet
rows that are:

- tradable
- positively predicted by the primary model
- in `medium` or `high` agreement buckets
- supported by at least one active policy signal

Key fields:

- `packet_id`
- `timestamp`
- `ticker`
- `group_name`
- `horizon`
- `target_type`
- `run_mode`
- `primary_model_name`
- `primary_prediction`
- `model_agreement_score`
- `agreement_bucket`
- `regime_label`
- `volatility_bucket`
- `active_signal_count`
- `top_policy_model`
- `top_policy_sharpe`
- `candidate_score`

## Fallback Provenance

Fallback behavior is explicit in outputs and should remain so for downstream
retrieval and analyst interpretation:

- risk fallback: `var_cvar_drawdown_fallback`
- regime fallback: `markov_switching_threshold_fallback`

These labels make it possible for future RAG, LLM, and human analyst layers to
distinguish full statistical context from degraded but reproducible fallback
context.
