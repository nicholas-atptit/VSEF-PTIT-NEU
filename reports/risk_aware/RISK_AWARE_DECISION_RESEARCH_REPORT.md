# Risk-Aware Decision Research Report

## 1. Executive Summary

The current evidence does not prove that the risk-aware ranking improves candidate utility over forecast-only ranking. This weakens the risk-layer value claim and suggests that future work should focus on better risk feature design, regime-aware filtering, or stricter candidate eligibility rules. Mean risk-aware minus forecast-only average realized return: `-0.000804978`. Mean risk-aware minus forecast-only return/volatility proxy: `-0.0316192`. Mean drawdown reduction versus forecast-only: `-0.00983917`. Mean hit-ratio difference versus forecast-only: `-0.0245066`.

## 2. Phase 3 Objective

Phase 3 evaluates whether a risk-aware candidate ranking improves diagnostic decision candidate utility compared with forecast-only ranking.

## 3. Relation To Phase 0, Phase 1, And Phase 2

- Phase 0 froze VSEF v1 governance boundaries.
- Phase 1 implemented config-driven experiment execution and artifact conventions.
- Phase 2 found limited evidence that forecasting models consistently outperform simple baselines on MAE/RMSE.
- Phase 3 therefore tests whether risk-aware filtering improves diagnostic candidate quality through drawdown, volatility, VaR, CVaR, hit ratio, and risk-adjusted behavior.

## 4. Candidate Policy Definitions

| policy | ranking_basis | risk_controls | diagnostic_only |
| --- | --- | --- | --- |
| forecast_only | expected_return_proxy, directional_confidence | disabled | True |
| risk_aware | forecast_score minus volatility, drawdown, VaR, CVaR, and missing-metric penalty | enabled | True |

## 5. Experiment Design

- EXP-RK-001: candidate comparison between forecast-only and risk-aware ranking.
- EXP-RK-002: equal-weight diagnostic candidate basket outcome evaluation for top-N baskets.
- Candidate baskets are diagnostic research evidence only and are not real portfolios.

## 6. Data And Source Artifact Evidence

| source | path | exists |
| --- | --- | --- |
| EXP-RK-001 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-RK-001\config\original_config.yaml | True |
| EXP-RK-001 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-RK-001\config\resolved_config.yaml | True |
| EXP-RK-001 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-RK-001\manifests\run_manifest.json | True |
| EXP-RK-001 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-RK-001\logs\run.log | True |
| EXP-RK-001 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-RK-001\logs\errors.log | True |
| EXP-RK-001 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-RK-001\metrics\metrics.csv | True |
| EXP-RK-001 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-RK-001\artifacts\candidate_comparison.csv | True |
| EXP-RK-001 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-RK-001\artifacts\topn_basket_metrics.csv | False |
| EXP-RK-001 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-RK-001\reports\summary.md | True |
| EXP-RK-002 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-RK-002\config\original_config.yaml | True |
| EXP-RK-002 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-RK-002\config\resolved_config.yaml | True |
| EXP-RK-002 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-RK-002\manifests\run_manifest.json | True |
| EXP-RK-002 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-RK-002\logs\run.log | True |
| EXP-RK-002 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-RK-002\logs\errors.log | True |
| EXP-RK-002 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-RK-002\metrics\metrics.csv | True |
| EXP-RK-002 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-RK-002\artifacts\candidate_comparison.csv | False |
| EXP-RK-002 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-RK-002\artifacts\topn_basket_metrics.csv | True |
| EXP-RK-002 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-RK-002\reports\summary.md | True |
| EXP-FC-001 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-001\predictions\predictions.csv | True |
| EXP-FC-001 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-001\metrics\metrics.csv | True |
| EXP-FC-001 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-001\manifests\run_manifest.json | True |
| EXP-FC-001 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-001\logs\run.log | True |
| EXP-FC-001 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-001\logs\errors.log | True |
| EXP-FC-003 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-003\predictions\predictions.csv | True |
| EXP-FC-003 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-003\metrics\metrics.csv | True |
| EXP-FC-003 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-003\manifests\run_manifest.json | True |
| EXP-FC-003 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-003\logs\run.log | True |
| EXP-FC-003 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-003\logs\errors.log | True |
| forecasting_core_report | K:\Repos\VSEF-PTIT-NEU\reports\forecasting_core\forecast_metrics.csv | True |
| forecasting_core_report | K:\Repos\VSEF-PTIT-NEU\reports\forecasting_core\model_ranking.csv | True |
| forecasting_core_report | K:\Repos\VSEF-PTIT-NEU\reports\forecasting_core\horizon_comparison.csv | True |

## 7. Candidate Comparison Results

| experiment_id | policy_id | candidate_type | candidate_date | ticker | horizon | rank | candidate_score | forecast_score | risk_penalty | risk_adjusted_score | expected_return_proxy | directional_confidence | realized_volatility | max_drawdown | var_95 | cvar_95 | model_name | model_type | source_experiment | diagnostics | diagnostic_only | realized_return | prediction_count | model_consensus_count | consensus_score | missing_prediction_rate | current_close | y_true |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-RK-001 | candidate_policy_forecast_only | forecast_only | 2024-07-02 | FPT | 5 | 1 | 0.00428391 | 0.00428391 | 0 | 0.00428391 | 0.00428391 | 0.5 | 0.0178793 | -0.059563 | -0.0179304 | -0.0293914 | ets | model | EXP-FC-003 | {"candidate_evidence": "aggregated_source_model_predictions", "not_investment_advice": true, "policy_ranking": "forecast_only", "realized_outcome_source": "forecast_artifact_y_true_vs_local_candidate_date_close", "risk_confidence_level": 0.95, "risk_controls_enabled": false, "risk_lookback_window": 20, "source_model_names": ["ets", "lightgbm", "sarimax", "xgboost"], "source_model_types": ["model"], "source_rows": 4, "source_selection": "highest_priority_source_experiment_per_date_ticker_horizon"} | True | 0.0742141 | 4 | 2 | 0.5 | 0 | 108.47 | 116.52 |
| EXP-RK-001 | candidate_policy_forecast_only | forecast_only | 2024-07-04 | MWG | 1 | 1 | 0.0280573 | 0.0280573 | 0 | 0.0280573 | 0.0280573 | 0.5 | 0.0163487 | -0.0311551 | -0.0161325 | -0.0235105 | sarimax | model | EXP-FC-003 | {"candidate_evidence": "aggregated_source_model_predictions", "not_investment_advice": true, "policy_ranking": "forecast_only", "realized_outcome_source": "forecast_artifact_y_true_vs_local_candidate_date_close", "risk_confidence_level": 0.95, "risk_controls_enabled": false, "risk_lookback_window": 20, "source_model_names": ["ets", "lightgbm", "sarimax", "xgboost"], "source_model_types": ["model"], "source_rows": 4, "source_selection": "highest_priority_source_experiment_per_date_ticker_horizon"} | True | 0.00763478 | 4 | 2 | 0.5 | 0 | 64.18 | 64.67 |
| EXP-RK-001 | candidate_policy_forecast_only | forecast_only | 2024-07-08 | MWG | 1 | 1 | 0.130005 | 0.130005 | 0 | 0.130005 | 0.130005 | 0.25 | 0.0166788 | -0.0311551 | -0.0161325 | -0.0235105 | sarimax | model | EXP-FC-003 | {"candidate_evidence": "aggregated_source_model_predictions", "not_investment_advice": true, "policy_ranking": "forecast_only", "realized_outcome_source": "forecast_artifact_y_true_vs_local_candidate_date_close", "risk_confidence_level": 0.95, "risk_controls_enabled": false, "risk_lookback_window": 20, "source_model_names": ["ets", "lightgbm", "sarimax", "xgboost"], "source_model_types": ["model"], "source_rows": 4, "source_selection": "highest_priority_source_experiment_per_date_ticker_horizon"} | True | -0.00302801 | 4 | 1 | 0.25 | 0 | 66.05 | 65.85 |
| EXP-RK-001 | candidate_policy_forecast_only | forecast_only | 2024-07-10 | MWG | 1 | 1 | 0.497218 | 0.497218 | 0 | 0.497218 | 0.497218 | 0.5 | 0.0171755 | -0.0311551 | -0.0235346 | -0.0239939 | sarimax | model | EXP-FC-003 | {"candidate_evidence": "aggregated_source_model_predictions", "not_investment_advice": true, "policy_ranking": "forecast_only", "realized_outcome_source": "forecast_artifact_y_true_vs_local_candidate_date_close", "risk_confidence_level": 0.95, "risk_controls_enabled": false, "risk_lookback_window": 20, "source_model_names": ["ets", "lightgbm", "sarimax", "xgboost"], "source_model_types": ["model"], "source_rows": 4, "source_selection": "highest_priority_source_experiment_per_date_ticker_horizon"} | True | -0.00606815 | 4 | 2 | 0.5 | 0 | 64.27 | 63.88 |
| EXP-RK-001 | candidate_policy_forecast_only | forecast_only | 2024-07-12 | MWG | 1 | 1 | 1.44112 | 1.44112 | 0 | 1.44112 | 1.44112 | 0.5 | 0.016738 | -0.0328539 | -0.0235346 | -0.0239939 | sarimax | model | EXP-FC-003 | {"candidate_evidence": "aggregated_source_model_predictions", "not_investment_advice": true, "policy_ranking": "forecast_only", "realized_outcome_source": "forecast_artifact_y_true_vs_local_candidate_date_close", "risk_confidence_level": 0.95, "risk_controls_enabled": false, "risk_lookback_window": 20, "source_model_names": ["ets", "lightgbm", "sarimax", "xgboost"], "source_model_types": ["model"], "source_rows": 4, "source_selection": "highest_priority_source_experiment_per_date_ticker_horizon"} | True | -0.00767063 | 4 | 2 | 0.5 | 0 | 63.88 | 63.39 |
| EXP-RK-001 | candidate_policy_forecast_only | forecast_only | 2024-07-16 | MWG | 1 | 1 | 4.02247 | 4.02247 | 0 | 4.02247 | 4.02247 | 0.5 | 0.0164426 | -0.0433005 | -0.0235346 | -0.0239939 | sarimax | model | EXP-FC-003 | {"candidate_evidence": "aggregated_source_model_predictions", "not_investment_advice": true, "policy_ranking": "forecast_only", "realized_outcome_source": "forecast_artifact_y_true_vs_local_candidate_date_close", "risk_confidence_level": 0.95, "risk_controls_enabled": false, "risk_lookback_window": 20, "source_model_names": ["ets", "lightgbm", "sarimax", "xgboost"], "source_model_types": ["model"], "source_rows": 4, "source_selection": "highest_priority_source_experiment_per_date_ticker_horizon"} | True | -0.00933692 | 4 | 2 | 0.5 | 0 | 63.19 | 62.6 |
| EXP-RK-001 | candidate_policy_forecast_only | forecast_only | 2024-07-18 | MWG | 1 | 1 | 10.7775 | 10.7775 | 0 | 10.7775 | 10.7775 | 0.5 | 0.0169604 | -0.0522332 | -0.0235346 | -0.0239939 | sarimax | model | EXP-FC-003 | {"candidate_evidence": "aggregated_source_model_predictions", "not_investment_advice": true, "policy_ranking": "forecast_only", "realized_outcome_source": "forecast_artifact_y_true_vs_local_candidate_date_close", "risk_confidence_level": 0.95, "risk_controls_enabled": false, "risk_lookback_window": 20, "source_model_names": ["ets", "lightgbm", "sarimax", "xgboost"], "source_model_types": ["model"], "source_rows": 4, "source_selection": "highest_priority_source_experiment_per_date_ticker_horizon"} | True | 0.00920724 | 4 | 2 | 0.5 | 0 | 64.08 | 64.67 |
| EXP-RK-001 | candidate_policy_forecast_only | forecast_only | 2024-07-18 | FPT | 5 | 1 | 0.00207131 | 0.00207131 | 0 | 0.00207131 | 0.00207131 | 0.5 | 0.0186486 | -0.0837701 | -0.0294382 | -0.0303274 | ets | model | EXP-FC-003 | {"candidate_evidence": "aggregated_source_model_predictions", "not_investment_advice": true, "policy_ranking": "forecast_only", "realized_outcome_source": "forecast_artifact_y_true_vs_local_candidate_date_close", "risk_confidence_level": 0.95, "risk_controls_enabled": false, "risk_lookback_window": 20, "source_model_names": ["ets", "lightgbm", "sarimax", "xgboost"], "source_model_types": ["model"], "source_rows": 4, "source_selection": "highest_priority_source_experiment_per_date_ticker_horizon"} | True | -0.0226958 | 4 | 2 | 0.5 | 0 | 108.39 | 105.93 |
| EXP-RK-001 | candidate_policy_forecast_only | forecast_only | 2024-07-19 | FPT | 5 | 1 | 0.0173001 | 0.0173001 | 0 | 0.0173001 | 0.0173001 | 0.5 | 0.0180719 | -0.0973795 | -0.0294382 | -0.0303274 | ets | model | EXP-FC-003 | {"candidate_evidence": "aggregated_source_model_predictions", "not_investment_advice": true, "policy_ranking": "forecast_only", "realized_outcome_source": "forecast_artifact_y_true_vs_local_candidate_date_close", "risk_confidence_level": 0.95, "risk_controls_enabled": false, "risk_lookback_window": 20, "source_model_names": ["ets", "lightgbm", "sarimax", "xgboost"], "source_model_types": ["model"], "source_rows": 4, "source_selection": "highest_priority_source_experiment_per_date_ticker_horizon"} | True | 0.0158269 | 4 | 2 | 0.5 | 0 | 106.78 | 108.47 |
| EXP-RK-001 | candidate_policy_forecast_only | forecast_only | 2024-07-22 | MWG | 1 | 1 | 29.0963 | 29.0963 | 0 | 29.0963 | 29.0963 | 0.5 | 0.0160796 | -0.0522332 | -0.0100698 | -0.0239939 | sarimax | model | EXP-FC-003 | {"candidate_evidence": "aggregated_source_model_predictions", "not_investment_advice": true, "policy_ranking": "forecast_only", "realized_outcome_source": "forecast_artifact_y_true_vs_local_candidate_date_close", "risk_confidence_level": 0.95, "risk_controls_enabled": false, "risk_lookback_window": 20, "source_model_names": ["ets", "lightgbm", "sarimax", "xgboost"], "source_model_types": ["model"], "source_rows": 4, "source_selection": "highest_priority_source_experiment_per_date_ticker_horizon"} | True | -0.0428771 | 4 | 2 | 0.5 | 0 | 64.37 | 61.61 |
| EXP-RK-001 | candidate_policy_forecast_only | forecast_only | 2024-07-22 | DGC | 1 | 2 | 0.00521978 | 0.00521978 | 0 | 0.00521978 | 0.00521978 | 0.5 | 0.0152904 | -0.109389 | -0.0160936 | -0.0601909 | ets | model | EXP-FC-003 | {"candidate_evidence": "aggregated_source_model_predictions", "not_investment_advice": true, "policy_ranking": "forecast_only", "realized_outcome_source": "forecast_artifact_y_true_vs_local_candidate_date_close", "risk_confidence_level": 0.95, "risk_controls_enabled": false, "risk_lookback_window": 20, "source_model_names": ["ets", "lightgbm", "sarimax", "xgboost"], "source_model_types": ["model"], "source_rows": 4, "source_selection": "highest_priority_source_experiment_per_date_ticker_horizon"} | True | -0.0490924 | 4 | 2 | 0.5 | 0 | 106.33 | 101.11 |
| EXP-RK-001 | candidate_policy_forecast_only | forecast_only | 2024-07-22 | DGC | 3 | 1 | 0.0163205 | 0.0163205 | 0 | 0.0163205 | 0.0163205 | 0.5 | 0.0152904 | -0.109389 | -0.0160936 | -0.0601909 | ets | model | EXP-FC-003 | {"candidate_evidence": "aggregated_source_model_predictions", "not_investment_advice": true, "policy_ranking": "forecast_only", "realized_outcome_source": "forecast_artifact_y_true_vs_local_candidate_date_close", "risk_confidence_level": 0.95, "risk_controls_enabled": false, "risk_lookback_window": 20, "source_model_names": ["ets", "lightgbm", "sarimax", "xgboost"], "source_model_types": ["model"], "source_rows": 4, "source_selection": "highest_priority_source_experiment_per_date_ticker_horizon"} | True | -0.0263331 | 4 | 2 | 0.5 | 0 | 106.33 | 103.53 |
| EXP-RK-001 | candidate_policy_forecast_only | forecast_only | 2024-07-22 | FPT | 5 | 1 | 0.0268158 | 0.0268158 | 0 | 0.0268158 | 0.0268158 | 0.5 | 0.0173069 | -0.110989 | -0.0256496 | -0.0303274 | ets | model | EXP-FC-003 | {"candidate_evidence": "aggregated_source_model_predictions", "not_investment_advice": true, "policy_ranking": "forecast_only", "realized_outcome_source": "forecast_artifact_y_true_vs_local_candidate_date_close", "risk_confidence_level": 0.95, "risk_controls_enabled": false, "risk_lookback_window": 20, "source_model_names": ["ets", "lightgbm", "sarimax", "xgboost"], "source_model_types": ["model"], "source_rows": 4, "source_selection": "highest_priority_source_experiment_per_date_ticker_horizon"} | True | 0.0394599 | 4 | 2 | 0.5 | 0 | 105.17 | 109.32 |
| EXP-RK-001 | candidate_policy_forecast_only | forecast_only | 2024-07-22 | DGC | 5 | 2 | 0.0251018 | 0.0251018 | 0 | 0.0251018 | 0.0251018 | 0.25 | 0.0152904 | -0.109389 | -0.0160936 | -0.0601909 | ets | model | EXP-FC-003 | {"candidate_evidence": "aggregated_source_model_predictions", "not_investment_advice": true, "policy_ranking": "forecast_only", "realized_outcome_source": "forecast_artifact_y_true_vs_local_candidate_date_close", "risk_confidence_level": 0.95, "risk_controls_enabled": false, "risk_lookback_window": 20, "source_model_names": ["ets", "lightgbm", "sarimax", "xgboost"], "source_model_types": ["model"], "source_rows": 4, "source_selection": "highest_priority_source_experiment_per_date_ticker_horizon"} | True | -0.00874636 | 4 | 1 | 0.25 | 0 | 106.33 | 105.4 |
| EXP-RK-001 | candidate_policy_forecast_only | forecast_only | 2024-07-23 | DGC | 1 | 1 | 0.0404931 | 0.0404931 | 0 | 0.0404931 | 0.0404931 | 0.5 | 0.01807 | -0.153112 | -0.0496474 | -0.0601909 | ets | model | EXP-FC-003 | {"candidate_evidence": "aggregated_source_model_predictions", "not_investment_advice": true, "policy_ranking": "forecast_only", "realized_outcome_source": "forecast_artifact_y_true_vs_local_candidate_date_close", "risk_confidence_level": 0.95, "risk_controls_enabled": false, "risk_lookback_window": 20, "source_model_names": ["ets", "lightgbm", "sarimax", "xgboost"], "source_model_types": ["model"], "source_rows": 4, "source_selection": "highest_priority_source_experiment_per_date_ticker_horizon"} | True | 0.0147364 | 4 | 2 | 0.5 | 0 | 101.11 | 102.6 |
| EXP-RK-001 | candidate_policy_forecast_only | forecast_only | 2024-07-23 | DGC | 3 | 1 | 0.0453995 | 0.0453995 | 0 | 0.0453995 | 0.0453995 | 0.5 | 0.01807 | -0.153112 | -0.0496474 | -0.0601909 | ets | model | EXP-FC-003 | {"candidate_evidence": "aggregated_source_model_predictions", "not_investment_advice": true, "policy_ranking": "forecast_only", "realized_outcome_source": "forecast_artifact_y_true_vs_local_candidate_date_close", "risk_confidence_level": 0.95, "risk_controls_enabled": false, "risk_lookback_window": 20, "source_model_names": ["ets", "lightgbm", "sarimax", "xgboost"], "source_model_types": ["model"], "source_rows": 4, "source_selection": "highest_priority_source_experiment_per_date_ticker_horizon"} | True | 0.0442093 | 4 | 2 | 0.5 | 0 | 101.11 | 105.58 |
| EXP-RK-001 | candidate_policy_forecast_only | forecast_only | 2024-07-23 | DGC | 5 | 1 | 0.0526916 | 0.0526916 | 0 | 0.0526916 | 0.0526916 | 0.75 | 0.01807 | -0.153112 | -0.0496474 | -0.0601909 | ets | model | EXP-FC-003 | {"candidate_evidence": "aggregated_source_model_predictions", "not_investment_advice": true, "policy_ranking": "forecast_only", "realized_outcome_source": "forecast_artifact_y_true_vs_local_candidate_date_close", "risk_confidence_level": 0.95, "risk_controls_enabled": false, "risk_lookback_window": 20, "source_model_names": ["ets", "lightgbm", "sarimax", "xgboost"], "source_model_types": ["model"], "source_rows": 4, "source_selection": "highest_priority_source_experiment_per_date_ticker_horizon"} | True | 0.0368905 | 4 | 3 | 0.75 | 0 | 101.11 | 104.84 |
| EXP-RK-001 | candidate_policy_forecast_only | forecast_only | 2024-07-23 | FPT | 5 | 2 | 0.022015 | 0.022015 | 0 | 0.022015 | 0.022015 | 0.5 | 0.0173031 | -0.110989 | -0.0256496 | -0.0303274 | ets | model | EXP-FC-003 | {"candidate_evidence": "aggregated_source_model_predictions", "not_investment_advice": true, "policy_ranking": "forecast_only", "realized_outcome_source": "forecast_artifact_y_true_vs_local_candidate_date_close", "risk_confidence_level": 0.95, "risk_controls_enabled": false, "risk_lookback_window": 20, "source_model_names": ["ets", "lightgbm", "sarimax", "xgboost"], "source_model_types": ["model"], "source_rows": 4, "source_selection": "highest_priority_source_experiment_per_date_ticker_horizon"} | True | 0.0199342 | 4 | 2 | 0.5 | 0 | 106.35 | 108.47 |
| EXP-RK-001 | candidate_policy_forecast_only | forecast_only | 2024-07-24 | MWG | 1 | 1 | 83.8523 | 83.8523 | 0 | 83.8523 | 83.8523 | 0.5 | 0.019257 | -0.0835731 | -0.0249381 | -0.0428771 | sarimax | model | EXP-FC-003 | {"candidate_evidence": "aggregated_source_model_predictions", "not_investment_advice": true, "policy_ranking": "forecast_only", "realized_outcome_source": "forecast_artifact_y_true_vs_local_candidate_date_close", "risk_confidence_level": 0.95, "risk_controls_enabled": false, "risk_lookback_window": 20, "source_model_names": ["ets", "lightgbm", "sarimax", "xgboost"], "source_model_types": ["model"], "source_rows": 4, "source_selection": "highest_priority_source_experiment_per_date_ticker_horizon"} | True | -0.00660829 | 4 | 2 | 0.5 | 0 | 60.53 | 60.13 |
| EXP-RK-001 | candidate_policy_forecast_only | forecast_only | 2024-07-24 | DGC | 1 | 2 | 0.013813 | 0.013813 | 0 | 0.013813 | 0.013813 | 0.5 | 0.0185771 | -0.153112 | -0.0496474 | -0.0601909 | ets | model | EXP-FC-003 | {"candidate_evidence": "aggregated_source_model_predictions", "not_investment_advice": true, "policy_ranking": "forecast_only", "realized_outcome_source": "forecast_artifact_y_true_vs_local_candidate_date_close", "risk_confidence_level": 0.95, "risk_controls_enabled": false, "risk_lookback_window": 20, "source_model_names": ["ets", "lightgbm", "sarimax", "xgboost"], "source_model_types": ["model"], "source_rows": 4, "source_selection": "highest_priority_source_experiment_per_date_ticker_horizon"} | True | 0.00906433 | 4 | 2 | 0.5 | 0 | 102.6 | 103.53 |

## 8. Basket Outcome Results

| experiment_id | policy_id | candidate_type | basket_date | horizon | top_n | candidate_count | average_realized_return | median_realized_return | hit_ratio | return_volatility_proxy | max_drawdown | var_95 | cvar_95 | worst_period_return | missing_outcome_rate | diagnostic_only | basket_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-RK-002 | candidate_policy_forecast_only | forecast_only | ALL | 1 | 1 | 120 | -0.000724738 | -0.0011937 | 0.425 | -0.0508879 | -0.146624 | -0.0174006 | -0.0388861 | -0.0550606 | 0 | True | 120 |
| EXP-RK-002 | candidate_policy_forecast_only | forecast_only | ALL | 1 | 3 | 284 | -0.000199654 | -0.000377621 | 0.5 | -0.0165112 | -0.0970679 | -0.0135517 | -0.0327625 | -0.0515285 | 0 | True | 120 |
| EXP-RK-002 | candidate_policy_forecast_only | forecast_only | ALL | 1 | 5 | 284 | -0.000199654 | -0.000377621 | 0.5 | -0.0165112 | -0.0970679 | -0.0135517 | -0.0327625 | -0.0515285 | 0 | True | 120 |
| EXP-RK-002 | candidate_policy_forecast_only | forecast_only | ALL | 3 | 1 | 112 | 0.00496341 | 0.00530532 | 0.589286 | 0.202464 | -0.209129 | -0.028795 | -0.0498929 | -0.0964005 | 0 | True | 112 |
| EXP-RK-002 | candidate_policy_forecast_only | forecast_only | ALL | 3 | 3 | 262 | 0.00144826 | 0.00219872 | 0.535714 | 0.0696683 | -0.2004 | -0.0323437 | -0.0511883 | -0.0713586 | 0 | True | 112 |
| EXP-RK-002 | candidate_policy_forecast_only | forecast_only | ALL | 3 | 5 | 268 | 0.00144091 | 0.00219872 | 0.535714 | 0.0700914 | -0.2004 | -0.0285964 | -0.050825 | -0.0713586 | 0 | True | 112 |
| EXP-RK-002 | candidate_policy_forecast_only | forecast_only | ALL | 5 | 1 | 113 | 0.0086207 | 0.00712251 | 0.663717 | 0.277398 | -0.295402 | -0.048412 | -0.0646174 | -0.104459 | 0 | True | 113 |
| EXP-RK-002 | candidate_policy_forecast_only | forecast_only | ALL | 5 | 3 | 315 | 0.00420748 | 0.00672908 | 0.566372 | 0.167211 | -0.237821 | -0.0346583 | -0.0609498 | -0.0925241 | 0 | True | 113 |
| EXP-RK-002 | candidate_policy_forecast_only | forecast_only | ALL | 5 | 5 | 338 | 0.00458099 | 0.00698916 | 0.575221 | 0.181084 | -0.238023 | -0.0388689 | -0.0603905 | -0.0925241 | 0 | True | 113 |
| EXP-RK-002 | candidate_policy_risk_aware | risk_aware | ALL | 1 | 1 | 120 | -0.00115281 | -0.00172254 | 0.4 | -0.0832651 | -0.221675 | -0.018434 | -0.0330814 | -0.0479965 | 0 | True | 120 |
| EXP-RK-002 | candidate_policy_risk_aware | risk_aware | ALL | 1 | 3 | 284 | -0.000199654 | -0.000377621 | 0.5 | -0.0165112 | -0.0970679 | -0.0135517 | -0.0327625 | -0.0515285 | 0 | True | 120 |
| EXP-RK-002 | candidate_policy_risk_aware | risk_aware | ALL | 1 | 5 | 284 | -0.000199654 | -0.000377621 | 0.5 | -0.0165112 | -0.0970679 | -0.0135517 | -0.0327625 | -0.0515285 | 0 | True | 120 |
| EXP-RK-002 | candidate_policy_risk_aware | risk_aware | ALL | 3 | 1 | 112 | 0.000658309 | 0.000883392 | 0.5 | 0.0285561 | -0.285621 | -0.0367576 | -0.0515097 | -0.0619545 | 0 | True | 112 |
| EXP-RK-002 | candidate_policy_risk_aware | risk_aware | ALL | 3 | 3 | 262 | 0.00111405 | 0.00147988 | 0.526786 | 0.0551854 | -0.2004 | -0.0285964 | -0.050825 | -0.0713586 | 0 | True | 112 |
| EXP-RK-002 | candidate_policy_risk_aware | risk_aware | ALL | 3 | 5 | 268 | 0.00144091 | 0.00219872 | 0.535714 | 0.0700914 | -0.2004 | -0.0285964 | -0.050825 | -0.0713586 | 0 | True | 112 |
| EXP-RK-002 | candidate_policy_risk_aware | risk_aware | ALL | 5 | 1 | 113 | 0.00538843 | 0.00380228 | 0.557522 | 0.17323 | -0.24585 | -0.0425998 | -0.0589257 | -0.0805891 | 0 | True | 113 |
| EXP-RK-002 | candidate_policy_risk_aware | risk_aware | ALL | 5 | 3 | 315 | 0.00526233 | 0.00698916 | 0.575221 | 0.207573 | -0.224382 | -0.0346583 | -0.0582878 | -0.0925241 | 0 | True | 113 |
| EXP-RK-002 | candidate_policy_risk_aware | risk_aware | ALL | 5 | 5 | 338 | 0.00458099 | 0.00698916 | 0.575221 | 0.181084 | -0.238023 | -0.0388689 | -0.0603905 | -0.0925241 | 0 | True | 113 |

## 9. Risk-Adjusted Utility Discussion

The current evidence does not prove that the risk-aware ranking improves candidate utility over forecast-only ranking. This weakens the risk-layer value claim and suggests that future work should focus on better risk feature design, regime-aware filtering, or stricter candidate eligibility rules.

- Mean risk-aware minus forecast-only average realized return: `-0.000804978`.
- Mean risk-aware minus forecast-only return/volatility proxy: `-0.0316192`.
- Mean drawdown reduction versus forecast-only: `-0.00983917`.
- Mean hit-ratio difference versus forecast-only: `-0.0245066`.

### Drawdown Comparison

| horizon | top_n | forecast_only_max_drawdown | risk_aware_max_drawdown | drawdown_reduction_vs_forecast_only | diagnostic_only |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | -0.146624 | -0.221675 | -0.0750505 | True |
| 1 | 3 | -0.0970679 | -0.0970679 | -4.996e-16 | True |
| 1 | 5 | -0.0970679 | -0.0970679 | -4.996e-16 | True |
| 3 | 1 | -0.209129 | -0.285621 | -0.0764923 | True |
| 3 | 3 | -0.2004 | -0.2004 | 0 | True |
| 3 | 5 | -0.2004 | -0.2004 | 0 | True |
| 5 | 1 | -0.295402 | -0.24585 | 0.0495518 | True |
| 5 | 3 | -0.237821 | -0.224382 | 0.0134384 | True |
| 5 | 5 | -0.238023 | -0.238023 | 0 | True |

### Hit Ratio Comparison

| horizon | top_n | forecast_only_hit_ratio | risk_aware_hit_ratio | hit_ratio_difference_vs_forecast_only | diagnostic_only |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.425 | 0.4 | -0.025 | True |
| 1 | 3 | 0.5 | 0.5 | 0 | True |
| 1 | 5 | 0.5 | 0.5 | 0 | True |
| 3 | 1 | 0.589286 | 0.5 | -0.0892857 | True |
| 3 | 3 | 0.535714 | 0.526786 | -0.00892857 | True |
| 3 | 5 | 0.535714 | 0.535714 | 0 | True |
| 5 | 1 | 0.663717 | 0.557522 | -0.106195 | True |
| 5 | 3 | 0.566372 | 0.575221 | 0.00884956 | True |
| 5 | 5 | 0.575221 | 0.575221 | 0 | True |

### Risk Summary

| source | candidate_type | horizon | top_n | average_realized_volatility | average_max_drawdown | average_var_95 | average_cvar_95 | average_realized_return | return_volatility_proxy | hit_ratio | diagnostic_only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_comparison | forecast_only | 1 |  | 0.0133442 | -0.0629478 | -0.0202652 | -0.0270264 | 7.24227e-06 |  | 0.454225 | True |
| candidate_comparison | forecast_only | 3 |  | 0.0135302 | -0.0692038 | -0.0215955 | -0.0283684 | 0.0023491 |  | 0.552239 | True |
| candidate_comparison | forecast_only | 5 |  | 0.0134004 | -0.0672192 | -0.0208451 | -0.0272868 | 0.00525006 |  | 0.573964 | True |
| candidate_comparison | risk_aware | 1 |  | 0.0133442 | -0.0629478 | -0.0202652 | -0.0270264 | 7.24227e-06 |  | 0.454225 | True |
| candidate_comparison | risk_aware | 3 |  | 0.0135302 | -0.0692038 | -0.0215955 | -0.0283684 | 0.0023491 |  | 0.552239 | True |
| candidate_comparison | risk_aware | 5 |  | 0.0134004 | -0.0672192 | -0.0208451 | -0.0272868 | 0.00525006 |  | 0.573964 | True |
| basket_outcome | forecast_only | 1 | 1 |  | -0.146624 | -0.0174006 | -0.0388861 | -0.000724738 | -0.0508879 | 0.425 | True |
| basket_outcome | forecast_only | 1 | 3 |  | -0.0970679 | -0.0135517 | -0.0327625 | -0.000199654 | -0.0165112 | 0.5 | True |
| basket_outcome | forecast_only | 1 | 5 |  | -0.0970679 | -0.0135517 | -0.0327625 | -0.000199654 | -0.0165112 | 0.5 | True |
| basket_outcome | forecast_only | 3 | 1 |  | -0.209129 | -0.028795 | -0.0498929 | 0.00496341 | 0.202464 | 0.589286 | True |
| basket_outcome | forecast_only | 3 | 3 |  | -0.2004 | -0.0323437 | -0.0511883 | 0.00144826 | 0.0696683 | 0.535714 | True |
| basket_outcome | forecast_only | 3 | 5 |  | -0.2004 | -0.0285964 | -0.050825 | 0.00144091 | 0.0700914 | 0.535714 | True |
| basket_outcome | forecast_only | 5 | 1 |  | -0.295402 | -0.048412 | -0.0646174 | 0.0086207 | 0.277398 | 0.663717 | True |
| basket_outcome | forecast_only | 5 | 3 |  | -0.237821 | -0.0346583 | -0.0609498 | 0.00420748 | 0.167211 | 0.566372 | True |
| basket_outcome | forecast_only | 5 | 5 |  | -0.238023 | -0.0388689 | -0.0603905 | 0.00458099 | 0.181084 | 0.575221 | True |
| basket_outcome | risk_aware | 1 | 1 |  | -0.221675 | -0.018434 | -0.0330814 | -0.00115281 | -0.0832651 | 0.4 | True |
| basket_outcome | risk_aware | 1 | 3 |  | -0.0970679 | -0.0135517 | -0.0327625 | -0.000199654 | -0.0165112 | 0.5 | True |
| basket_outcome | risk_aware | 1 | 5 |  | -0.0970679 | -0.0135517 | -0.0327625 | -0.000199654 | -0.0165112 | 0.5 | True |
| basket_outcome | risk_aware | 3 | 1 |  | -0.285621 | -0.0367576 | -0.0515097 | 0.000658309 | 0.0285561 | 0.5 | True |
| basket_outcome | risk_aware | 3 | 3 |  | -0.2004 | -0.0285964 | -0.050825 | 0.00111405 | 0.0551854 | 0.526786 | True |
| basket_outcome | risk_aware | 3 | 5 |  | -0.2004 | -0.0285964 | -0.050825 | 0.00144091 | 0.0700914 | 0.535714 | True |
| basket_outcome | risk_aware | 5 | 1 |  | -0.24585 | -0.0425998 | -0.0589257 | 0.00538843 | 0.17323 | 0.557522 | True |
| basket_outcome | risk_aware | 5 | 3 |  | -0.224382 | -0.0346583 | -0.0582878 | 0.00526233 | 0.207573 | 0.575221 | True |
| basket_outcome | risk_aware | 5 | 5 |  | -0.238023 | -0.0388689 | -0.0603905 | 0.00458099 | 0.181084 | 0.575221 | True |

## 10. Missing Artifacts And Limitations

Missing or unavailable artifacts were disclosed and were not replaced with fabricated values.

| source | path | exists |
| --- | --- | --- |
| EXP-RK-001 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-RK-001\artifacts\topn_basket_metrics.csv | False |
| EXP-RK-002 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-RK-002\artifacts\candidate_comparison.csv | False |

Charts:

| chart_note |
| --- |
| generated:candidate_policy_comparison_summary.png |
| generated:drawdown_comparison_topn_horizon.png |
| generated:hit_ratio_comparison_topn_horizon.png |
| generated:return_volatility_proxy_topn_horizon.png |
| generated:var_cvar_comparison_topn_horizon.png |

Limitations:

- Candidate rankings use historical Phase 2 forecast artifacts and local daily OHLCV files.
- The expected return proxy inherits prediction outliers from Phase 2 y_pred artifacts; this is forecast instability evidence, not a decision-layer value claim.
- Risk metrics use a 20-day realized lookback and may not capture broader regime behavior.
- Candidate baskets are equal-weight diagnostic baskets, not portfolio allocations.
- Candidate row count: `1780`.
- Basket metric row count: `18`.

## 11. Acceptance Criteria

| acceptance_criterion | status | diagnostic_only |
| --- | --- | --- |
| candidate_policy_forecast_only.yaml exists | pass | True |
| candidate_policy_risk_aware.yaml exists | pass | True |
| EXP-RK-001.yaml exists | pass | True |
| EXP-RK-002.yaml exists | pass | True |
| Candidate comparison table exists | pass | True |
| Top-N basket metrics table exists | pass | True |
| Risk-aware report generated from artifacts | pass | True |
| Every candidate row has diagnostics | pass | True |
| Every candidate row has diagnostic_only=true | pass | True |
| Risk-aware compared against forecast-only | pass | True |
| Drawdown comparison exists | pass | True |
| Hit ratio comparison exists | pass | True |
| Risk-adjusted ranking exists | pass | True |

## 12. Diagnostic-Only Disclaimer

All Phase 3 outputs are diagnostic decision-support research artifacts only. They are not BUY / SELL / HOLD advice, capital allocation guidance, broker execution instructions, portfolio recommendations, or proof of guaranteed profitable trading.

## Generated Files

- `candidate_comparison.csv`
- `topn_basket_metrics.csv`
- `risk_summary.csv`
- `risk_adjusted_ranking.csv`
- `drawdown_comparison.csv`
- `hit_ratio_comparison.csv`
- `charts/` when chart generation succeeded

## Experiments

`EXP-RK-001`, `EXP-RK-002`
