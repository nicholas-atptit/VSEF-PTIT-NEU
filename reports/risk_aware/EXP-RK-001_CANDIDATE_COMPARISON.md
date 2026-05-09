# EXP-RK-001 Candidate Comparison

## Purpose

Compare forecast-only ranking against risk-aware ranking using the same Phase 2 forecast evidence where possible.

## Candidate Rows

- Total candidate rows: `1780`
- Forecast-only rows: `890`
- Risk-aware rows: `890`
- Every row is diagnostic-only and not investment advice.

## Source Artifact Evidence

| experiment_id | path | relative_path | exists | required |
| --- | --- | --- | --- | --- |
| EXP-FC-001 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-001\predictions\predictions.csv | predictions/predictions.csv | True | True |
| EXP-FC-001 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-001\metrics\metrics.csv | metrics/metrics.csv | True | False |
| EXP-FC-001 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-001\manifests\run_manifest.json | manifests/run_manifest.json | True | False |
| EXP-FC-001 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-001\logs\run.log | logs/run.log | True | False |
| EXP-FC-001 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-001\logs\errors.log | logs/errors.log | True | False |
| EXP-FC-003 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-003\predictions\predictions.csv | predictions/predictions.csv | True | True |
| EXP-FC-003 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-003\metrics\metrics.csv | metrics/metrics.csv | True | False |
| EXP-FC-003 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-003\manifests\run_manifest.json | manifests/run_manifest.json | True | False |
| EXP-FC-003 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-003\logs\run.log | logs/run.log | True | False |
| EXP-FC-003 | K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-003\logs\errors.log | logs/errors.log | True | False |

## Candidate Overlap

| experiment_id | policy_id | candidate_type | horizon | metric_name | metric_value | sample_size | notes | diagnostic_only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-RK-001 | forecast_only_vs_risk_aware | policy_comparison | 1 | candidate_overlap_rate | 1 | 120 | average_ticker_overlap_per_candidate_date | True |
| EXP-RK-001 | forecast_only_vs_risk_aware | policy_comparison | 3 | candidate_overlap_rate | 1 | 112 | average_ticker_overlap_per_candidate_date | True |
| EXP-RK-001 | forecast_only_vs_risk_aware | policy_comparison | 5 | candidate_overlap_rate | 1 | 113 | average_ticker_overlap_per_candidate_date | True |

## Candidate Preview

| candidate_type | candidate_date | ticker | horizon | rank | candidate_score | expected_return_proxy | realized_volatility | max_drawdown | diagnostic_only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| forecast_only | 2024-07-02 | FPT | 5 | 1 | 0.00428391 | 0.00428391 | 0.0178793 | -0.059563 | True |
| forecast_only | 2024-07-04 | MWG | 1 | 1 | 0.0280573 | 0.0280573 | 0.0163487 | -0.0311551 | True |
| forecast_only | 2024-07-08 | MWG | 1 | 1 | 0.130005 | 0.130005 | 0.0166788 | -0.0311551 | True |
| forecast_only | 2024-07-10 | MWG | 1 | 1 | 0.497218 | 0.497218 | 0.0171755 | -0.0311551 | True |
| forecast_only | 2024-07-12 | MWG | 1 | 1 | 1.44112 | 1.44112 | 0.016738 | -0.0328539 | True |
| forecast_only | 2024-07-16 | MWG | 1 | 1 | 4.02247 | 4.02247 | 0.0164426 | -0.0433005 | True |
| forecast_only | 2024-07-18 | MWG | 1 | 1 | 10.7775 | 10.7775 | 0.0169604 | -0.0522332 | True |
| forecast_only | 2024-07-18 | FPT | 5 | 1 | 0.00207131 | 0.00207131 | 0.0186486 | -0.0837701 | True |
| forecast_only | 2024-07-19 | FPT | 5 | 1 | 0.0173001 | 0.0173001 | 0.0180719 | -0.0973795 | True |
| forecast_only | 2024-07-22 | MWG | 1 | 1 | 29.0963 | 29.0963 | 0.0160796 | -0.0522332 | True |
| forecast_only | 2024-07-22 | DGC | 1 | 2 | 0.00521978 | 0.00521978 | 0.0152904 | -0.109389 | True |
| forecast_only | 2024-07-22 | DGC | 3 | 1 | 0.0163205 | 0.0163205 | 0.0152904 | -0.109389 | True |
| forecast_only | 2024-07-22 | FPT | 5 | 1 | 0.0268158 | 0.0268158 | 0.0173069 | -0.110989 | True |
| forecast_only | 2024-07-22 | DGC | 5 | 2 | 0.0251018 | 0.0251018 | 0.0152904 | -0.109389 | True |
| forecast_only | 2024-07-23 | DGC | 1 | 1 | 0.0404931 | 0.0404931 | 0.01807 | -0.153112 | True |
| forecast_only | 2024-07-23 | DGC | 3 | 1 | 0.0453995 | 0.0453995 | 0.01807 | -0.153112 | True |
| forecast_only | 2024-07-23 | DGC | 5 | 1 | 0.0526916 | 0.0526916 | 0.01807 | -0.153112 | True |
| forecast_only | 2024-07-23 | FPT | 5 | 2 | 0.022015 | 0.022015 | 0.0173031 | -0.110989 | True |
| forecast_only | 2024-07-24 | MWG | 1 | 1 | 83.8523 | 83.8523 | 0.019257 | -0.0835731 | True |
| forecast_only | 2024-07-24 | DGC | 1 | 2 | 0.013813 | 0.013813 | 0.0185771 | -0.153112 | True |

## Disclaimer

All Phase 3 outputs are diagnostic decision-support research artifacts only. They are not BUY / SELL / HOLD advice, capital allocation guidance, broker execution instructions, portfolio recommendations, or proof of guaranteed profitable trading.
