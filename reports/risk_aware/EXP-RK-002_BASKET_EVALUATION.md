# EXP-RK-002 Basket Evaluation

## Purpose

Evaluate realized outcomes for equal-weight diagnostic candidate baskets formed from forecast-only and risk-aware policies.

## Top-N Basket Metrics

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

## Drawdown Comparison

| horizon | top_n | forecast_only_max_drawdown | risk_aware_max_drawdown | drawdown_reduction_vs_forecast_only | diagnostic_only |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | -0.146624 | -0.221675 | -0.0750505 | True |
| 1 | 3 | -0.0970679 | -0.0970679 | -4.44089e-16 | True |
| 1 | 5 | -0.0970679 | -0.0970679 | -4.44089e-16 | True |
| 3 | 1 | -0.209129 | -0.285621 | -0.0764923 | True |
| 3 | 3 | -0.2004 | -0.2004 | 0 | True |
| 3 | 5 | -0.2004 | -0.2004 | 0 | True |
| 5 | 1 | -0.295402 | -0.24585 | 0.0495518 | True |
| 5 | 3 | -0.237821 | -0.224382 | 0.0134384 | True |
| 5 | 5 | -0.238023 | -0.238023 | 0 | True |

## Hit Ratio Comparison

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

## Disclaimer

All Phase 3 outputs are diagnostic decision-support research artifacts only. They are not BUY / SELL / HOLD advice, capital allocation guidance, broker execution instructions, portfolio recommendations, or proof of guaranteed profitable trading.
