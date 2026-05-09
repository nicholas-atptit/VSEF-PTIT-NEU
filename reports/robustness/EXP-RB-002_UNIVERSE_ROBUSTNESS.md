# EXP-RB-002 Universe Robustness

Config: `EXP-RB-002`

Ticker universe sensitivity was computed from forecasting-core metric artifacts grouped by configured universes.

Rows generated: 1155
Warning or limitation rows: 0

## Preview

| universe_group | ticker | horizon | model_name | model_type | metric_name | metric_value | rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| large_cap_core | FPT | 1 | ets | model | directional_accuracy | 0.565891 | 1 |
| large_cap_core | FPT | 1 | moving_average_rule | baseline | directional_accuracy | 0.48062 | 2 |
| large_cap_core | FPT | 1 | lightgbm | model | directional_accuracy | 0.426357 | 3 |
| large_cap_core | FPT | 1 | sarimax | model | directional_accuracy | 0.426357 | 4 |
| large_cap_core | FPT | 1 | xgboost | model | directional_accuracy | 0.426357 | 5 |
| large_cap_core | FPT | 1 | persistence | baseline | directional_accuracy | 0.0310078 | 6 |
| large_cap_core | FPT | 1 | zero_return | baseline | directional_accuracy | 0.0310078 | 7 |
| large_cap_core | FPT | 1 | persistence | baseline | mae | 1.20132 | 1 |
| large_cap_core | FPT | 1 | zero_return | baseline | mae | 1.20132 | 2 |
| large_cap_core | FPT | 1 | moving_average_rule | baseline | mae | 1.8956 | 3 |
| large_cap_core | FPT | 1 | ets | model | mae | 5.20234 | 4 |
| large_cap_core | FPT | 1 | lightgbm | model | mae | 15.3805 | 5 |
| large_cap_core | FPT | 1 | xgboost | model | mae | 15.444 | 6 |
| large_cap_core | FPT | 1 | sarimax | model | mae | 9.72937e+22 | 7 |
| large_cap_core | FPT | 1 | ets | model | missing_prediction_rate | 0 | 1 |
| large_cap_core | FPT | 1 | lightgbm | model | missing_prediction_rate | 0 | 2 |
| large_cap_core | FPT | 1 | moving_average_rule | baseline | missing_prediction_rate | 0 | 3 |
| large_cap_core | FPT | 1 | persistence | baseline | missing_prediction_rate | 0 | 4 |
| large_cap_core | FPT | 1 | sarimax | model | missing_prediction_rate | 0 | 5 |
| large_cap_core | FPT | 1 | xgboost | model | missing_prediction_rate | 0 | 6 |

## Interpretation Guardrail

Null values and warnings indicate computations that were not supported by the available local artifacts.
All Phase 6 outputs are robustness and statistical research artifacts only. They are not BUY / SELL / HOLD advice, capital allocation guidance, broker execution instructions, portfolio recommendations, statistical proof of future profitability, or proof of guaranteed profitable trading.
