# EXP-ST-001 Diebold-Mariano Test

Config: `EXP-ST-001`

DM tests compare aligned model and baseline forecast errors from local prediction artifacts.

Rows generated: 464
Warning or limitation rows: 36

## Preview

| source_experiment | ticker | horizon | model_name | baseline_name | loss | dm_statistic | p_value | effect_size | warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-FC-001 | ACB | 1 | sarimax | persistence | squared |  |  |  | invalid_or_zero_loss_differential_variance |
| EXP-FC-001 | ACB | 1 | sarimax | persistence | absolute | 1.21417 | 0.226919 | 0.106486 |  |
| EXP-FC-001 | ACB | 1 | sarimax | zero_return | squared |  |  |  | invalid_or_zero_loss_differential_variance |
| EXP-FC-001 | ACB | 1 | sarimax | zero_return | absolute | 1.21417 | 0.226919 | 0.106486 |  |
| EXP-FC-001 | ACB | 1 | ets | persistence | squared | 9.67092 | 6.08769e-17 | 0.84817 |  |
| EXP-FC-001 | ACB | 1 | ets | persistence | absolute | 12.4849 | 6.86161e-24 | 1.09497 |  |
| EXP-FC-001 | ACB | 1 | ets | zero_return | squared | 9.67092 | 6.08769e-17 | 0.84817 |  |
| EXP-FC-001 | ACB | 1 | ets | zero_return | absolute | 12.4849 | 6.86161e-24 | 1.09497 |  |
| EXP-FC-001 | ACB | 1 | xgboost | persistence | squared | 15.2295 | 1.59926e-30 | 1.33568 |  |
| EXP-FC-001 | ACB | 1 | xgboost | persistence | absolute | 25.3391 | 1.02296e-51 | 2.22232 |  |
| EXP-FC-001 | ACB | 1 | xgboost | zero_return | squared | 15.2295 | 1.59926e-30 | 1.33568 |  |
| EXP-FC-001 | ACB | 1 | xgboost | zero_return | absolute | 25.3391 | 1.02296e-51 | 2.22232 |  |
| EXP-FC-001 | ACB | 1 | lightgbm | persistence | squared | 14.3007 | 2.61111e-28 | 1.25422 |  |
| EXP-FC-001 | ACB | 1 | lightgbm | persistence | absolute | 23.3716 | 5.1296e-48 | 2.04976 |  |
| EXP-FC-001 | ACB | 1 | lightgbm | zero_return | squared | 14.3007 | 2.61111e-28 | 1.25422 |  |
| EXP-FC-001 | ACB | 1 | lightgbm | zero_return | absolute | 23.3716 | 5.1296e-48 | 2.04976 |  |
| EXP-FC-001 | DGC | 1 | sarimax | persistence | squared | 1.50114 | 0.135783 | 0.131654 |  |
| EXP-FC-001 | DGC | 1 | sarimax | persistence | absolute | 1.48587 | 0.139773 | 0.130315 |  |
| EXP-FC-001 | DGC | 1 | sarimax | zero_return | squared | 1.50114 | 0.135783 | 0.131654 |  |
| EXP-FC-001 | DGC | 1 | sarimax | zero_return | absolute | 1.48587 | 0.139773 | 0.130315 |  |

## Interpretation Guardrail

Null values and warnings indicate computations that were not supported by the available local artifacts.
All Phase 6 outputs are robustness and statistical research artifacts only. They are not BUY / SELL / HOLD advice, capital allocation guidance, broker execution instructions, portfolio recommendations, statistical proof of future profitability, or proof of guaranteed profitable trading.
