# EXP-RB-001 Window Robustness

Config: `EXP-RB-001`

Train/test window sensitivity was evaluated only where an exact local source artifact matched the requested train/test split.

Rows generated: 632
Warning or limitation rows: 2

## Preview

| setting_id | ticker | horizon | model_name | model_type | metric_name | metric_value | rank | robustness_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| window_2023_train_2024_test_h1 | ACB | 1 | ets | model | directional_accuracy | 0.527132 | 1 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | lightgbm | model | directional_accuracy | 0.44186 | 2 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | moving_average_rule | baseline | directional_accuracy | 0.44186 | 3 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | sarimax | model | directional_accuracy | 0.44186 | 4 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | xgboost | model | directional_accuracy | 0.44186 | 5 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | persistence | baseline | directional_accuracy | 0.108527 | 6 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | zero_return | baseline | directional_accuracy | 0.108527 | 7 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | persistence | baseline | mae | 0.147597 | 1 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | zero_return | baseline | mae | 0.147597 | 2 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | moving_average_rule | baseline | mae | 0.234589 | 3 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | ets | model | mae | 0.550385 | 4 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | lightgbm | model | mae | 1.18948 | 5 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | xgboost | model | mae | 1.28213 | 6 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | sarimax | model | mae | 1.12475e+91 | 7 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | persistence | baseline | mape | 0.712166 | 1 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | zero_return | baseline | mape | 0.712166 | 2 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | moving_average_rule | baseline | mape | 1.12911 | 3 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | ets | model | mape | 2.64898 | 4 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | lightgbm | model | mape | 5.675 | 5 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | xgboost | model | mape | 6.12133 | 6 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |

## Interpretation Guardrail

Null values and warnings indicate computations that were not supported by the available local artifacts.
All Phase 6 outputs are robustness and statistical research artifacts only. They are not BUY / SELL / HOLD advice, capital allocation guidance, broker execution instructions, portfolio recommendations, statistical proof of future profitability, or proof of guaranteed profitable trading.
