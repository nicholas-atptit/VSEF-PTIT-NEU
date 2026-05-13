# VN30 Hourly Available-Window Cost/Slippage Proxy Validation Report

## Source

- Prediction artifact: `outputs/vn30_hourly_available_window_benchmark/hourly/predicted_vs_actual.csv`.
- Frequency: hourly only.
- Study: VN30 hourly available-window.
- Cost grid: transaction cost bps 5, 10, 15, 20; slippage bps 5, 10, 15, 20.
- Baselines: buy_and_hold, flat_no_trade, always_up, moving_average_signal, previous_direction_signal.

## Top Proxy Rows

| slice_name | baseline | transaction_cost_bps | slippage_bps | row_count | gross_return | net_return | turnover | max_drawdown | profit_factor | win_rate | trade_count | exposure | benchmark_comparison |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regime_lightgbm_h1_insufficient_history_ACB | model_signal | 5 | 5 | 0 |  |  |  |  |  |  | 0 |  |  |
| regime_lightgbm_h1_insufficient_history_ACB | model_signal | 5 | 10 | 0 |  |  |  |  |  |  | 0 |  |  |
| regime_lightgbm_h1_insufficient_history_ACB | model_signal | 5 | 15 | 0 |  |  |  |  |  |  | 0 |  |  |
| regime_lightgbm_h1_insufficient_history_ACB | model_signal | 5 | 20 | 0 |  |  |  |  |  |  | 0 |  |  |
| regime_lightgbm_h1_insufficient_history_ACB | model_signal | 10 | 5 | 0 |  |  |  |  |  |  | 0 |  |  |
| regime_lightgbm_h1_insufficient_history_ACB | model_signal | 10 | 10 | 0 |  |  |  |  |  |  | 0 |  |  |
| regime_lightgbm_h1_insufficient_history_ACB | model_signal | 10 | 15 | 0 |  |  |  |  |  |  | 0 |  |  |
| regime_lightgbm_h1_insufficient_history_ACB | model_signal | 10 | 20 | 0 |  |  |  |  |  |  | 0 |  |  |
| regime_lightgbm_h1_insufficient_history_ACB | model_signal | 15 | 5 | 0 |  |  |  |  |  |  | 0 |  |  |
| regime_lightgbm_h1_insufficient_history_ACB | model_signal | 15 | 10 | 0 |  |  |  |  |  |  | 0 |  |  |
| regime_lightgbm_h1_insufficient_history_ACB | model_signal | 15 | 15 | 0 |  |  |  |  |  |  | 0 |  |  |
| regime_lightgbm_h1_insufficient_history_ACB | model_signal | 15 | 20 | 0 |  |  |  |  |  |  | 0 |  |  |
| regime_lightgbm_h1_insufficient_history_ACB | model_signal | 20 | 5 | 0 |  |  |  |  |  |  | 0 |  |  |
| regime_lightgbm_h1_insufficient_history_ACB | model_signal | 20 | 10 | 0 |  |  |  |  |  |  | 0 |  |  |
| regime_lightgbm_h1_insufficient_history_ACB | model_signal | 20 | 15 | 0 |  |  |  |  |  |  | 0 |  |  |
| regime_lightgbm_h1_insufficient_history_ACB | model_signal | 20 | 20 | 0 |  |  |  |  |  |  | 0 |  |  |
| regime_lightgbm_h1_insufficient_history_ACB | buy_and_hold | 5 | 5 | 0 |  |  |  |  |  |  | 0 |  |  |
| regime_lightgbm_h1_insufficient_history_ACB | buy_and_hold | 5 | 10 | 0 |  |  |  |  |  |  | 0 |  |  |
| regime_lightgbm_h1_insufficient_history_ACB | buy_and_hold | 5 | 15 | 0 |  |  |  |  |  |  | 0 |  |  |
| regime_lightgbm_h1_insufficient_history_ACB | buy_and_hold | 5 | 20 | 0 |  |  |  |  |  |  | 0 |  |  |

## Boundary

This remains proxy diagnostics. It does not establish live trading readiness because real order book depth, fills, liquidity filters, and execution policy are not implemented.
