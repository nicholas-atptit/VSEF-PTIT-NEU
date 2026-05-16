# VN30 Hourly 2015 Cost/Slippage Proxy Diagnostics

## Source

- Prediction artifact: `outputs/vn30_hourly_2015_jan2025_benchmark/hourly/predicted_vs_actual.csv`.
- Frequency: hourly only.
- Signal proxy: model predicted direction applied to realized benchmark prediction returns.
- Strategies: long_flat, direction_following_long_short.
- Transaction cost grid bps: 0, 5, 10, 20.
- Slippage grid bps: 0, 5, 10, 20.

## Top Proxy Rows

| model | horizon | strategy | cost_bps | slippage_bps | net_return | max_drawdown | win_rate | profit_factor | trade_count | exposure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_forest | 20 | long_flat | 0 | 0 | 26334.87% | -79.43% | 33.44% | 1.844 | 522 | 60.21% |
| random_forest | 20 | long_flat | 0 | 5 | 25858.38% | -79.46% | 33.44% | 1.839 | 522 | 60.21% |
| random_forest | 20 | long_flat | 5 | 0 | 25858.38% | -79.46% | 33.44% | 1.839 | 522 | 60.21% |
| random_forest | 20 | long_flat | 0 | 10 | 25390.44% | -79.49% | 33.44% | 1.834 | 522 | 60.21% |
| random_forest | 20 | long_flat | 5 | 5 | 25390.44% | -79.49% | 33.44% | 1.834 | 522 | 60.21% |
| random_forest | 20 | long_flat | 10 | 0 | 25390.44% | -79.49% | 33.44% | 1.834 | 522 | 60.21% |
| random_forest | 20 | long_flat | 5 | 10 | 24930.88% | -79.52% | 33.44% | 1.829 | 522 | 60.21% |
| random_forest | 20 | long_flat | 10 | 5 | 24930.88% | -79.52% | 33.44% | 1.829 | 522 | 60.21% |
| random_forest | 20 | long_flat | 0 | 20 | 24479.55% | -79.55% | 33.42% | 1.824 | 522 | 60.21% |
| random_forest | 20 | long_flat | 10 | 10 | 24479.55% | -79.55% | 33.42% | 1.824 | 522 | 60.21% |
| random_forest | 20 | long_flat | 20 | 0 | 24479.55% | -79.55% | 33.42% | 1.824 | 522 | 60.21% |
| random_forest | 20 | long_flat | 5 | 20 | 24036.32% | -79.58% | 33.40% | 1.819 | 522 | 60.21% |
| random_forest | 20 | long_flat | 20 | 5 | 24036.32% | -79.58% | 33.40% | 1.819 | 522 | 60.21% |
| random_forest | 20 | long_flat | 10 | 20 | 23601.03% | -79.61% | 33.36% | 1.814 | 522 | 60.21% |
| random_forest | 20 | long_flat | 20 | 10 | 23601.03% | -79.61% | 33.36% | 1.814 | 522 | 60.21% |
| random_forest | 20 | long_flat | 20 | 20 | 22753.72% | -79.67% | 33.31% | 1.805 | 522 | 60.21% |
| lightgbm | 20 | long_flat | 0 | 0 | 15818.23% | -79.54% | 30.53% | 1.879 | 494 | 53.84% |
| lightgbm | 20 | long_flat | 0 | 5 | 15539.18% | -79.58% | 30.53% | 1.874 | 494 | 53.84% |
| lightgbm | 20 | long_flat | 5 | 0 | 15539.18% | -79.58% | 30.53% | 1.874 | 494 | 53.84% |
| xgboost | 20 | long_flat | 0 | 0 | 15349.84% | -78.84% | 30.16% | 1.93 | 501 | 53.25% |

## Standard 10 bps Cost + 10 bps Slippage Diagnostic

- Best row: random_forest h=20 long_flat.
- Net return proxy: 24479.55%.
- Max drawdown proxy: -79.55%.
- Win rate: 33.42%.
- Trade count: 522.
- Exposure: 60.21%.

## Boundary

- This is a signal diagnostic proxy, not an executable backtest.
- It does not model order book depth, fill probability, liquidity filters, position sizing, borrow costs, taxes, or real execution constraints.
- No trading-readiness or profitability claim is made.
