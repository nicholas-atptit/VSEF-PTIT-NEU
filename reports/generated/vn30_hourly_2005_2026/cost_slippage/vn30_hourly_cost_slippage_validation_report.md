# VN30 Hourly Cost/Slippage Proxy Validation Report

## Source

- Prediction artifact: `outputs/vn30_hourly_official_2005_2026_traincutoff/hourly/predicted_vs_actual.csv`.
- Frequency: hourly only.
- Cost grid: transaction cost bps 5, 10, 15, 20; slippage bps 5, 10, 15, 20.
- Baselines: buy-and-hold, flat/no-trade, always-up, moving-average signal, previous-direction signal.

## Top Proxy Rows

No cost/slippage proxy diagnostics are available.

## Boundary

This remains proxy diagnostics. It does not establish live trading readiness because real entry/exit execution prices, liquidity filters, and fill assumptions are not implemented.
