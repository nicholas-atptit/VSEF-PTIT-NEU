# VN30 Hourly 2015 Data Readiness Plan

## Design

The active VN30/index hourly data design starts at `2015-01-01 00:00:00`. Earlier 2005/2006 full-history designs are superseded for this track and must not be used as the required start.

- Frequency: hourly only, `1H`.
- Training/history period: `2015-01-01 00:00:00` to `2024-12-31 23:59:59`.
- Evaluation/comparison start: `2025-01-01 00:00:00`.
- Evaluation end: provider-current/latest available timestamp, not a future date.
- Provider path: `src.data.providers.vn_price_gateway.fetch_price_history`.
- Daily data: not allowed.
- Daily-to-hourly resampling: not allowed.
- Synthetic missing bars: not allowed.

## Supported Index Codes

Use only provider-supported names unless a later provider probe proves otherwise:

- `VNINDEX`
- `HNXINDEX`
- `UPCOMINDEX`
- `VN30`
- `HNX30`
- `VN100`

Do not use `VN30INDEX`; use `VN30`. Do not use `VNXALL` under the current provider evidence.

## Workflow

1. Reset old working artifacts with `scripts/research/reset_vn30_hourly_2015_workspace.py`.
2. Fetch index hourly data from 2015 with `scripts/research/fetch_supported_indices_hourly_gateway_2015.py`.
3. Validate index hourly data with `scripts/research/validate_supported_indices_hourly_gateway_2015.py`.
4. Fetch frozen VN30 stock hourly data from 2015 with `scripts/research/fetch_vn30_stocks_hourly_gateway_2015.py`.
5. Validate VN30 stock hourly data with `scripts/research/validate_vn30_stocks_hourly_gateway_2015.py`.
6. Build readiness only with `scripts/research/build_vn30_2015_benchmark_readiness_manifest.py`.

Benchmarking can proceed only after the readiness manifest says yes. This plan does not run benchmark, model training, confidence sweeps, regime diagnostics, cost/slippage diagnostics, paper generation, or DOCX generation.
