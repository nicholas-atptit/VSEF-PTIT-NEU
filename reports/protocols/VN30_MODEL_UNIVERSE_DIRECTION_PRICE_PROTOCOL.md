# VN30 Model Universe Direction + Price Forecasting Protocol

## Scope

- This benchmark is offline diagnostic-only.
- Scope is VN30 stock hourly forecasting only.
- VN100 is out of scope.
- Direction and price/return targets are separate benchmark families.
- Price-level forecasting and return forecasting must be reported separately.
- Index data may be used only as lagged market-context features or market-relative target context.
- No index-as-stock claim is made.

## Claim Boundary

- No raw trading recommendation is made.
- No BUY/SELL signal is claimed.
- No profitability claim is made.
- No investment recommendation is made.
- No live deployment or production-readiness claim is made.
- Final-ranked rows are exploratory_not_claimable.
- Claimable interpretation, if any, must be validation-governed and future-blind-safe.

## Split Discipline

Every evaluated row must enforce both feature_timestamp and target_timestamp boundaries:

- Train: `feature_timestamp <= 2023-12-31 23:59:59` and `target_timestamp <= 2023-12-31 23:59:59`.
- Validation: both timestamps inside calendar year 2024.
- Final: both timestamps on or after `2025-01-01 00:00:00`.

Every result row must include ticker, feature_timestamp, target_timestamp, horizon, target_variant, and split lineage through the generated audit artifacts.

## Targets

Direction targets:

- `absolute_direction`
- `market_relative_vn30`
- `market_relative_vnindex`
- `top_quantile_forward_return`

Price/return targets:

- `forward_simple_return_h`
- `forward_log_return_h`
- `future_close_h`
- `market_excess_return_h`
- `volatility_adjusted_return_h`

Horizons: h5, h10, h20, h40, h60.

## Required Reporting

Final reporting must include:

- strongest direction baselines and lift/error improvement;
- strongest price/return baselines and error improvement;
- validation-selected direction result;
- validation-selected price/return result;
- separate exploratory final leaderboards;
- skipped model families and skipped reasons;
- comparison against the 61.61% absolute-direction classical champion only on comparable absolute-direction scope;
- comparison against the 64.44% QML V8 market-relative result only on comparable market_relative_vn30 scope.

No direction accuracy may be mixed with price RMSE, MAE, or MAPE. No price-level RMSE may be mixed with return-forecast metrics.
