# EXP-ST-002 Bootstrap CI

Config: `EXP-ST-002`

Bootstrap confidence intervals were computed from basket period return rows with a fixed seed.

Rows generated: 108
Warning or limitation rows: 0

## Preview

| candidate_type | top_n | horizon | metric_name | estimate | ci_lower | ci_upper | sample_size | warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| forecast_only | 1 | 1 | average_realized_return | -0.000724738 | -0.00318455 | 0.00183747 | 120 |  |
| forecast_only | 1 | 1 | hit_ratio | 0.425 | 0.341458 | 0.516667 | 120 |  |
| forecast_only | 1 | 1 | return_volatility_proxy | -0.0508879 | -0.218983 | 0.134914 | 120 |  |
| forecast_only | 1 | 1 | max_drawdown | -0.146624 | -0.364588 | -0.0764094 | 120 |  |
| forecast_only | 1 | 1 | var_95 | -0.0174006 | -0.036042 | -0.0152976 | 120 |  |
| forecast_only | 1 | 1 | cvar_95 | -0.0388861 | -0.0525142 | -0.0210503 | 120 |  |
| forecast_only | 1 | 3 | average_realized_return | 0.00496341 | 0.000515597 | 0.009476 | 112 |  |
| forecast_only | 1 | 3 | hit_ratio | 0.589286 | 0.5 | 0.6875 | 112 |  |
| forecast_only | 1 | 3 | return_volatility_proxy | 0.202464 | 0.0214355 | 0.424016 | 112 |  |
| forecast_only | 1 | 3 | max_drawdown | -0.209129 | -0.288743 | -0.0686745 | 112 |  |
| forecast_only | 1 | 3 | var_95 | -0.028795 | -0.0472148 | -0.0211311 | 112 |  |
| forecast_only | 1 | 3 | cvar_95 | -0.0498929 | -0.0747995 | -0.0280201 | 112 |  |
| forecast_only | 1 | 5 | average_realized_return | 0.0086207 | 0.00300101 | 0.0144107 | 113 |  |
| forecast_only | 1 | 5 | hit_ratio | 0.663717 | 0.584071 | 0.743363 | 113 |  |
| forecast_only | 1 | 5 | return_volatility_proxy | 0.277398 | 0.0936251 | 0.492721 | 113 |  |
| forecast_only | 1 | 5 | max_drawdown | -0.295402 | -0.300257 | -0.0797395 | 113 |  |
| forecast_only | 1 | 5 | var_95 | -0.048412 | -0.0569439 | -0.0273199 | 113 |  |
| forecast_only | 1 | 5 | cvar_95 | -0.0646174 | -0.0834827 | -0.0441309 | 113 |  |
| forecast_only | 3 | 1 | average_realized_return | -0.000199654 | -0.00230089 | 0.0018439 | 120 |  |
| forecast_only | 3 | 1 | hit_ratio | 0.5 | 0.408333 | 0.591667 | 120 |  |

## Interpretation Guardrail

Null values and warnings indicate computations that were not supported by the available local artifacts.
All Phase 6 outputs are robustness and statistical research artifacts only. They are not BUY / SELL / HOLD advice, capital allocation guidance, broker execution instructions, portfolio recommendations, statistical proof of future profitability, or proof of guaranteed profitable trading.
