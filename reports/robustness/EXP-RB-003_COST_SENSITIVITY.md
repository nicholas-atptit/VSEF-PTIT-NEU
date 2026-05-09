# EXP-RB-003 Cost Sensitivity

Config: `EXP-RB-003`

Cost/slippage sensitivity was computed from diagnostic basket period returns when available.

Rows generated: 90
Warning or limitation rows: 0

## Preview

| cost_scenario_id | candidate_type | top_n | horizon | gross_average_realized_return | net_average_realized_return | net_hit_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| cost_0bps | forecast_only | 1 | 1 | -0.000724738 | -0.000724738 | 0.425 |
| cost_0bps | forecast_only | 1 | 3 | 0.00496341 | 0.00496341 | 0.589286 |
| cost_0bps | forecast_only | 1 | 5 | 0.0086207 | 0.0086207 | 0.663717 |
| cost_0bps | forecast_only | 3 | 1 | -0.000199654 | -0.000199654 | 0.5 |
| cost_0bps | forecast_only | 3 | 3 | 0.00144826 | 0.00144826 | 0.535714 |
| cost_0bps | forecast_only | 3 | 5 | 0.00420748 | 0.00420748 | 0.566372 |
| cost_0bps | forecast_only | 5 | 1 | -0.000199654 | -0.000199654 | 0.5 |
| cost_0bps | forecast_only | 5 | 3 | 0.00144091 | 0.00144091 | 0.535714 |
| cost_0bps | forecast_only | 5 | 5 | 0.00458099 | 0.00458099 | 0.575221 |
| cost_0bps | risk_aware | 1 | 1 | -0.00115281 | -0.00115281 | 0.4 |
| cost_0bps | risk_aware | 1 | 3 | 0.000658309 | 0.000658309 | 0.5 |
| cost_0bps | risk_aware | 1 | 5 | 0.00538843 | 0.00538843 | 0.557522 |
| cost_0bps | risk_aware | 3 | 1 | -0.000199654 | -0.000199654 | 0.5 |
| cost_0bps | risk_aware | 3 | 3 | 0.00111405 | 0.00111405 | 0.526786 |
| cost_0bps | risk_aware | 3 | 5 | 0.00526233 | 0.00526233 | 0.575221 |
| cost_0bps | risk_aware | 5 | 1 | -0.000199654 | -0.000199654 | 0.5 |
| cost_0bps | risk_aware | 5 | 3 | 0.00144091 | 0.00144091 | 0.535714 |
| cost_0bps | risk_aware | 5 | 5 | 0.00458099 | 0.00458099 | 0.575221 |
| cost_5bps | forecast_only | 1 | 1 | -0.000724738 | -0.00172474 | 0.416667 |
| cost_5bps | forecast_only | 1 | 3 | 0.00496341 | 0.00396341 | 0.589286 |

## Interpretation Guardrail

Null values and warnings indicate computations that were not supported by the available local artifacts.
All Phase 6 outputs are robustness and statistical research artifacts only. They are not BUY / SELL / HOLD advice, capital allocation guidance, broker execution instructions, portfolio recommendations, statistical proof of future profitability, or proof of guaranteed profitable trading.
