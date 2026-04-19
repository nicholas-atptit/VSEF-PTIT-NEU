# Stress Test Report

## Scenario Assumptions
- `volatility_shock`: Amplify realized volatility and risk metrics after the midpoint.
- `drawdown_shock`: Inject a concentrated negative block loss and deeper drawdown.
- `liquidity_cost_shock`: Increase fees and slippage to simulate thinner liquidity.
- `regime_persistence_shock`: Force crisis-like regime persistence after the first shock.

## Summary
| stress_scenario | benchmark_mode | stressed_sharpe | stressed_max_drawdown | stressed_tail_loss | stressed_exposure | stressed_turnover | delta_sharpe | delta_tail_loss | delta_drawdown | delta_exposure | regime_reaction_speed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drawdown_shock | forecast_plus_risk_and_regime | -6.6 | -0.3059 | -0.021 | 0.4127 | 15.0 | -0.68 | -0.0028 | -0.0482 | 0.0 | 0.0 |
| drawdown_shock | forecast_plus_risk_features | -6.6 | -0.3059 | -0.021 | 0.4127 | 15.0 | -0.68 | -0.0028 | -0.0482 | 0.0 | 0.0 |
| drawdown_shock | full_system | -5.82 | -0.1588 | -0.0137 | 0.1746 | 8.25 | -0.33 | 0.0024 | 0.0254 | -0.0417 | 0.0 |
| drawdown_shock | legacy_forecast_only | -6.6 | -0.3059 | -0.021 | 0.4127 | 15.0 | -0.68 | -0.0028 | -0.0482 | 0.0 | nan |
| liquidity_cost_shock | forecast_plus_risk_and_regime | -8.11 | -0.3485 | -0.0195 | 0.4127 | 15.0 | -2.19 | -0.0013 | -0.0908 | 0.0 | 0.0 |
| liquidity_cost_shock | forecast_plus_risk_features | -8.11 | -0.3485 | -0.0195 | 0.4127 | 15.0 | -2.19 | -0.0013 | -0.0908 | 0.0 | 0.0 |
| liquidity_cost_shock | full_system | -7.48 | -0.2693 | -0.0177 | 0.2163 | 12.75 | -1.99 | -0.0016 | -0.0851 | 0.0 | 0.0 |
| liquidity_cost_shock | legacy_forecast_only | -8.11 | -0.3485 | -0.0195 | 0.4127 | 15.0 | -2.19 | -0.0013 | -0.0908 | 0.0 | nan |
| regime_persistence_shock | forecast_plus_risk_and_regime | -6.97 | -0.3064 | -0.0197 | 0.4127 | 15.0 | -1.05 | -0.0015 | -0.0487 | 0.0 | 0.0 |
| regime_persistence_shock | forecast_plus_risk_features | -6.97 | -0.3064 | -0.0197 | 0.4127 | 15.0 | -1.05 | -0.0015 | -0.0487 | 0.0 | 0.0 |
| regime_persistence_shock | full_system | -5.75 | -0.1433 | -0.0119 | 0.1567 | 8.25 | -0.26 | 0.0042 | 0.0409 | -0.0596 | 0.0 |
| regime_persistence_shock | legacy_forecast_only | -6.97 | -0.3064 | -0.0197 | 0.4127 | 15.0 | -1.05 | -0.0015 | -0.0487 | 0.0 | 0.0 |
| volatility_shock | forecast_plus_risk_and_regime | -4.94 | -0.3656 | -0.0345 | 0.4127 | 15.0 | 0.98 | -0.0163 | -0.1079 | 0.0 | 0.0 |
| volatility_shock | forecast_plus_risk_features | -4.94 | -0.3656 | -0.0345 | 0.4127 | 15.0 | 0.98 | -0.0163 | -0.1079 | 0.0 | 0.0 |
| volatility_shock | full_system | -5.45 | -0.1672 | -0.0147 | 0.15 | 5.95 | 0.04 | 0.0014 | 0.017 | -0.0663 | 0.0 |
| volatility_shock | legacy_forecast_only | -4.94 | -0.3656 | -0.0345 | 0.4127 | 15.0 | 0.98 | -0.0163 | -0.1079 | 0.0 | 0.0 |

## Output Files
- Detail CSV: `reports/stress_alignment.csv`
- Summary CSV: `reports/stress_alignment_summary.csv`
- JSON: `reports/stress_alignment.json`