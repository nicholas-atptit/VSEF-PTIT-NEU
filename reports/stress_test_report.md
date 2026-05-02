# Stress Test Report

## Scenario Assumptions
- `volatility_shock`: Amplify realized volatility and risk metrics after the midpoint.
- `drawdown_shock`: Inject a concentrated negative block loss and deeper drawdown.
- `liquidity_cost_shock`: Increase fees and slippage to simulate thinner liquidity.
- `regime_persistence_shock`: Force crisis-like regime persistence after the first shock.

## Summary
| stress_scenario | benchmark_mode | stressed_sharpe | stressed_max_drawdown | stressed_tail_loss | stressed_exposure | stressed_turnover | delta_sharpe | delta_tail_loss | delta_drawdown | delta_exposure | regime_reaction_speed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drawdown_shock | forecast_plus_risk_and_regime | -1.3 | -0.1888 | -0.0229 | 0.4951 | 11.0 | -1.3 | -0.008 | -0.0582 | 0.0 | 0.0 |
| drawdown_shock | forecast_plus_risk_features | -3.32 | -0.2202 | -0.0229 | 0.5728 | 13.0 | -1.17 | -0.0069 | -0.0702 | 0.0 | 0.0 |
| drawdown_shock | full_system | 0.34 | -0.0553 | -0.01 | 0.2913 | 6.5 | 1.16 | 0.0034 | 0.0333 | -0.0509 | 0.0 |
| drawdown_shock | legacy_forecast_only | -1.93 | -0.2155 | -0.0229 | 0.4563 | 9.0 | -1.53 | -0.0078 | -0.0707 | 0.0 | nan |
| liquidity_cost_shock | forecast_plus_risk_and_regime | -1.96 | -0.2055 | -0.0198 | 0.4951 | 11.0 | -1.96 | -0.0049 | -0.0749 | 0.0 | 0.0 |
| liquidity_cost_shock | forecast_plus_risk_features | -4.01 | -0.2305 | -0.0205 | 0.5728 | 13.0 | -1.86 | -0.0045 | -0.0805 | 0.0 | 0.0 |
| liquidity_cost_shock | full_system | -3.09 | -0.1727 | -0.0171 | 0.3422 | 11.0 | -2.27 | -0.0037 | -0.0841 | 0.0 | 0.0 |
| liquidity_cost_shock | legacy_forecast_only | -2.07 | -0.2041 | -0.0173 | 0.4563 | 9.0 | -1.67 | -0.0022 | -0.0593 | 0.0 | nan |
| regime_persistence_shock | forecast_plus_risk_and_regime | -0.51 | -0.1378 | -0.0156 | 0.4951 | 11.0 | -0.51 | -0.0007 | -0.0072 | 0.0 | 0.0 |
| regime_persistence_shock | forecast_plus_risk_features | -3.08 | -0.1723 | -0.0169 | 0.5728 | 13.0 | -0.93 | -0.0009 | -0.0223 | 0.0 | 0.0 |
| regime_persistence_shock | full_system | 0.25 | -0.0406 | -0.0064 | 0.2146 | 6.8 | 1.07 | 0.007 | 0.048 | -0.1276 | 0.0 |
| regime_persistence_shock | legacy_forecast_only | -0.98 | -0.1536 | -0.0155 | 0.4563 | 9.0 | -0.58 | -0.0004 | -0.0088 | 0.0 | 0.0 |
| volatility_shock | forecast_plus_risk_and_regime | 1.01 | -0.1569 | -0.0189 | 0.4951 | 11.0 | 1.01 | -0.004 | -0.0263 | 0.0 | 0.0 |
| volatility_shock | forecast_plus_risk_features | -1.57 | -0.1785 | -0.0238 | 0.5728 | 13.0 | 0.58 | -0.0078 | -0.0285 | 0.0 | 0.0 |
| volatility_shock | full_system | 1.08 | -0.0492 | -0.0066 | 0.1951 | 5.2 | 1.9 | 0.0068 | 0.0394 | -0.1471 | 0.0 |
| volatility_shock | legacy_forecast_only | 0.71 | -0.1735 | -0.0182 | 0.4563 | 9.0 | 1.11 | -0.0031 | -0.0287 | 0.0 | 0.0 |

## Output Files
- Detail CSV: `K:\Repos\VSEF-PTIT-NEU\tmp\pytest_tmpdirs\case_44fde54b03774e4096595e7060107f73\reports\stress_alignment.csv`
- Summary CSV: `K:\Repos\VSEF-PTIT-NEU\tmp\pytest_tmpdirs\case_44fde54b03774e4096595e7060107f73\reports\stress_alignment_summary.csv`
- JSON: `K:\Repos\VSEF-PTIT-NEU\tmp\pytest_tmpdirs\case_44fde54b03774e4096595e7060107f73\reports\stress_alignment.json`