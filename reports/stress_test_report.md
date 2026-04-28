# Stress Test Report

## Scenario Assumptions
- `volatility_shock`: Amplify realized volatility and risk metrics after the midpoint.
- `drawdown_shock`: Inject a concentrated negative block loss and deeper drawdown.
- `liquidity_cost_shock`: Increase fees and slippage to simulate thinner liquidity.
- `regime_persistence_shock`: Force crisis-like regime persistence after the first shock.

## Summary
| stress_scenario | benchmark_mode | stressed_sharpe | stressed_max_drawdown | stressed_tail_loss | stressed_exposure | stressed_turnover | delta_sharpe | delta_tail_loss | delta_drawdown | delta_exposure | regime_reaction_speed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drawdown_shock | forecast_plus_risk_and_regime | -1.55 | -0.2024 | -0.0232 | 0.5098 | 11.0 | -1.59 | -0.0083 | -0.0718 | 0.0 | 0.0 |
| drawdown_shock | forecast_plus_risk_features | -3.3 | -0.2201 | -0.0227 | 0.5882 | 13.0 | -1.18 | -0.0067 | -0.0701 | 0.0 | 0.0 |
| drawdown_shock | full_system | 0.27 | -0.0588 | -0.01 | 0.3039 | 6.5 | 1.03 | 0.0034 | 0.0298 | -0.0515 | 0.0 |
| drawdown_shock | legacy_forecast_only | -1.91 | -0.2154 | -0.0227 | 0.4706 | 9.0 | -1.56 | -0.0076 | -0.0706 | 0.0 | nan |
| liquidity_cost_shock | forecast_plus_risk_and_regime | -1.93 | -0.2055 | -0.0198 | 0.5098 | 11.0 | -1.97 | -0.0049 | -0.0749 | 0.0 | 0.0 |
| liquidity_cost_shock | forecast_plus_risk_features | -3.99 | -0.2305 | -0.0205 | 0.5882 | 13.0 | -1.87 | -0.0045 | -0.0805 | 0.0 | 0.0 |
| liquidity_cost_shock | full_system | -3.05 | -0.1727 | -0.0171 | 0.3554 | 11.0 | -2.29 | -0.0037 | -0.0841 | 0.0 | 0.0 |
| liquidity_cost_shock | legacy_forecast_only | -2.03 | -0.2041 | -0.0173 | 0.4706 | 9.0 | -1.68 | -0.0022 | -0.0593 | 0.0 | nan |
| regime_persistence_shock | forecast_plus_risk_and_regime | -0.51 | -0.1395 | -0.0156 | 0.5098 | 11.0 | -0.55 | -0.0007 | -0.0089 | 0.0 | 0.0 |
| regime_persistence_shock | forecast_plus_risk_features | -3.09 | -0.1739 | -0.0169 | 0.5882 | 13.0 | -0.97 | -0.0009 | -0.0239 | 0.0 | 0.0 |
| regime_persistence_shock | full_system | 0.29 | -0.0411 | -0.0064 | 0.2225 | 6.8 | 1.05 | 0.007 | 0.0475 | -0.1329 | 0.0 |
| regime_persistence_shock | legacy_forecast_only | -0.98 | -0.1553 | -0.0155 | 0.4706 | 9.0 | -0.63 | -0.0004 | -0.0105 | 0.0 | 0.0 |
| volatility_shock | forecast_plus_risk_and_regime | 1.01 | -0.1588 | -0.0189 | 0.5098 | 11.0 | 0.97 | -0.004 | -0.0282 | 0.0 | 0.0 |
| volatility_shock | forecast_plus_risk_features | -1.58 | -0.1804 | -0.0238 | 0.5882 | 13.0 | 0.54 | -0.0078 | -0.0304 | 0.0 | 0.0 |
| volatility_shock | full_system | 1.12 | -0.0498 | -0.0066 | 0.2029 | 5.2 | 1.88 | 0.0068 | 0.0388 | -0.1525 | 0.0 |
| volatility_shock | legacy_forecast_only | 0.7 | -0.1754 | -0.0182 | 0.4706 | 9.0 | 1.05 | -0.0031 | -0.0306 | 0.0 | 0.0 |

## Output Files
- Detail CSV: `K:\Repos\VSEF-PTIT-NEU\tmp\pytest_tmpdirs\case_0d3253944af842f787d39d0989cf4574\reports\stress_alignment.csv`
- Summary CSV: `K:\Repos\VSEF-PTIT-NEU\tmp\pytest_tmpdirs\case_0d3253944af842f787d39d0989cf4574\reports\stress_alignment_summary.csv`
- JSON: `K:\Repos\VSEF-PTIT-NEU\tmp\pytest_tmpdirs\case_0d3253944af842f787d39d0989cf4574\reports\stress_alignment.json`