# Stress Test Report

## Scenario Assumptions
- `volatility_shock`: Amplify realized volatility and risk metrics after the midpoint.
- `drawdown_shock`: Inject a concentrated negative block loss and deeper drawdown.
- `liquidity_cost_shock`: Increase fees and slippage to simulate thinner liquidity.
- `regime_persistence_shock`: Force crisis-like regime persistence after the first shock.

## Summary
| stress_scenario | benchmark_mode | stressed_sharpe | stressed_max_drawdown | stressed_tail_loss | stressed_exposure | stressed_turnover | delta_sharpe | delta_tail_loss | delta_drawdown | delta_exposure | regime_reaction_speed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drawdown_shock | forecast_plus_risk_and_regime | -1.44 | -0.1833 | -0.0212 | 0.4854 | 11.0 | -1.31 | -0.0063 | -0.0527 | 0.0 | 0.0 |
| drawdown_shock | forecast_plus_risk_features | -3.87 | -0.2172 | -0.0212 | 0.5728 | 11.0 | -1.21 | -0.0054 | -0.0695 | 0.0 | 0.0 |
| drawdown_shock | full_system | 0.16 | -0.0553 | -0.0099 | 0.2816 | 6.5 | 1.15 | 0.0035 | 0.0333 | -0.0509 | 0.0 |
| drawdown_shock | legacy_forecast_only | -2.1 | -0.2102 | -0.0212 | 0.4466 | 9.0 | -1.57 | -0.0061 | -0.0654 | 0.0 | nan |
| liquidity_cost_shock | forecast_plus_risk_and_regime | -2.07 | -0.2055 | -0.0198 | 0.4854 | 11.0 | -1.94 | -0.0049 | -0.0749 | 0.0 | 0.0 |
| liquidity_cost_shock | forecast_plus_risk_features | -4.27 | -0.2281 | -0.0191 | 0.5728 | 11.0 | -1.61 | -0.0033 | -0.0804 | 0.0 | 0.0 |
| liquidity_cost_shock | full_system | -3.24 | -0.1727 | -0.0171 | 0.3325 | 11.0 | -2.25 | -0.0037 | -0.0841 | 0.0 | 0.0 |
| liquidity_cost_shock | legacy_forecast_only | -2.19 | -0.2041 | -0.0173 | 0.4466 | 9.0 | -1.66 | -0.0022 | -0.0593 | 0.0 | nan |
| regime_persistence_shock | forecast_plus_risk_and_regime | -0.59 | -0.136 | -0.0152 | 0.4854 | 11.0 | -0.46 | -0.0003 | -0.0054 | 0.0 | 0.0 |
| regime_persistence_shock | forecast_plus_risk_features | -3.68 | -0.1881 | -0.0168 | 0.5728 | 11.0 | -1.02 | -0.001 | -0.0404 | 0.0 | 0.0 |
| regime_persistence_shock | full_system | 0.1 | -0.0401 | -0.0064 | 0.2087 | 6.8 | 1.09 | 0.007 | 0.0485 | -0.1238 | 0.0 |
| regime_persistence_shock | legacy_forecast_only | -1.07 | -0.1519 | -0.0155 | 0.4466 | 9.0 | -0.54 | -0.0004 | -0.0071 | 0.0 | 0.0 |
| volatility_shock | forecast_plus_risk_and_regime | 1.08 | -0.1493 | -0.0178 | 0.4854 | 11.0 | 1.21 | -0.0029 | -0.0187 | 0.0 | 0.0 |
| volatility_shock | forecast_plus_risk_features | -2.02 | -0.1799 | -0.025 | 0.5728 | 11.0 | 0.64 | -0.0092 | -0.0322 | 0.0 | 0.0 |
| volatility_shock | full_system | 1.05 | -0.0471 | -0.0066 | 0.1893 | 5.2 | 2.04 | 0.0068 | 0.0415 | -0.1432 | 0.0 |
| volatility_shock | legacy_forecast_only | 0.77 | -0.166 | -0.0175 | 0.4466 | 9.0 | 1.3 | -0.0024 | -0.0212 | 0.0 | 0.0 |

## Output Files
- Detail CSV: `K:\Repos\VSEF-PTIT-NEU\tmp\pytest_tmpdirs\case_3c09f1f50f1b4248919869e635339823\reports\stress_alignment.csv`
- Summary CSV: `K:\Repos\VSEF-PTIT-NEU\tmp\pytest_tmpdirs\case_3c09f1f50f1b4248919869e635339823\reports\stress_alignment_summary.csv`
- JSON: `K:\Repos\VSEF-PTIT-NEU\tmp\pytest_tmpdirs\case_3c09f1f50f1b4248919869e635339823\reports\stress_alignment.json`