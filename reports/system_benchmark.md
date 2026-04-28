# System Benchmark

## Summary
| benchmark_mode | cumulative_return | cagr | volatility | sharpe | sortino | calmar | max_drawdown | avg_drawdown | tail_loss | turnover | exposure | trade_count | rmse | mae | directional_accuracy | test_rows | delta_sharpe_vs_legacy | delta_cagr_vs_legacy | delta_mdd_vs_legacy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| forecast_plus_risk_and_regime | -0.0004 | -0.001 | 0.1072 | 0.04 | 0.06 | -0.0075 | -0.1306 | -0.0492 | -0.0149 | 11.0 | 0.5098 | 6.0 | 0.050195 | 0.039752 | 0.539216 | 102.0 | 0.39 | 0.0399 | 0.0142 |
| forecast_plus_risk_features | -0.097 | -0.2228 | 0.1169 | -2.12 | -3.24 | -1.4861 | -0.15 | -0.0745 | -0.016 | 13.0 | 0.5882 | 7.0 | 0.050195 | 0.039752 | 0.441176 | 102.0 | -1.77 | -0.1819 | -0.0052 |
| full_system | -0.0254 | -0.0617 | 0.08 | -0.76 | -0.79 | -0.6958 | -0.0886 | -0.0529 | -0.0134 | 11.0 | 0.3554 | 6.0 | 0.050195 | 0.039752 | 0.539216 | 102.0 | -0.41 | -0.0208 | 0.0562 |
| legacy_forecast_only | -0.0168 | -0.0409 | 0.1046 | -0.35 | -0.48 | -0.2825 | -0.1448 | -0.0601 | -0.0151 | 9.0 | 0.4706 | 5.0 | 0.050195 | 0.039752 | 0.519608 | 102.0 | 0.0 | 0.0 | 0.0 |

## Best Rows By Mode
| benchmark_mode | ticker | horizon | algorithm | mode_description | train_rows | val_rows | test_rows | cumulative_return | cagr | volatility | sharpe | sortino | calmar | max_drawdown | avg_drawdown | tail_loss | turnover | exposure | trade_count | rmse | mae | directional_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| forecast_plus_risk_and_regime | ALIGN | short | cart | Forecasting with risk features and regime-aware inputs. | 470 | 97 | 102 | -0.0004 | -0.001 | 0.1072 | 0.04 | 0.06 | -0.0075 | -0.1306 | -0.0492 | -0.0149 | 11.0 | 0.5098 | 6 | 0.050195 | 0.039752 | 0.539216 |
| legacy_forecast_only | ALIGN | short | cart | Forecasting only, no risk/regime/allocation extensions. | 470 | 97 | 102 | -0.0168 | -0.0409 | 0.1046 | -0.35 | -0.48 | -0.2825 | -0.1448 | -0.0601 | -0.0151 | 9.0 | 0.4706 | 5 | 0.050195 | 0.039752 | 0.519608 |
| full_system | ALIGN | short | cart | Forecasting with risk, regime, and allocation overlays. | 470 | 97 | 102 | -0.0254 | -0.0617 | 0.08 | -0.76 | -0.79 | -0.6958 | -0.0886 | -0.0529 | -0.0134 | 11.0 | 0.3554 | 6 | 0.050195 | 0.039752 | 0.539216 |
| forecast_plus_risk_features | ALIGN | short | cart | Forecasting with rolling VaR/CVaR/CoVaR/Drawdown features. | 470 | 97 | 102 | -0.097 | -0.2228 | 0.1169 | -2.12 | -3.24 | -1.4861 | -0.15 | -0.0745 | -0.016 | 13.0 | 0.5882 | 7 | 0.050195 | 0.039752 | 0.441176 |

## Output Files
- Detail CSV: `K:\Repos\VSEF-PTIT-NEU\tmp\pytest_tmpdirs\case_7131b2eba85f421f8a9ff36cad5b2571\reports\alignment.csv`
- Summary CSV: `K:\Repos\VSEF-PTIT-NEU\tmp\pytest_tmpdirs\case_7131b2eba85f421f8a9ff36cad5b2571\reports\alignment_summary.csv`
- JSON: `K:\Repos\VSEF-PTIT-NEU\tmp\pytest_tmpdirs\case_7131b2eba85f421f8a9ff36cad5b2571\reports\alignment.json`