# System Benchmark

## Summary
| benchmark_mode | cumulative_return | cagr | volatility | sharpe | sortino | calmar | max_drawdown | avg_drawdown | tail_loss | turnover | exposure | trade_count | rmse | mae | directional_accuracy | test_rows | delta_sharpe_vs_legacy | delta_cagr_vs_legacy | delta_mdd_vs_legacy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| forecast_plus_risk_and_regime | -0.0077 | -0.0187 | 0.1062 | -0.13 | -0.17 | -0.1433 | -0.1306 | -0.049 | -0.0149 | 11.0 | 0.4854 | 6.0 | 0.050806 | 0.040505 | 0.553398 | 103.0 | 0.4 | 0.0388 | 0.0142 |
| forecast_plus_risk_features | -0.1171 | -0.2626 | 0.1133 | -2.66 | -4.05 | -1.7775 | -0.1477 | -0.0746 | -0.0158 | 11.0 | 0.5728 | 6.0 | 0.050806 | 0.040505 | 0.427184 | 103.0 | -2.13 | -0.2051 | -0.0029 |
| full_system | -0.0325 | -0.0777 | 0.079 | -0.99 | -1.02 | -0.8773 | -0.0886 | -0.0533 | -0.0134 | 11.0 | 0.3325 | 6.0 | 0.050806 | 0.040505 | 0.553398 | 103.0 | -0.46 | -0.0202 | 0.0562 |
| legacy_forecast_only | -0.0239 | -0.0575 | 0.1036 | -0.53 | -0.71 | -0.3974 | -0.1448 | -0.06 | -0.0151 | 9.0 | 0.4466 | 5.0 | 0.050806 | 0.040505 | 0.533981 | 103.0 | 0.0 | 0.0 | 0.0 |

## Best Rows By Mode
| benchmark_mode | ticker | horizon | algorithm | mode_description | train_rows | val_rows | test_rows | cumulative_return | cagr | volatility | sharpe | sortino | calmar | max_drawdown | avg_drawdown | tail_loss | turnover | exposure | trade_count | rmse | mae | directional_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| forecast_plus_risk_and_regime | ALIGN | short | cart | Forecasting with risk features and regime-aware inputs. | 472 | 97 | 103 | -0.0077 | -0.0187 | 0.1062 | -0.13 | -0.17 | -0.1433 | -0.1306 | -0.049 | -0.0149 | 11.0 | 0.4854 | 6 | 0.050806 | 0.040505 | 0.553398 |
| legacy_forecast_only | ALIGN | short | cart | Forecasting only, no risk/regime/allocation extensions. | 472 | 97 | 103 | -0.0239 | -0.0575 | 0.1036 | -0.53 | -0.71 | -0.3974 | -0.1448 | -0.06 | -0.0151 | 9.0 | 0.4466 | 5 | 0.050806 | 0.040505 | 0.533981 |
| full_system | ALIGN | short | cart | Forecasting with risk, regime, and allocation overlays. | 472 | 97 | 103 | -0.0325 | -0.0777 | 0.079 | -0.99 | -1.02 | -0.8773 | -0.0886 | -0.0533 | -0.0134 | 11.0 | 0.3325 | 6 | 0.050806 | 0.040505 | 0.553398 |
| forecast_plus_risk_features | ALIGN | short | cart | Forecasting with rolling VaR/CVaR/CoVaR/Drawdown features. | 472 | 97 | 103 | -0.1171 | -0.2626 | 0.1133 | -2.66 | -4.05 | -1.7775 | -0.1477 | -0.0746 | -0.0158 | 11.0 | 0.5728 | 6 | 0.050806 | 0.040505 | 0.427184 |

## Output Files
- Detail CSV: `K:\Repos\VSEF-PTIT-NEU\tmp\pytest_tmpdirs\case_d814c6696ca045dab992f00e6c58e25a\reports\alignment.csv`
- Summary CSV: `K:\Repos\VSEF-PTIT-NEU\tmp\pytest_tmpdirs\case_d814c6696ca045dab992f00e6c58e25a\reports\alignment_summary.csv`
- JSON: `K:\Repos\VSEF-PTIT-NEU\tmp\pytest_tmpdirs\case_d814c6696ca045dab992f00e6c58e25a\reports\alignment.json`