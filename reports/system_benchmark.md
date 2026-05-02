# System Benchmark

## Summary
| benchmark_mode | cumulative_return | cagr | volatility | sharpe | sortino | calmar | max_drawdown | avg_drawdown | tail_loss | turnover | exposure | trade_count | rmse | mae | directional_accuracy | test_rows | delta_sharpe_vs_legacy | delta_cagr_vs_legacy | delta_mdd_vs_legacy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| forecast_plus_risk_and_regime | -0.0024 | -0.0059 | 0.1066 | 0.0 | 0.0 | -0.045 | -0.1306 | -0.0491 | -0.0149 | 11.0 | 0.4951 | 6.0 | 0.050621 | 0.040329 | 0.553398 | 103.0 | 0.4 | 0.0393 | 0.0142 |
| forecast_plus_risk_features | -0.0988 | -0.2248 | 0.1162 | -2.15 | -3.27 | -1.4989 | -0.15 | -0.0758 | -0.016 | 13.0 | 0.5728 | 7.0 | 0.050621 | 0.040329 | 0.456311 | 103.0 | -1.75 | -0.1796 | -0.0052 |
| full_system | -0.0274 | -0.0657 | 0.0795 | -0.82 | -0.85 | -0.7413 | -0.0886 | -0.0532 | -0.0134 | 11.0 | 0.3422 | 6.0 | 0.050621 | 0.040329 | 0.553398 | 103.0 | -0.42 | -0.0205 | 0.0562 |
| legacy_forecast_only | -0.0187 | -0.0452 | 0.104 | -0.4 | -0.54 | -0.3123 | -0.1448 | -0.06 | -0.0151 | 9.0 | 0.4563 | 5.0 | 0.050621 | 0.040329 | 0.533981 | 103.0 | 0.0 | 0.0 | 0.0 |

## Best Rows By Mode
| benchmark_mode | ticker | horizon | algorithm | mode_description | train_rows | val_rows | test_rows | cumulative_return | cagr | volatility | sharpe | sortino | calmar | max_drawdown | avg_drawdown | tail_loss | turnover | exposure | trade_count | rmse | mae | directional_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| forecast_plus_risk_and_regime | ALIGN | short | cart | Forecasting with risk features and regime-aware inputs. | 471 | 97 | 103 | -0.0024 | -0.0059 | 0.1066 | -0.0 | -0.0 | -0.045 | -0.1306 | -0.0491 | -0.0149 | 11.0 | 0.4951 | 6 | 0.050621 | 0.040329 | 0.553398 |
| legacy_forecast_only | ALIGN | short | cart | Forecasting only, no risk/regime/allocation extensions. | 471 | 97 | 103 | -0.0187 | -0.0452 | 0.104 | -0.4 | -0.54 | -0.3123 | -0.1448 | -0.06 | -0.0151 | 9.0 | 0.4563 | 5 | 0.050621 | 0.040329 | 0.533981 |
| full_system | ALIGN | short | cart | Forecasting with risk, regime, and allocation overlays. | 471 | 97 | 103 | -0.0274 | -0.0657 | 0.0795 | -0.82 | -0.85 | -0.7413 | -0.0886 | -0.0532 | -0.0134 | 11.0 | 0.3422 | 6 | 0.050621 | 0.040329 | 0.553398 |
| forecast_plus_risk_features | ALIGN | short | cart | Forecasting with rolling VaR/CVaR/CoVaR/Drawdown features. | 471 | 97 | 103 | -0.0988 | -0.2248 | 0.1162 | -2.15 | -3.27 | -1.4989 | -0.15 | -0.0758 | -0.016 | 13.0 | 0.5728 | 7 | 0.050621 | 0.040329 | 0.456311 |

## Output Files
- Detail CSV: `K:\Repos\VSEF-PTIT-NEU\tmp\pytest_tmpdirs\case_d5b3632a52ad4eab8978d42ac57415f8\reports\alignment.csv`
- Summary CSV: `K:\Repos\VSEF-PTIT-NEU\tmp\pytest_tmpdirs\case_d5b3632a52ad4eab8978d42ac57415f8\reports\alignment_summary.csv`
- JSON: `K:\Repos\VSEF-PTIT-NEU\tmp\pytest_tmpdirs\case_d5b3632a52ad4eab8978d42ac57415f8\reports\alignment.json`