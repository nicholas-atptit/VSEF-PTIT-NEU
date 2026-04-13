# System Benchmark

## Summary
| benchmark_mode | cumulative_return | cagr | volatility | sharpe | sortino | calmar | max_drawdown | avg_drawdown | tail_loss | turnover | exposure | trade_count | rmse | mae | directional_accuracy | test_rows | delta_sharpe_vs_legacy | delta_cagr_vs_legacy | delta_mdd_vs_legacy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| forecast_plus_risk_and_regime | -0.2538 | -0.4431 | 0.0987 | -5.92 | -6.12 | -1.7196 | -0.2577 | -0.0975 | -0.0182 | 15.0 | 0.4127 | 7.0 | 0.051754 | 0.042012 | 0.396825 | 126.0 | 0.0 | 0.0 | 0.0 |
| forecast_plus_risk_features | -0.2538 | -0.4431 | 0.0987 | -5.92 | -6.12 | -1.7196 | -0.2577 | -0.0975 | -0.0182 | 15.0 | 0.4127 | 7.0 | 0.051754 | 0.042012 | 0.396825 | 126.0 | 0.0 | 0.0 | 0.0 |
| full_system | -0.1799 | -0.3275 | 0.0723 | -5.49 | -4.6 | -1.7773 | -0.1842 | -0.0903 | -0.0161 | 12.75 | 0.2163 | 7.0 | 0.051754 | 0.042012 | 0.396825 | 126.0 | 0.43 | 0.1156 | 0.0735 |
| legacy_forecast_only | -0.2538 | -0.4431 | 0.0987 | -5.92 | -6.12 | -1.7196 | -0.2577 | -0.0975 | -0.0182 | 15.0 | 0.4127 | 7.0 | 0.051754 | 0.042012 | 0.396825 | 126.0 | 0.0 | 0.0 | 0.0 |

## Best Rows By Mode
| benchmark_mode | ticker | horizon | algorithm | mode_description | train_rows | val_rows | test_rows | cumulative_return | cagr | volatility | sharpe | sortino | calmar | max_drawdown | avg_drawdown | tail_loss | turnover | exposure | trade_count | rmse | mae | directional_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_system | ALIGN | short | cart | Forecasting with risk, regime, and allocation overlays. | 578 | 120 | 126 | -0.1799 | -0.3275 | 0.0723 | -5.49 | -4.6 | -1.7773 | -0.1842 | -0.0903 | -0.0161 | 12.75 | 0.2163 | 7 | 0.051754 | 0.042012 | 0.396825 |
| forecast_plus_risk_and_regime | ALIGN | short | cart | Forecasting with risk features and regime-aware inputs. | 578 | 120 | 126 | -0.2538 | -0.4431 | 0.0987 | -5.92 | -6.12 | -1.7196 | -0.2577 | -0.0975 | -0.0182 | 15.0 | 0.4127 | 7 | 0.051754 | 0.042012 | 0.396825 |
| forecast_plus_risk_features | ALIGN | short | cart | Forecasting with rolling VaR/CVaR/CoVaR/Drawdown features. | 578 | 120 | 126 | -0.2538 | -0.4431 | 0.0987 | -5.92 | -6.12 | -1.7196 | -0.2577 | -0.0975 | -0.0182 | 15.0 | 0.4127 | 7 | 0.051754 | 0.042012 | 0.396825 |
| legacy_forecast_only | ALIGN | short | cart | Forecasting only, no risk/regime/allocation extensions. | 578 | 120 | 126 | -0.2538 | -0.4431 | 0.0987 | -5.92 | -6.12 | -1.7196 | -0.2577 | -0.0975 | -0.0182 | 15.0 | 0.4127 | 7 | 0.051754 | 0.042012 | 0.396825 |

## Output Files
- Detail CSV: `H:\AI-ML-LLM in Stock_march26_PTIT_NEU\tmp\pytest_tmpdirs\case_0a56428c730440eebf6277c7c5911961\reports\alignment.csv`
- Summary CSV: `H:\AI-ML-LLM in Stock_march26_PTIT_NEU\tmp\pytest_tmpdirs\case_0a56428c730440eebf6277c7c5911961\reports\alignment_summary.csv`
- JSON: `H:\AI-ML-LLM in Stock_march26_PTIT_NEU\tmp\pytest_tmpdirs\case_0a56428c730440eebf6277c7c5911961\reports\alignment.json`