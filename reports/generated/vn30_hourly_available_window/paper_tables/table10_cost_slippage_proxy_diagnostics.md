# Table 10: Cost/slippage proxy diagnostics

| slice_name | candidate_source | baseline | transaction_cost_bps | slippage_bps | row_count | gross_return | net_return | turnover | max_drawdown | profit_factor | win_rate | trade_count | average_trade_return | exposure | benchmark_comparison | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| confidence_xgboost_h1_t0.79 | confidence_sweep | model_signal | 5 | 5 | 924 | 15.19% | 10.35% | 21.43% | -0.0377089 | 1.63948 | 60.97% | 198 | 0.22% | 29.11% | 5.66% | proxy_diagnostic |
| confidence_xgboost_h1_t0.79 | confidence_sweep | model_signal | 5 | 10 | 924 | 15.19% | 8.00% | 21.43% | -0.0406678 | 1.57039 | 60.59% | 198 | 0.20% | 29.11% | 3.46% | proxy_diagnostic |
| confidence_xgboost_h1_t0.79 | confidence_sweep | model_signal | 10 | 5 | 924 | 15.19% | 8.00% | 21.43% | -0.0406678 | 1.57039 | 60.59% | 198 | 0.20% | 29.11% | 3.46% | proxy_diagnostic |
| confidence_random_forest_h1_t0.62 | confidence_sweep | model_signal | 5 | 5 | 1526 | 10.08% | 6.61% | 16.45% | -0.0183887 | 1.54739 | 59.67% | 251 | 0.19% | 19.99% | 8.18% | proxy_diagnostic |
| confidence_xgboost_h1_t0.79 | confidence_sweep | model_signal | 15 | 5 | 924 | 15.19% | 5.70% | 21.43% | -0.0436185 | 1.5032 | 58.36% | 198 | 0.18% | 29.11% | 1.32% | proxy_diagnostic |
| confidence_xgboost_h1_t0.79 | confidence_sweep | model_signal | 5 | 15 | 924 | 15.19% | 5.70% | 21.43% | -0.0436185 | 1.5032 | 58.36% | 198 | 0.18% | 29.11% | 1.32% | proxy_diagnostic |
| confidence_xgboost_h1_t0.79 | confidence_sweep | model_signal | 10 | 10 | 924 | 15.19% | 5.70% | 21.43% | -0.0436185 | 1.5032 | 58.36% | 198 | 0.18% | 29.11% | 1.32% | proxy_diagnostic |
| confidence_random_forest_h1_t0.62 | confidence_sweep | model_signal | 10 | 5 | 1526 | 10.08% | 4.91% | 16.45% | -0.0203004 | 1.47707 | 58.69% | 251 | 0.17% | 19.99% | 6.56% | proxy_diagnostic |
| confidence_random_forest_h1_t0.62 | confidence_sweep | model_signal | 5 | 10 | 1526 | 10.08% | 4.91% | 16.45% | -0.0203004 | 1.47707 | 58.69% | 251 | 0.17% | 19.99% | 6.56% | proxy_diagnostic |
| confidence_xgboost_h1_t0.79 | confidence_sweep | buy_and_hold | 5 | 5 | 924 | 4.99% | 4.69% | 2.92% | -0.0831069 | 97.55% | 42.42% | 27 | -0.000100346 | 100.00% | 0.00% | proxy_diagnostic |
| confidence_xgboost_h1_t0.79 | confidence_sweep | always_up | 5 | 5 | 924 | 4.99% | 4.69% | 2.92% | -0.0831069 | 97.55% | 42.42% | 27 | -0.000100346 | 100.00% | 0.00% | proxy_diagnostic |
| confidence_xgboost_h1_t0.79 | confidence_sweep | buy_and_hold | 5 | 10 | 924 | 4.99% | 4.54% | 2.92% | -0.0839825 | 97.20% | 42.32% | 27 | -0.000114957 | 100.00% | 0.00% | proxy_diagnostic |
| confidence_xgboost_h1_t0.79 | confidence_sweep | always_up | 5 | 10 | 924 | 4.99% | 4.54% | 2.92% | -0.0839825 | 97.20% | 42.32% | 27 | -0.000114957 | 100.00% | 0.00% | proxy_diagnostic |
| confidence_xgboost_h1_t0.79 | confidence_sweep | buy_and_hold | 10 | 5 | 924 | 4.99% | 4.54% | 2.92% | -0.0839825 | 97.20% | 42.32% | 27 | -0.000114957 | 100.00% | 0.00% | proxy_diagnostic |
| confidence_xgboost_h1_t0.79 | confidence_sweep | always_up | 10 | 5 | 924 | 4.99% | 4.54% | 2.92% | -0.0839825 | 97.20% | 42.32% | 27 | -0.000114957 | 100.00% | 0.00% | proxy_diagnostic |
| confidence_xgboost_h1_t0.79 | confidence_sweep | buy_and_hold | 5 | 15 | 924 | 4.99% | 4.38% | 2.92% | -0.0848573 | 96.86% | 42.32% | 27 | -0.000129567 | 100.00% | 0.00% | proxy_diagnostic |
| confidence_xgboost_h1_t0.79 | confidence_sweep | buy_and_hold | 15 | 5 | 924 | 4.99% | 4.38% | 2.92% | -0.0848573 | 96.86% | 42.32% | 27 | -0.000129567 | 100.00% | 0.00% | proxy_diagnostic |
| confidence_xgboost_h1_t0.79 | confidence_sweep | always_up | 5 | 15 | 924 | 4.99% | 4.38% | 2.92% | -0.0848573 | 96.86% | 42.32% | 27 | -0.000129567 | 100.00% | 0.00% | proxy_diagnostic |
| confidence_xgboost_h1_t0.79 | confidence_sweep | always_up | 10 | 10 | 924 | 4.99% | 4.38% | 2.92% | -0.0848573 | 96.86% | 42.32% | 27 | -0.000129567 | 100.00% | 0.00% | proxy_diagnostic |
| confidence_xgboost_h1_t0.79 | confidence_sweep | buy_and_hold | 10 | 10 | 924 | 4.99% | 4.38% | 2.92% | -0.0848573 | 96.86% | 42.32% | 27 | -0.000129567 | 100.00% | 0.00% | proxy_diagnostic |

## Note

Proxy cost/slippage diagnostics only; not live-trading evidence.
