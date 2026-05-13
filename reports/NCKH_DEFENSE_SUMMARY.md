# NCKH Defense Summary

## 1-Minute Summary

This study evaluates VN100 stock-direction forecasting with a leakage-aware walk-forward benchmark. The official artifact family enforces a 2024-12-31 training-label cutoff and evaluates 2025 outcomes out of sample. The tested models are LightGBM, XGBoost, random forest, and stacking, compared with simple directional baselines.

The global benchmark does not pass 60%: daily accuracy is 53.19% over 26,104 predictions, and hourly accuracy is 51.29% over 127,944 predictions. However, conditional diagnostics show signal: hourly stacking h=1 at threshold 0.57 reaches 60.03% accuracy with 31.30% coverage, and daily bear-regime h=20 diagnostics exceed 63%. These are diagnostic findings only. The current evidence covers seven tickers and does not establish full-market VN100 representativeness or trading readiness.

## 3-Minute Summary

The research problem is that stock forecasting papers can overstate model quality if they use weak validation or treat selected slices as broad market proof. This study focuses on methodology and evidence governance. It uses official VN100 benchmark artifacts with a clear training-label cutoff at 2024-12-31 and a held-out 2025 evaluation window.

The model set includes LightGBM, XGBoost, random forest, and stacking. The benchmark also compares simple directional baselines such as always-up, previous-direction, seeded random direction, and moving-average signal. Evaluation uses directional accuracy, baseline deltas, confidence filtering, regime-specific accuracy, binomial p-values, bootstrap confidence intervals, and concentration diagnostics.

The main result is negative at the global benchmark level: neither daily nor hourly results pass the 60% threshold. The daily result is 53.19%, and the hourly result is 51.29%. The conditional results are more interesting. Hourly stacking h=1 with confidence threshold 0.57 reaches 60.03% filtered accuracy, but only at 31.30% coverage. Daily bear-regime h=20 diagnostics exceed 63%, including LightGBM at 69.59% and XGBoost at 69.14% over 444 observations.

The key conclusion is cautious: conditional predictive signal exists, but the current evidence is not enough for a full-market VN100 claim or trading-readiness claim. Future work must expand cache coverage, validate multiple windows, broaden confidence sweeps, define ex-ante regime rules, and add transaction-cost and slippage backtests.

## 5-Minute Summary

This project studies whether machine learning and ensemble models can forecast VN100 stock direction under a strict walk-forward design. The motivation is methodological: financial forecasting results can be inflated by leakage, non-chronological splits, weak baselines, selective reporting, or unsupported trading claims. The project therefore treats artifact quality and claim boundaries as part of the research contribution.

The evidence comes from the official artifact directory `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff`. The run records `train_cutoff = 2024-12-31` and `training_label_cutoff_rule = target_timestamp <= train_cutoff`. The evaluation window is 2025. This supports a leakage-aware out-of-sample interpretation, while still allowing 2025 actual rows to be used as evaluation labels.

The study evaluates LightGBM, XGBoost, random forest, and stacking. It compares them with simple baselines and reports global accuracy, baseline deltas, confidence-filtered accuracy, coverage ratio, regime-specific accuracy, p-values, and bootstrap intervals. It also adds concentration and readiness reviews to prevent overclaiming.

The official global result is not a benchmark pass. Daily accuracy is 53.19% over 26,104 predictions, and hourly accuracy is 51.29% over 127,944 predictions. Both are below 60%. Some model/horizon rows beat simple baselines, and several rows are statistically above a 50% null, but those points do not convert into a global pass.

The selected confidence result is hourly stacking h=1 at threshold 0.57. It reaches 60.03% filtered accuracy with 31.30% coverage over 2,297 evaluated rows. This is the main strategy-level diagnostic pass. It should not be described as a global benchmark pass. The evidence is also concentrated: the selected slice contains five tickers, with the top three contributing about 79.49% of selected rows.

The regime results show strong daily bear-regime h=20 diagnostics. LightGBM reaches 69.59%, and XGBoost reaches 69.14% over 444 observations. This is meaningful as a conditional diagnostic, but it is not a stable full-market 63% method. It still requires ex-ante regime validation and multi-window testing.

Finally, trading readiness is not established. The selected official VN100 slices do not include cost-adjusted returns, slippage-applied execution, turnover, drawdown, profit factor, trade lists, or equity curves. Therefore, the project contributes a careful benchmark and evidence framework, not a deployable trading strategy.

## Likely Committee Questions and Safe Answers

| Question | Safe answer |
|---|---|
| Did the model pass the official 60% benchmark? | No. The official daily and hourly benchmark summaries both report no global 60% pass. |
| Why is the selected confidence result still important? | It shows conditional signal: hourly stacking h=1 at threshold 0.57 reaches 60.03% accuracy, but only at 31.30% coverage. It is a strategy-level diagnostic, not a global pass. |
| Can we claim the model reaches 63% accuracy? | Only as a regime-specific diagnostic. Daily bear-regime h=20 rows exceed 63%, but this is not a stable full-market method. |
| Is the evidence representative of the full VN100? | No. The official evaluated tickers are ANV, BCM, BID, BMP, BVH, BWE, and CII. Cache coverage limits representativeness. |
| Why use walk-forward validation? | It respects time order and supports out-of-sample evaluation. The official run uses a 2024-12-31 training-label cutoff and 2025 held-out evaluation. |
| Is this ready for trading? | No. The official selected slices do not yet include cost-adjusted returns, slippage, turnover, drawdown, profit factor, or executable trade artifacts. |
| Did this phase add new model families? | No. It documents the existing official evidence and does not change runtime or model logic. |
| What is the most important future work? | Expand cache coverage, validate additional windows, broaden confidence sweeps, validate ex-ante regime rules, and run cost/slippage-aware backtests. |

## Claim Boundary Reminders

- Safe: The official 2025 VN100 benchmark did not pass the global 60% threshold.
- Safe: Conditional confidence-filtered and bear-regime diagnostics show signal that merits further validation.
- Unsafe: The model is ready for live trading.
- Unsafe: The benchmark proves a stable full-market 63% method.
- Unsafe: The current seven-ticker evidence represents the full VN100 universe.
- Unsafe: Statistical significance proves profitability.
