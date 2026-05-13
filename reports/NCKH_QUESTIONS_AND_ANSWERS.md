# NCKH Committee Questions and Safe Answers

## 1. Did the official benchmark pass the global 60% threshold?

No. The official daily and hourly benchmark summaries both report no global 60% pass. Daily accuracy is 53.19%, and hourly accuracy is 51.29%.

## 2. If the global benchmark failed, what is the main result?

The main result is conditional signal, not global success. The selected hourly stacking h=1 confidence-filtered slice reaches 60.03% accuracy at 31.30% coverage, and daily bear-regime h=20 diagnostics exceed 63%.

## 3. Why study VN100 if only seven tickers are evaluated?

The target research universe is VN100, but current benchmark-usable cache coverage limits the official evaluated set to seven tickers. The paper states this limitation clearly and does not claim full VN100 representativeness.

## 4. Which seven tickers are evaluated?

ANV, BCM, BID, BMP, BVH, BWE, and CII.

## 5. Does seven-ticker coverage invalidate the study?

No. It limits representativeness, but the study remains useful as a leakage-aware benchmark framework and conditional diagnostic analysis. It should not be presented as a definitive full-market VN100 conclusion.

## 6. What is the confidence threshold 0.57?

It is the selected threshold in the available official confidence-threshold sweep for hourly stacking h=1. At this threshold, filtered accuracy is 60.03% and coverage is 31.30%.

## 7. Why does 31.30% coverage matter?

Coverage shows how many prediction rows remain after filtering. A 60.03% accuracy at 31.30% coverage is narrower than a global result. It supports a strategy-level diagnostic claim only.

## 8. What happens at higher coverage floors?

Under the available 50% and 40% coverage floors, no confidence-filtered row reaches 60% accuracy. This is why the selected confidence result must be framed carefully.

## 9. Does the confidence-filtered result prove the model is good?

It shows conditional signal in one selected slice. It does not prove broad model superiority, full-market representativeness, or trading readiness.

## 10. What does the bear-regime result mean?

Daily bear-regime h=20 diagnostics show high directional accuracy in that regime. LightGBM reaches 69.59%, and XGBoost reaches 69.14% over 444 observations.

## 11. Is the 63%+ regime result stable?

Not established. It is a regime-specific diagnostic from the official 2025 artifact window. It requires ex-ante regime validation and multi-window testing before it can be called stable.

## 12. Is this a tradable strategy?

No. The official selected slices do not include cost-adjusted returns, slippage, turnover, drawdown, profit factor, trade lists, or equity curves.

## 13. Why is directional accuracy not enough for trading?

Directional accuracy ignores transaction costs, slippage, position sizing, liquidity, turnover, drawdown, and return magnitude. A directional signal can be statistically interesting but economically weak after costs.

## 14. How is leakage controlled?

The official run records `train_cutoff = 2024-12-31` and `training_label_cutoff_rule = target_timestamp <= train_cutoff`. The 2025 outcomes are used for held-out evaluation, not for training labels.

## 15. Are 2025 actual rows used?

Yes, but as evaluation labels. The key safeguard is that training labels are capped at the 2024-12-31 cutoff.

## 16. Which models are evaluated?

LightGBM, XGBoost, random forest, and stacking. No new model family is added in the paper package.

## 17. Why include simple baselines?

Baselines help test whether model results exceed simple directional rules such as always-up, previous-direction, random seeded direction, and moving-average signal.

## 18. Did models beat baselines?

Some model/horizon rows outperform simple baselines. This is diagnostic evidence, but it is not the same as a global 60% benchmark pass.

## 19. What statistical tests are used?

The official artifacts include binomial p-values against a 50% null and bootstrap confidence intervals. These support forecast-evaluation interpretation, not profitability claims.

## 20. Does statistical significance prove trading value?

No. Statistical significance can show directional evidence, but trading value requires cost/slippage and portfolio-level evidence.

## 21. What is missing from the cost/slippage side?

The official selected slices lack transaction-cost, slippage, turnover, drawdown, profit-factor, trade-list, equity-curve, and cost-adjusted return artifacts.

## 22. What are the most important future experiments?

Expand cache coverage, rerun the official benchmark across additional windows, broaden confidence sweeps, validate ex-ante regime rules, and run cost/slippage-aware backtests.

## 23. Why not add new model families now?

The evidence base should be hardened first. Adding new model families before coverage, leakage, confidence, regime, and trading-readiness checks are stable could increase complexity without improving claim quality.

## 24. What is the safest final conclusion?

The official 2025 VN100 walk-forward benchmark did not pass the global 60% threshold, but selected confidence-filtered and bear-regime diagnostics show conditional predictive signal that merits further validation.
