# Chapter 4: Empirical Results Draft

## 4.1 Official Global Benchmark Results

The official 2025 VN100 walk-forward benchmark produced nonzero prediction artifacts for both daily and hourly frequencies. The daily benchmark records 26,104 predictions with overall directional accuracy of 0.5318725099601593. The hourly benchmark records 127,944 predictions with overall directional accuracy of 0.5128571875195398.

Neither frequency passed the global 60% benchmark threshold. This is the central empirical boundary for the paper: the official run does not support a global benchmark pass. Table 3 at `reports/generated/paper_tables/table3_global_benchmark_results.md` provides the paper-ready summary.

## 4.2 Baseline Comparison

The official baseline delta artifacts compare model accuracy against always-up, previous-direction, seeded random direction, and moving-average signal baselines. Some model/horizon rows outperform simple baselines. For example, the daily best model accuracy is 0.5697236180904522, while the best daily baseline accuracy recorded in the benchmark summary is 0.5452261306532663. The hourly best model accuracy is 0.5559340509606213, while the best hourly baseline accuracy is 0.5054466230936819.

These deltas are useful diagnostic evidence, but they are not a global benchmark pass and they are not trading-profitability evidence. Table 4 at `reports/generated/paper_tables/table4_baseline_delta_summary.md` lists the paper-ready baseline delta rows.

## 4.3 Confidence-Filtered Strategy Diagnostics

The official confidence-threshold sweep identifies one selected strategy-level diagnostic slice: hourly stacking at horizon 1 with threshold 0.57. This slice has 2,297 evaluated rows, coverage ratio 0.3129854203569969, and filtered accuracy 0.6003482803656944. Therefore, the paper may state that a strategy-level diagnostic pass exists at 60.03% accuracy and 31.30% coverage.

The claim must remain narrow. Under coverage floors of 50% and 40%, no available confidence-filtered row reaches 60% accuracy. The daily threshold-sweep file contains no data rows, and the available combined sweep rows cover hourly stacking h=1 only. Table 5 at `reports/generated/paper_tables/table5_confidence_filtered_diagnostics.md` and Figure 4 at `reports/generated/paper_figures/figure4_confidence_threshold_coverage_accuracy.png` summarize this evidence.

## 4.4 Regime-Specific Diagnostics

The strongest regime-specific daily diagnostic appears in the bear regime at horizon 20. Daily LightGBM h=20 reaches 0.6959459459459459 accuracy over 444 observations, and daily XGBoost h=20 reaches 0.6914414414414415 over the same observation count. These are regime-specific 63%+ diagnostics, not global benchmark results.

The paper should not describe these rows as a stable full-market 63% method. The finding is conditional on the regime slice, the seven evaluated tickers, and the official 2025 window. Table 6 at `reports/generated/paper_tables/table6_regime_specific_diagnostics.md` and Figure 5 at `reports/generated/paper_figures/figure5_regime_specific_accuracy.png` provide the paper-ready regime summary.

## 4.5 Statistical Significance

The official significance summaries include binomial p-values against a 50% null and bootstrap confidence intervals. Several model/horizon rows are statistically above the 50% null. For instance, hourly stacking h=1 records accuracy 0.5559340509606213 with a binomial p-value of 4.771490994537493e-22.

Statistical significance should be interpreted as directional-accuracy evidence only. It does not prove practical trading readiness, cost-adjusted profitability, or multi-window robustness. Table 7 at `reports/generated/paper_tables/table7_statistical_significance_summary.md` lists the statistical summary rows.

## 4.6 Ticker Concentration and Representativeness

Ticker concentration diagnostics show that the global daily and global hourly prediction rows are not dominated by a single ticker by prediction count. However, the selected hourly confidence slice is concentrated: it contains five tickers, and the top three tickers contribute about 79.49% of selected rows. In that selected slice, BID contributes 31.17% of rows, CII contributes 28.73%, and BCM contributes 19.59%.

This concentration affects the interpretation of the selected strategy-level pass. The result remains useful as a conditional diagnostic, but it should not be generalized as full-market VN100 evidence. The concentration report is available at `reports/generated/vn100_ticker_concentration_summary.md`.

## 4.7 Cost/Slippage Readiness Gap

The official selected VN100 slices have not been evaluated with transaction costs, slippage, turnover, drawdown, profit factor, trade lists, equity curves, or cost-adjusted returns. The repository contains backtest modules that can model fees, slippage, turnover, drawdown, and profit factor, but those modules have not produced official artifacts for the selected VN100 confidence or regime slices.

Therefore, practical trading readiness is not established. The paper must not claim profitability, deployment readiness, or executable investment suitability. The cost/slippage readiness review is available at `reports/generated/vn100_cost_slippage_readiness_review.md`.

## 4.8 Discussion of Empirical Limitations

The empirical results support a careful and bounded conclusion. Global benchmark pass: no. Strategy-level diagnostic pass: yes, hourly stacking h=1 at threshold 0.57 with 60.03% accuracy and 31.30% coverage. Stable full-market 63% method: no. Regime-specific 63%+ diagnostic: yes, but bear-regime only and not a global pass. Practical trading readiness: not established.

The strongest interpretation is that the official benchmark provides leakage-aware evidence of conditional predictive signal under limited cache coverage. The weakest interpretation, which should be avoided, is that the current artifacts prove a deployable trading system or a stable full-market VN100 method.
