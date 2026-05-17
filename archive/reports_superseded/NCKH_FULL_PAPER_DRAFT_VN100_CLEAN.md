# Walk-Forward Evaluation of Machine Learning and Ensemble Models for VN100 Stock Direction Forecasting

Vietnamese title: Đánh giá thực nghiệm các mô hình học máy và ensemble trong dự báo xu hướng cổ phiếu VN100 theo phương pháp walk-forward ngoài mẫu.

## Abstract

This paper evaluates machine learning and ensemble models for VN100 stock-direction forecasting under a leakage-aware walk-forward design. The study uses an official VN100 benchmark artifact family that applies a 2024-12-31 training-label cutoff and evaluates 2025 held-out outcomes. The model set consists of LightGBM, XGBoost, random forest, and stacking, compared with simple directional baselines. The official daily benchmark records 26,104 predictions with 53.19% directional accuracy, while the official hourly benchmark records 127,944 predictions with 51.29% directional accuracy. Neither frequency passes the global 60% benchmark threshold. However, conditional diagnostics show selected evidence of signal: hourly stacking at horizon 1 and confidence threshold 0.57 reaches 60.03% filtered accuracy at 31.30% coverage, and daily bear-regime horizon-20 diagnostics exceed 63% accuracy. These findings are constrained by limited benchmark-usable cache coverage, seven evaluated tickers, partial confidence-sweep evidence, selected-slice concentration, and the absence of cost-adjusted trading artifacts. The paper contributes a reproducible VN100 benchmarking framework and a disciplined claim boundary: conditional predictive signal exists in the official artifacts, but global benchmark success, full-market representativeness, stable 63% performance, and practical trading readiness are not established.

## Keywords

VN100; stock direction forecasting; walk-forward validation; machine learning; stacking ensemble; directional accuracy; confidence filtering; regime diagnostics; leakage control.

# Chapter 1: Introduction

## 1.1 Research Context

Stock direction forecasting is a common application of machine learning in financial markets. In the Vietnamese equity market, such studies are especially attractive because historical price data, market microstructure changes, and sector-specific behavior can produce patterns that are difficult to summarize with simple rules. At the same time, financial forecasting is vulnerable to overclaiming. A model can appear accurate if evaluation windows are not separated chronologically, if future labels leak into training, or if a selected result is presented as a broad market conclusion.

This study addresses that methodological problem through an official VN100 benchmark artifact family that enforces a 2024-12-31 training-label cutoff and evaluates 2025 outcomes out of sample. The benchmark supports daily and hourly frequency diagnostics, baseline comparisons, confidence filtering, regime analysis, and statistical testing. It is therefore suitable as a foundation for a National Student Scientific Research style paper, provided that the paper clearly separates global benchmark results from selected diagnostic slices.

[Insert Figure 1 here]

## 1.2 Problem Statement

The main research problem is whether machine learning and ensemble models show directional forecasting value for VN100 stocks under a strict walk-forward design. The problem is not only whether a model reaches a high accuracy number. It is also whether the result survives clear train/evaluation separation, baseline comparison, confidence and regime diagnostics, statistical checks, and data-coverage scrutiny.

The official evidence does not support a global 60% benchmark pass. It does show conditional diagnostic signal in selected confidence-filtered and regime-specific slices. The paper therefore asks how to report these findings accurately without converting narrow diagnostic evidence into unsupported full-market or trading-readiness claims.

## 1.3 Research Objectives

The study has five objectives:

1. Build a reproducible VN100 walk-forward benchmark with explicit separation between training labels and held-out evaluation labels.
2. Compare supported machine learning models against simple directional baselines.
3. Evaluate whether confidence filtering or regime segmentation reveals conditional predictive signal.
4. Document cache coverage, ticker concentration, and artifact-schema limitations.
5. Define the remaining evidence required before making practical trading-readiness claims.

## 1.4 Research Questions

The study is guided by the following research questions:

1. Do the supported VN100 models exceed the official 60% global directional-accuracy benchmark under the 2025 held-out evaluation window?
2. Do machine learning and ensemble models outperform simple directional baselines in selected model/horizon slices?
3. Does confidence filtering produce a coverage-qualified strategy-level diagnostic pass?
4. Are predictive signals concentrated in specific market regimes such as bear regimes or high-volatility regimes?
5. Does the current usable-cache coverage support a full-market VN100 conclusion?
6. What evidence is still missing before practical trading-readiness claims can be made?

## 1.5 Scope and Limitations

The empirical scope is limited to the official VN100 benchmark artifact family. The model set consists of LightGBM, XGBoost, random forest, and stacking. No new model families are introduced in this paper draft.

The official evaluated tickers are ANV, BCM, BID, BMP, BVH, BWE, and CII. This means the current evidence should be treated as an initial VN100 benchmark implementation under limited usable-cache coverage, not as a definitive full-market VN100 evaluation. The selected confidence-filtered diagnostic is also concentrated in five tickers, with the top three contributing most selected rows.

The study does not claim trading readiness. The official selected slices do not yet have transaction-cost, slippage, turnover, drawdown, profit-factor, trade-list, equity-curve, or cost-adjusted return artifacts.

## 1.6 Contribution

The contribution of this study is methodological and empirical. Methodologically, it provides a leakage-aware VN100 walk-forward benchmark with explicit artifact-backed date fields, train-cutoff enforcement, baseline comparison, confidence filtering, regime diagnostics, statistical summaries, and claim governance. Empirically, it documents that the global benchmark does not pass 60%, while selected conditional diagnostics show evidence of directional signal that merits further validation.

The key contribution is not a deployment-ready trading strategy. It is a disciplined evidence framework that distinguishes benchmark-wide performance, selected strategy-level diagnostics, regime-specific diagnostics, and unresolved evidence gaps.

## 1.7 Paper Structure

Chapter 2 reviews the theoretical background for stock direction forecasting, walk-forward validation, machine learning models, ensemble methods, directional accuracy, statistical testing, and trading practicality. Chapter 3 describes the data scope, cache usability, model groups, validation design, metrics, and leakage safeguards. Chapter 4 presents empirical results and limitations. Chapter 5 concludes with the main findings and future research directions. The appendix maps each paper claim to the supporting artifact.

# Chapter 2: Literature Review and Theoretical Background

## 2.1 Stock Direction Forecasting

Stock direction forecasting focuses on predicting whether a future price or return will move upward or downward. This differs from point forecasting because the target is a directional class rather than an exact future price. Directional forecasts can be useful for research because they simplify evaluation into hit-rate style metrics, but they can also be misleading if accuracy is interpreted without considering class balance, transaction costs, and trading implementation constraints (Christoffersen & Diebold, 2006).

In the VN100 context, direction forecasting is challenging because market behavior may vary by ticker, liquidity, sector, and market regime. A result that appears strong in one subset may not generalize to the full universe. For this reason, the current study treats direction accuracy as a diagnostic forecasting metric rather than a direct trading-profit metric.

## 2.2 Time-Series Validation

Time-series forecasting requires chronological validation. Random splits can leak future information into training or validation and can overstate performance. Walk-forward validation is a practical alternative because it mimics the sequence in which models would be trained on past data and evaluated on later outcomes (Bergmeir & Benítez, 2012; Hyndman & Athanasopoulos, 2021).

The official VN100 benchmark follows this principle by recording a training-label cutoff of 2024-12-31 and evaluating held-out 2025 outcomes. The metadata field `training_label_cutoff_rule = target_timestamp <= train_cutoff` is central to the study because it defines the leakage boundary.

## 2.3 Machine Learning Models

The official benchmark evaluates LightGBM, XGBoost, random forest, and stacking. Random forests are tree ensembles based on bootstrap aggregation and feature randomness (Breiman, 2001). XGBoost and LightGBM are gradient-boosting methods designed to fit additive trees sequentially and efficiently (Chen & Guestrin, 2016; Ke et al., 2017). These models are common choices for tabular financial prediction because they can represent nonlinear interactions and handle mixed technical indicators without requiring sequence architectures.

The paper does not add sequence models or other new model families. This is intentional. The current research phase focuses on hardening the evidence base for the existing official benchmark before expanding the modeling space.

## 2.4 Ensemble and Stacking Models

Ensemble learning combines multiple models to improve stability or predictive performance. Stacking is an ensemble method in which base model outputs are combined by a meta-model (Wolpert, 1992). In this study, stacking is important because the selected confidence-filtered diagnostic comes from hourly stacking at horizon 1.

This result must be interpreted carefully. The selected stacking slice reaches 60.03% filtered accuracy only at 31.30% coverage. It is a strategy-level diagnostic pass, not a global benchmark pass and not proof of full-market performance.

## 2.5 Directional Accuracy and Statistical Testing

Directional accuracy measures the share of predictions where the predicted direction matches the actual direction. The official benchmark also includes binomial p-values against a 50% null and bootstrap confidence intervals. These statistics help distinguish weak directional variation from rows that are statistically above a simple chance benchmark. Forecast-comparison literature also motivates formal predictive-accuracy tests beyond raw score differences (Diebold & Mariano, 1995).

However, statistical significance is not the same as economic usefulness. A statistically significant directional hit rate can still fail after transaction costs, slippage, turnover, or poor position sizing. Therefore, the paper treats statistical tests as forecast-evaluation evidence, not as trading-readiness proof.

## 2.6 Transaction Cost and Trading Practicality

Trading practicality requires more evidence than directional accuracy. A practical trading claim would require executable signal rules, entry and exit conventions, transaction costs, slippage, liquidity assumptions, turnover, drawdown, profit factor, equity curves, and cost-adjusted returns. Execution-cost literature shows why such frictions can materially change realized performance (Almgren & Chriss, 2001).

The repository contains backtest modules that can support cost and slippage evaluation, but the official selected VN100 slices have not yet been evaluated through those modules. The current paper therefore explicitly states that practical trading readiness is not established.

## 2.7 Research Gap

The research gap is methodological. Many forecasting studies focus on model choice or headline accuracy, while this study emphasizes artifact-backed evaluation, train/evaluation separation, baseline comparison, confidence-filter diagnostics, regime-specific diagnostics, statistical evidence, concentration checks, and claim boundaries. The goal is to create a defensible VN100 evidence base before claiming broad market performance or trading usefulness.

# Chapter 3: Data and Methodology

## 3.1 Data Scope

This study evaluates VN100 stock-direction forecasting using the official artifact family at `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff`. The benchmark is configured for the VN100 universe, classification target mode, and a 2025 held-out evaluation window. The official configuration records a raw daily cache request range from 2006-01-01 to 2015-12-31 and a raw hourly cache request range from 2016-01-01 to 2025-12-31.

The daily benchmark should not be described as using only daily rows from 2006-2015. The manifest states that the daily benchmark method combines daily OHLCV for 2006-2015 with hourly OHLCV from 2016 onward resampled to daily. The official daily benchmark summary records effective training from 2006-01-03 to 2024-12-31 and effective evaluation from 2025-01-02 to 2025-12-31. The hourly benchmark summary records effective training from 2024-01-02 to 2024-12-31 and effective evaluation from 2025-01-02 to 2025-12-31.

The official evaluated tickers are ANV, BCM, BID, BMP, BVH, BWE, and CII.

[Insert Table 1 here]

## 3.2 Data Coverage and Cache Usability

The official run is a limited-cache VN100 benchmark rather than a definitive full-market VN100 evaluation. The artifact `usable_cache_summary.csv` shows no standalone daily cache rows marked benchmark-usable, because standalone daily rows end before the 2025 evaluation window. The benchmark-usable rows are seven partial hourly pairs, covering ANV, BCM, BID, BMP, BVH, BWE, and CII with actual/effective range from 2024-01-02 to 2025-12-31.

This coverage limitation affects interpretation. The empirical results are useful for testing the repository's walk-forward framework and conditional diagnostics, but they should not be generalized to all VN100 constituents. The paper should explicitly state that cache usability restricts representativeness.

[Insert Figure 2 here]

## 3.3 Model Groups and Baselines

The official model set consists of LightGBM, XGBoost, random forest, and stacking. These models are already supported by the benchmark runner and no new model family is introduced in this evidence pack. The benchmark also evaluates simple directional baselines: always-up, previous-direction, seeded random direction, and moving-average signal.

The role of the baselines is methodological: they provide directional comparison points so that model accuracy can be interpreted against simple non-ML policies.

[Insert Table 2 here]

## 3.4 Walk-Forward Validation Design

The benchmark uses walk-forward out-of-sample evaluation with a strict training-label cutoff. The official configuration records `train_cutoff = 2024-12-31` and `training_label_cutoff_rule = target_timestamp <= train_cutoff`. This rule is central to the study because it prevents target labels from the 2025 evaluation window from entering training labels.

The official evaluation window is 2025-01-01 to 2025-12-31, with effective evaluation rows beginning on 2025-01-02 in both daily and hourly summaries.

## 3.5 Evaluation Metrics

The primary metric is directional accuracy, measured by comparing `actual_direction` and `predicted_direction` in the official prediction files. The paper also reports model/horizon accuracy, baseline deltas, confidence-filtered accuracy, coverage ratio after filtering, regime-specific accuracy, binomial p-values against a 50% null, and bootstrap confidence intervals.

The official generated tables map these metrics to paper evidence:

- Table 3: global benchmark results.
- Table 4: baseline delta summary.
- Table 5: confidence-filtered diagnostics.
- Table 6: regime-specific diagnostics.
- Table 7: statistical significance summary.
- Table 8: robustness and limitation matrix.

## 3.6 Methodological Safeguards Against Leakage

The main leakage safeguard is the training-label cutoff rule recorded in the official run configuration. The benchmark allows actual rows after the train cutoff only for out-of-sample label evaluation, not for training labels. This distinction is important because the official artifact family evaluates 2025 outcomes while restricting training labels to target timestamps on or before 2024-12-31.

The artifact verification note records the relevant date fields and confirms that the raw daily, raw hourly, effective training, and effective evaluation ranges should be described separately. The paper should preserve this distinction to avoid implying that all 2025 information was available during training.

## 3.7 Limitations of the Current Dataset

The current dataset has four material limitations. First, the official evaluated ticker set contains only seven tickers, which weakens full-market representativeness. Second, daily standalone cache rows are not benchmark-usable for the 2025 evaluation window, so the daily benchmark depends on the documented hybrid daily construction. Third, confidence-threshold sweep evidence is partial: the combined sweep artifact covers hourly stacking h=1, while the daily threshold-sweep file contains no data rows. Fourth, the current official artifacts do not include cost-adjusted returns, turnover, drawdown, profit factor, slippage-applied execution, or trade-level PnL for selected diagnostic slices.

These limitations do not invalidate the benchmark as a research framework, but they constrain the claims. The paper should present the results as evidence of leakage-aware VN100 diagnostic forecasting under limited cache coverage, not as proof of deployable trading performance or full-market stability.

# Chapter 4: Empirical Results and Discussion

## 4.1 Official Global Benchmark Results

The official 2025 VN100 walk-forward benchmark produced nonzero prediction artifacts for both daily and hourly frequencies. The daily benchmark records 26,104 predictions with overall directional accuracy of 0.5318725099601593. The hourly benchmark records 127,944 predictions with overall directional accuracy of 0.5128571875195398.

Neither frequency passed the global 60% benchmark threshold. This is the central empirical boundary for the paper: the official run does not support a global benchmark pass.

[Insert Table 3 here]

[Insert Figure 3 here]

## 4.2 Baseline Comparison

The official baseline delta artifacts compare model accuracy against always-up, previous-direction, seeded random direction, and moving-average signal baselines. Some model/horizon rows outperform simple baselines. For example, the daily best model accuracy is 0.5697236180904522, while the best daily baseline accuracy recorded in the benchmark summary is 0.5452261306532663. The hourly best model accuracy is 0.5559340509606213, while the best hourly baseline accuracy is 0.5054466230936819.

These deltas are useful diagnostic evidence, but they are not a global benchmark pass and they are not trading-profitability evidence.

[Insert Table 4 here]

## 4.3 Confidence-Filtered Strategy Diagnostics

The official confidence-threshold sweep identifies one selected strategy-level diagnostic slice: hourly stacking at horizon 1 with threshold 0.57. This slice has 2,297 evaluated rows, coverage ratio 0.3129854203569969, and filtered accuracy 0.6003482803656944. Therefore, the paper may state that a strategy-level diagnostic pass exists at 60.03% accuracy and 31.30% coverage.

The claim must remain narrow. Under coverage floors of 50% and 40%, no available confidence-filtered row reaches 60% accuracy. The daily threshold-sweep file contains no data rows, and the available combined sweep rows cover hourly stacking h=1 only.

[Insert Table 5 here]

[Insert Figure 4 here]

## 4.4 Regime-Specific Diagnostics

The strongest regime-specific daily diagnostic appears in the bear regime at horizon 20. Daily LightGBM h=20 reaches 0.6959459459459459 accuracy over 444 observations, and daily XGBoost h=20 reaches 0.6914414414414415 over the same observation count. These are regime-specific 63%+ diagnostics, not global benchmark results.

The paper should not describe these rows as a stable full-market 63% method. The finding is conditional on the regime slice, the seven evaluated tickers, and the official 2025 window.

[Insert Table 6 here]

[Insert Figure 5 here]

## 4.5 Statistical Significance

The official significance summaries include binomial p-values against a 50% null and bootstrap confidence intervals. Several model/horizon rows are statistically above the 50% null. For instance, hourly stacking h=1 records accuracy 0.5559340509606213 with a binomial p-value of 4.771490994537493e-22.

Statistical significance should be interpreted as directional-accuracy evidence only. It does not prove practical trading readiness, cost-adjusted profitability, or multi-window robustness.

[Insert Table 7 here]

## 4.6 Ticker Concentration and Representativeness

Ticker concentration diagnostics show that the global daily and global hourly prediction rows are not dominated by a single ticker by prediction count. However, the selected hourly confidence slice is concentrated: it contains five tickers, and the top three tickers contribute about 79.49% of selected rows. In that selected slice, BID contributes 31.17% of rows, CII contributes 28.73%, and BCM contributes 19.59%.

This concentration affects the interpretation of the selected strategy-level pass. The result remains useful as a conditional diagnostic, but it should not be generalized as full-market VN100 evidence.

## 4.7 Cost/Slippage Readiness Gap

The official selected VN100 slices have not been evaluated with transaction costs, slippage, turnover, drawdown, profit factor, trade lists, equity curves, or cost-adjusted returns. The repository contains backtest modules that can model fees, slippage, turnover, drawdown, and profit factor, but those modules have not produced official artifacts for the selected VN100 confidence or regime slices.

Therefore, practical trading readiness is not established. The paper must not claim profitability, deployment readiness, or executable investment suitability.

## 4.8 Discussion of Empirical Limitations

The empirical results support a careful and bounded conclusion. Global benchmark pass: no. Strategy-level diagnostic pass: yes, hourly stacking h=1 at threshold 0.57 with 60.03% accuracy and 31.30% coverage. Stable full-market 63% method: no. Regime-specific 63%+ diagnostic: yes, but bear-regime only and not a global pass. Practical trading readiness: not established.

The strongest interpretation is that the official benchmark provides leakage-aware evidence of conditional predictive signal under limited cache coverage. The weakest interpretation, which should be avoided, is that the current artifacts prove a deployable trading system or a stable full-market VN100 method.

[Insert Table 8 here]

# Chapter 5: Conclusion and Recommendations

## 5.1 Main Findings

The official 2025 VN100 walk-forward benchmark does not pass the global 60% directional-accuracy threshold. Daily overall accuracy is 53.19% over 26,104 predictions, and hourly overall accuracy is 51.29% over 127,944 predictions.

The evidence nevertheless shows conditional signal. The selected confidence-filtered diagnostic, hourly stacking h=1 at threshold 0.57, reaches 60.03% accuracy with 31.30% coverage. Daily bear-regime h=20 diagnostics exceed 63% accuracy, including LightGBM at 69.59% and XGBoost at 69.14% over 444 observations.

These findings must remain bounded. The current evidence does not establish a stable full-market 63% method, does not represent the full VN100 universe, and does not establish practical trading readiness.

## 5.2 Academic Contribution

The study contributes a reproducible and leakage-aware VN100 benchmark design. It also contributes a claim-governance approach that separates global benchmark results from selected confidence diagnostics, regime diagnostics, statistical evidence, concentration checks, and practical-readiness gaps.

This contribution is useful because it prevents attractive diagnostic slices from being overstated as full-market proof. It also creates a clear roadmap for future experiments.

## 5.3 Practical Implications

The results suggest that conditional directional signal may exist in selected VN100 contexts. However, this implication is research-oriented rather than deployment-oriented. The current artifacts should not be used to justify live trading, portfolio allocation, or investment recommendations.

Before practical use can be considered, the selected signal slices must be converted into executable strategies and tested with transaction costs, slippage, turnover, drawdown, profit factor, liquidity assumptions, and out-of-sample multi-window robustness.

## 5.4 Recommendations for Future Work

Future work should prioritize five steps:

1. Expand benchmark-usable VN100 cache coverage beyond the current seven evaluated tickers.
2. Repeat the official benchmark across additional evaluation windows.
3. Generate broader confidence sweeps, including daily sweep rows and all relevant model/horizon combinations.
4. Validate regime findings using ex-ante regime rules instead of post-hoc interpretation.
5. Run cost/slippage-aware backtests for selected confidence and regime slices, including turnover, drawdown, profit factor, trade-level records, equity curves, and cost-adjusted returns.

Only after these steps should the project consider adding new model families.

## 5.5 Final Claim Boundary

The safe final claim is:

The official 2025 VN100 walk-forward benchmark did not pass the global 60% directional-accuracy threshold, but confidence-filtered and bear-regime diagnostics show conditional predictive signal that merits further validation.

The paper must not claim that the benchmark passed globally, that the model is ready for live trading, that the current evidence proves full-market VN100 representativeness, or that the project has established a stable full-market 63% method.

# References

Use `reports/NCKH_REFERENCES_APA7.md` as the APA 7 reference scaffold. Final submission should verify institutional formatting requirements before submission.

# Appendix: Artifact Map

| Artifact | Paper use | Status |
|---|---|---|
| `reports/NCKH_RESEARCH_DESIGN_VN100.md` | Research questions, objectives, hypotheses, claim boundary | ready |
| `reports/NCKH_EXPERIMENT_INVENTORY.md` | Script, module, output, and missing-experiment inventory | ready |
| `reports/NCKH_PAPER_OUTLINE_VN100.md` | Chapter structure and safe-claim guidance | ready |
| `reports/NCKH_PAPER_TABLES_AND_FIGURES_PLAN.md` | Table and figure plan | ready |
| `reports/NCKH_CHAPTER3_METHODOLOGY_DRAFT.md` | Source for Chapter 3 | ready |
| `reports/NCKH_CHAPTER4_EMPIRICAL_RESULTS_DRAFT.md` | Source for Chapter 4 | ready |
| `reports/NCKH_RESULTS_CLAIM_REGISTER.md` | Safe, conditional, and unsafe wording | ready |
| `reports/generated/paper_tables/` | Tables 1-8 as CSV and Markdown | ready or partial by table |
| `reports/generated/paper_figures/` | Figures 1-5 as Markdown schematics and PNG charts | ready or partial by figure |
| `reports/generated/paper_notes/` | Artifact status and claim-boundary notes | ready |
| `reports/generated/vn100_artifact_date_schema_verification.md` | Date/schema verification | ready |
| `reports/generated/vn100_confidence_coverage_review.md` | Coverage-floor interpretation | partial because daily sweep rows are missing |
| `reports/generated/vn100_cost_slippage_readiness_review.md` | Trading-readiness gap | ready |
| `reports/generated/vn100_ticker_concentration_summary.md` | Concentration and representativeness diagnostics | ready |
