# Walk-Forward Evaluation of Machine Learning and Ensemble Models for VN100 Stock Direction Forecasting: V2 Evidence-Upgrade Manuscript

Vietnamese title: Đánh giá thực nghiệm các mô hình học máy và ensemble trong dự báo xu hướng cổ phiếu VN100 theo phương pháp walk-forward ngoài mẫu.

## Abstract

This V2 manuscript evaluates machine learning and ensemble models for VN100 stock-direction forecasting under a leakage-aware walk-forward design. The official VN100 benchmark artifact family applies a 2024-12-31 training-label cutoff and evaluates 2025 held-out outcomes. The model set consists of LightGBM, XGBoost, random forest, and stacking, compared with simple directional baselines. The official daily benchmark records 26,104 predictions with 53.19% directional accuracy, while the official hourly benchmark records 127,944 predictions with 51.29% directional accuracy. Neither frequency passes the global 60% benchmark threshold. Conditional evidence exists: the original hourly stacking h=1 confidence-filtered diagnostic reaches 60.03% accuracy at 31.30% coverage, and daily bear-regime h=20 diagnostics exceed 63% in the 2025 window. Additional derived diagnostics broaden the confidence-filter evidence and support a lagged ex-ante bear-regime result in the 2025 window, while ticker coverage, multi-window validation, and executable trading-readiness evidence remain limited. The paper contributes a reproducible VN100 benchmarking framework and a disciplined claim boundary: conditional predictive signal is present in selected diagnostics, but global benchmark success, full-market VN100 representativeness, stable multi-window 63% performance, trading readiness, and profitability are not established.

## Keywords

VN100; stock direction forecasting; walk-forward validation; machine learning; stacking ensemble; directional accuracy; confidence filtering; ex-ante regime validation; transaction-cost proxy; leakage control.

# Chapter 1: Introduction

## 1.1 Research Context

Stock direction forecasting is a common application of machine learning in financial markets. In the Vietnamese equity market, such studies are attractive because historical prices, intraday behavior, market-regime shifts, and ticker-specific liquidity can create nonlinear patterns that are difficult to summarize with simple rules. At the same time, financial forecasting is vulnerable to overclaiming. A result can appear strong if evaluation windows are not separated chronologically, if future labels leak into training, or if a selected diagnostic slice is presented as a broad market conclusion.

This study addresses that methodological problem through an official VN100 benchmark artifact family that enforces a 2024-12-31 training-label cutoff and evaluates 2025 outcomes out of sample. The benchmark supports daily and hourly frequency diagnostics, baseline comparisons, confidence filtering, regime analysis, statistical testing, concentration checks, and claim governance.

**Figure 1. Research Pipeline**

Source schematic: `reports/generated/paper_figures/figure1_research_pipeline.md`

```text
VN100 official cache artifacts
  -> cache usability and date/schema verification
  -> train-label cutoff: target_timestamp <= 2024-12-31
  -> walk-forward out-of-sample prediction on 2025 labels
  -> model and baseline accuracy summaries
  -> confidence-filter and regime diagnostics
  -> paper tables, figures, and claim register
```

## 1.2 Problem Statement

The main research problem is whether machine learning and ensemble models show directional forecasting value for VN100 stocks under a strict walk-forward design. The question is not only whether a model reaches a high accuracy number. It is also whether the result survives clear train/evaluation separation, baseline comparison, confidence and regime diagnostics, statistical checks, ticker-concentration review, and data-coverage scrutiny.

The official evidence does not support a global 60% benchmark pass. It does show conditional diagnostic signal in selected confidence-filtered and regime-specific slices. The V2 evidence upgrade adds derived confidence-sweep expansion and lagged ex-ante regime validation, but these additions remain diagnostic and do not establish full-market or trading-readiness claims.

## 1.3 Research Objectives

The study has five objectives:

1. Build a reproducible VN100 walk-forward benchmark with explicit separation between training labels and held-out evaluation labels.
2. Compare supported machine learning models against simple directional baselines.
3. Evaluate whether confidence filtering or regime segmentation reveals conditional predictive signal.
4. Document cache coverage, ticker concentration, evidence-gap closure, and artifact-schema limitations.
5. Define the remaining evidence required before practical trading-readiness claims can be made.

## 1.4 Research Questions

The study is guided by the following research questions:

1. Do the supported VN100 models exceed the official 60% global directional-accuracy benchmark under the 2025 held-out evaluation window?
2. Do machine learning and ensemble models outperform simple directional baselines in selected model/horizon slices?
3. Does confidence filtering produce a coverage-qualified strategy-level diagnostic pass?
4. Do derived V2 confidence diagnostics broaden the original confidence-filtering evidence?
5. Are predictive signals concentrated in specific regimes, and do lagged ex-ante regime proxies support the 2025 bear-regime diagnostics?
6. Does the current usable-cache coverage support a full-market VN100 conclusion?
7. What evidence is still missing before practical trading-readiness claims can be made?

## 1.5 Scope and Limitations

The empirical scope is limited to the official VN100 benchmark artifact family and the derived V2 diagnostics generated from that artifact family. The model set consists of LightGBM, XGBoost, random forest, and stacking. No new model families are introduced in this manuscript.

The official evaluated tickers are ANV, BCM, BID, BMP, BVH, BWE, and CII. The cache audit confirms that the current evidence should be treated as a limited usable-cache VN100 benchmark, not as a definitive full-market VN100 evaluation. The selected confidence-filtered diagnostic is also concentrated in a subset of the evaluated tickers.

The study does not claim trading readiness. V2 adds cost/slippage proxy diagnostics, but those diagnostics use target-return proxies rather than executable entry/exit prices, liquidity filters, fill assumptions, or deployment constraints.

## 1.6 Contribution

The contribution of this study is methodological and empirical. Methodologically, it provides a leakage-aware VN100 walk-forward benchmark with explicit artifact-backed date fields, train-cutoff enforcement, baseline comparison, confidence filtering, regime diagnostics, statistical summaries, concentration checks, evidence-gap closure, and claim governance. Empirically, it documents that the global benchmark does not pass 60%, while selected conditional diagnostics show evidence of directional signal that merits further validation.

The V2 contribution is an evidence-upgrade integration: derived confidence sweeps broaden the confidence-filtering evidence, lagged ex-ante regime diagnostics reduce post-hoc concern for the 2025 bear-regime h=20 finding, and cost/slippage proxy diagnostics clarify why trading readiness remains unresolved.

## 1.7 Paper Structure

Chapter 2 reviews the theoretical background for stock direction forecasting, walk-forward validation, machine learning models, ensemble methods, directional accuracy, statistical testing, and trading practicality. Chapter 3 describes the data scope, cache usability, model groups, validation design, metrics, and leakage safeguards. Chapter 4 presents empirical results, V2 evidence upgrades, and limitations. Chapter 5 concludes with the main findings and future research directions. The appendices map artifacts, cost/slippage proxy diagnostics, and evidence-gap closure status.

# Chapter 2: Literature Review and Theoretical Background

## 2.1 Stock Direction Forecasting

Stock direction forecasting predicts whether a future price or return will move upward or downward. This differs from point forecasting because the target is a directional class rather than an exact future price. Directional forecasts can be useful for research because they simplify evaluation into hit-rate style metrics, but they can also be misleading if accuracy is interpreted without considering class balance, transaction costs, and implementation constraints (Christoffersen & Diebold, 2006).

In the VN100 context, direction forecasting is challenging because behavior can vary by ticker, liquidity, sector, and market regime. A result that appears strong in one subset may not generalize to the full universe. This study therefore treats directional accuracy as a forecasting diagnostic, not as a direct trading-profit metric.

## 2.2 Time-Series Validation

Time-series forecasting requires chronological validation. Random splits can leak future information into training or validation and can overstate performance. Walk-forward validation is a practical alternative because it mimics the sequence in which models are trained on past data and evaluated on later outcomes (Bergmeir & Benítez, 2012; Hyndman & Athanasopoulos, 2021).

The official VN100 benchmark follows this principle by recording a training-label cutoff of 2024-12-31 and evaluating held-out 2025 outcomes. The metadata field `training_label_cutoff_rule = target_timestamp <= train_cutoff` defines the leakage boundary.

## 2.3 Machine Learning Models

The official benchmark evaluates LightGBM, XGBoost, random forest, and stacking. Random forests are tree ensembles based on bootstrap aggregation and feature randomness (Breiman, 2001). XGBoost and LightGBM are gradient-boosting methods designed to fit additive trees sequentially and efficiently (Chen & Guestrin, 2016; Ke et al., 2017). These models are common choices for tabular financial prediction because they can represent nonlinear interactions and handle mixed technical indicators without requiring sequence architectures.

This paper does not add sequence models or new model families. The research focus is evidence governance for the supported official benchmark.

## 2.4 Ensemble and Stacking Models

Ensemble learning combines multiple models to improve stability or predictive performance. Stacking is an ensemble method in which base model outputs are combined by a meta-model (Wolpert, 1992). In this study, stacking is important because the original selected confidence-filtered diagnostic comes from hourly stacking at horizon 1.

This result must be interpreted carefully. The selected stacking slice reaches 60.03% filtered accuracy only at 31.30% coverage. It is a strategy-level diagnostic pass, not a global benchmark pass and not proof of full-market performance.

## 2.5 Directional Accuracy and Statistical Testing

Directional accuracy measures the share of predictions where the predicted direction matches the actual direction. The official benchmark also includes binomial p-values against a 50% null and bootstrap confidence intervals. These statistics help distinguish weak directional variation from rows that are statistically above a simple chance benchmark. Forecast-comparison literature also motivates formal predictive-accuracy tests beyond raw score differences (Diebold & Mariano, 1995).

Statistical significance is not the same as economic usefulness. A statistically significant hit rate can still fail after transaction costs, slippage, turnover, or poor position sizing. Therefore, statistical tests are treated as forecast-evaluation evidence, not as trading-readiness proof.

## 2.6 Transaction Cost and Trading Practicality

Trading practicality requires more evidence than directional accuracy. A practical trading claim would require executable signal rules, entry and exit conventions, transaction costs, slippage, liquidity assumptions, turnover, drawdown, profit factor, equity curves, and cost-adjusted returns. Execution-cost literature shows why such frictions can materially change realized performance (Almgren & Chriss, 2001).

The V2 evidence package adds cost/slippage proxy diagnostics. These are useful for research but are not execution-ready because they do not model real entry/exit fills or liquidity constraints. The paper therefore explicitly states that practical trading readiness is not established.

## 2.7 Research Gap

The research gap is methodological. Many forecasting studies focus on model choice or headline accuracy, while this study emphasizes artifact-backed evaluation, train/evaluation separation, baseline comparison, confidence-filter diagnostics, regime-specific diagnostics, statistical evidence, concentration checks, cost/slippage proxy review, and claim boundaries. The goal is to create a defensible VN100 evidence base before claiming broad market performance or trading usefulness.

# Chapter 3: Data and Methodology

## 3.1 Official Artifact Family and Evaluation Scope

This study evaluates VN100 stock-direction forecasting using the official artifact family at `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff`. The benchmark is configured for the VN100 universe, classification target mode, and a 2025 held-out evaluation window. The official configuration records `train_cutoff = 2024-12-31` and `training_label_cutoff_rule = target_timestamp <= train_cutoff`.

The official evaluation window is 2025-01-01 to 2025-12-31. The effective evaluation rows begin on 2025-01-02 for both daily and hourly summaries. The official evaluated tickers are ANV, BCM, BID, BMP, BVH, BWE, and CII.

## 3.2 Hybrid Daily Construction

The daily benchmark should not be described as using only daily rows from 2006-2015. The manifest states that the daily benchmark method combines daily OHLCV for 2006-2015 with hourly OHLCV from 2016 onward resampled to daily. The official daily benchmark summary records effective training from 2006-01-03 to 2024-12-31 and effective evaluation from 2025-01-02 to 2025-12-31. The hourly benchmark summary records effective training from 2024-01-02 to 2024-12-31 and effective evaluation from 2025-01-02 to 2025-12-31.

**Figure 2. Walk-Forward Validation Design**

Source schematic: `reports/generated/paper_figures/figure2_walk_forward_design.md`

```text
Raw daily request:  2006-01-01 -> 2015-12-31
Raw hourly request: 2016-01-01 -> 2025-12-31

Training labels allowed through: 2024-12-31
Cutoff rule: target_timestamp <= train_cutoff

Official evaluation window: 2025-01-01 -> 2025-12-31
Effective daily evaluation: 2025-01-02 -> 2025-12-31
Effective hourly evaluation: 2025-01-02 -> 2025-12-31
```

## 3.3 Dataset and Cache Usability

**Table 1. Dataset and Evaluation Scope**

Source: `reports/generated/paper_tables/table1_dataset_evaluation_scope.md`

| item | value | detail |
| --- | --- | --- |
| Universe | VN100 | Official benchmark universe. |
| Raw daily cache request range | 2006-01-01 to 2015-12-31 | Manifest raw daily range: 2006-01-01 to 2015-12-31. |
| Raw hourly cache request range | 2016-01-01 to 2025-12-31 | Manifest raw hourly range: 2016-01-01 to 2025-12-31. |
| Training-label cutoff | 2024-12-31 | Rule: target_timestamp <= train_cutoff. |
| Official evaluation window | 2025-01-01 to 2025-12-31 | Held-out 2025 target outcomes. |
| Effective daily evaluation range | 2025-01-02 to 2025-12-31 | Daily predictions: 26,104. |
| Effective hourly evaluation range | 2025-01-02 to 2025-12-31 | Hourly predictions: 127,944. |
| Evaluated tickers | ANV, BCM, BID, BMP, BVH, BWE, CII | 7 tickers evaluated in official summaries. |
| Daily benchmark-usable cache rows | 0 of 104 | Usable tickers: none; actual range: n/a to n/a. |
| Hourly benchmark-usable cache rows | 7 of 104 | Usable tickers: ANV, BCM, BID, BMP, BVH, BWE, CII; actual range: 2024-01-02 to 2025-12-31. |

V2 cache-audit note: `reports/generated/evidence_gap_closure/vn100_cache_coverage_audit.md` reports 104 considered VN100 tickers, 60 local daily cache files, 86 local hourly cache files, and only 7 benchmark-usable 2025 tickers. The expanded official benchmark was not generated because the audit did not find additional benchmark-usable 2025 tickers without a cache-expansion or heavy rerun. The official evaluated universe therefore remains limited.

## 3.4 Model Groups and Baselines

The official model set consists of LightGBM, XGBoost, random forest, and stacking. These models are already supported by the benchmark runner, and no new model family is introduced in V2. The benchmark also evaluates simple directional baselines: always-up, previous-direction, seeded random direction, and moving-average signal.

The baselines provide directional comparison points so model accuracy can be interpreted against simple non-ML policies. They do not provide cost-adjusted trading evidence.

## 3.5 Walk-Forward Validation Design

The benchmark uses walk-forward out-of-sample evaluation with a strict training-label cutoff. The official configuration records `train_cutoff = 2024-12-31` and `training_label_cutoff_rule = target_timestamp <= train_cutoff`. This rule prevents target labels from the 2025 evaluation window from entering training labels.

The official evaluation window is 2025-01-01 to 2025-12-31, with effective evaluation rows beginning on 2025-01-02. V2 diagnostics are derived from existing official prediction rows and do not rerun training or benchmarking.

## 3.6 Evaluation Metrics

The primary metric is directional accuracy, measured by comparing `actual_direction` and `predicted_direction`. The paper also reports model/horizon accuracy, baseline deltas, confidence-filtered accuracy, coverage ratio after filtering, regime-specific accuracy, binomial p-values against a 50% null, bootstrap confidence intervals, concentration measures, and cost/slippage proxy metrics.

V2 adds derived confidence-sweep metrics across all available prediction-row combinations and lagged ex-ante regime proxy metrics. These are diagnostic extensions, not new official benchmark runs.

## 3.7 Methodological Safeguards Against Leakage

The main leakage safeguard is the training-label cutoff rule recorded in the official run configuration. The benchmark allows actual rows after the train cutoff only for out-of-sample label evaluation, not for training labels. This distinction is important because the official artifact family evaluates 2025 outcomes while restricting training labels to target timestamps on or before 2024-12-31.

The V2 ex-ante regime proxy uses lagged prior realized target-return information only. It does not use the current row's future return to label the current prediction row.

## 3.8 Limitations of the Current Dataset

The current dataset has material limitations. First, the official evaluated ticker set contains only seven benchmark-usable tickers. Second, standalone daily cache rows are not benchmark-usable for the 2025 evaluation window, so the daily benchmark depends on documented hybrid daily construction. Third, V2 confidence-sweep expansion is derived from existing prediction rows and remains single-window evidence. Fourth, cost/slippage diagnostics are proxy diagnostics, not execution-ready trading artifacts. Fifth, official 2022-2024 walk-forward artifacts remain unavailable.

# Chapter 4: Empirical Results and Discussion

## 4.1 Official Global Benchmark Results

The official 2025 VN100 walk-forward benchmark produced nonzero prediction artifacts for both daily and hourly frequencies. The daily benchmark records 26,104 predictions with 53.19% directional accuracy. The hourly benchmark records 127,944 predictions with 51.29% directional accuracy.

Global 60% benchmark pass: no. This is the central empirical boundary for the paper.

**Table 3. Global Benchmark Results**

Source: `reports/generated/paper_tables/table3_global_benchmark_results.md`

| frequency | overall_accuracy | n_predictions | best_model_accuracy | best_model | best_model_horizon | best_baseline_accuracy | best_model_delta_vs_best_baseline | passed_60pct_global | evaluated_tickers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| daily | 53.19% | 26,104 | 56.97% | random_forest | 20 | 54.52% | 2.45 pp | false | ANV, BCM, BID, BMP, BVH, BWE, CII |
| hourly | 51.29% | 127,944 | 55.59% | stacking | 1 | 50.54% | 5.05 pp | false | ANV, BCM, BID, BMP, BVH, BWE, CII |

![Figure 3. Accuracy by model/horizon](reports/generated/paper_figures/figure3_accuracy_by_model_horizon.png)

## 4.2 Baseline Comparison

The official baseline delta artifacts compare model accuracy against always-up, previous-direction, seeded random direction, and moving-average signal baselines. Some model/horizon rows outperform simple baselines. For example, the daily best model accuracy is 56.97%, while the best daily baseline accuracy is 54.52%. The hourly best model accuracy is 55.59%, while the best hourly baseline accuracy is 50.54%.

These deltas are useful diagnostic evidence. They are not a global benchmark pass and they are not trading-profitability evidence.

**Table 4. Baseline Delta Summary**

Source: `reports/generated/paper_tables/table4_baseline_delta_summary.md`

| frequency | model | horizon | comparison baseline | model_accuracy | baseline_accuracy | accuracy_delta | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| daily | random_forest | 20 | best daily baseline | 56.97% | 54.52% | +2.45 pp | Best daily model row exceeds best daily baseline but remains below 60%. |
| daily | xgboost | 20 | always_up | 56.97% | 54.52% | +2.45 pp | Daily h=20 boosted-tree row beats always-up baseline. |
| hourly | stacking | 1 | best hourly baseline | 55.59% | 50.54% | +5.05 pp | Best hourly model row exceeds best hourly baseline but remains below 60%. |
| hourly | xgboost | 1 | random_seeded_direction | 54.94% | 50.02% | +4.92 pp | Hourly h=1 XGBoost beats random directional baseline. |

The complete generated Table 4 is retained in `reports/generated/paper_tables/table4_baseline_delta_summary.md`.

## 4.3 Original Confidence-Filtered Diagnostic

The original official confidence-threshold sweep identifies one selected strategy-level diagnostic slice: hourly stacking at horizon 1 with threshold 0.57. This slice has 2,297 evaluated rows, 31.30% coverage, and 60.03% filtered accuracy. It remains a narrow strategy-level diagnostic.

**Table 5. Original Confidence-Filtered Strategy Diagnostics**

Source: `reports/generated/paper_tables/table5_confidence_filtered_diagnostics.md`

| scope | frequency | model | horizon | threshold | total_rows | evaluated_rows | coverage_ratio | filtered_accuracy | passed_60pct | selected_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_at_coverage_floor_50% | hourly | stacking | 1 | 0.55 | 7,339 | 3,782 | 51.53% | 58.17% | false | false |
| best_at_coverage_floor_40% | hourly | stacking | 1 | 0.56 | 7,339 | 3,019 | 41.14% | 59.06% | false | false |
| best_at_coverage_floor_30% | hourly | stacking | 1 | 0.57 | 7,339 | 2,297 | 31.30% | 60.03% | true | true |
| selected_candidate | hourly | stacking | 1 | 0.57 | 7,339 | 2,297 | 31.30% | 60.03% | true | true |

The result should not be described as a global benchmark pass. It is a confidence-filtered selected slice with limited coverage.

## 4.4 Derived V2 Confidence-Sweep Expansion

A derived v2 confidence sweep over official prediction rows broadens the confidence-filtering evidence beyond the original hourly stacking h=1 artifact. The v2 sweep is derived from existing official prediction rows. It is not a new official benchmark rerun. It covers 32 available frequency/model/horizon combinations and uses 154,048 prediction rows.

**Table 5B. V2 Coverage-Floor Confidence Candidates**

Source: `reports/generated/evidence_gap_closure/vn100_full_confidence_sweep_report.md`

| coverage floor | candidate | evaluated rows | coverage | filtered accuracy | passed 60% |
| --- | --- | --- | --- | --- | --- |
| >= 50% | daily XGBoost h=20 threshold 0.86 | 844 | 53.02% | 60.55% | yes |
| >= 40% | daily XGBoost h=20 threshold 0.90 | 714 | 44.85% | 60.78% | yes |
| >= 30% | daily stacking h=20 threshold 0.69 | 482 | 30.28% | 62.03% | yes |
| >= 20% | daily stacking h=20 threshold 0.71 | 344 | 21.61% | 64.53% | yes |

![Figure 4. V2 confidence threshold versus coverage/accuracy](reports/generated/evidence_gap_closure/vn100_confidence_threshold_coverage_accuracy_v2.png)

These findings strengthen conditional confidence-filtering evidence. They do not change the official global benchmark result. The global benchmark still fails the 60% threshold, and the V2 sweep remains a derived single-window diagnostic.

## 4.5 Original Regime-Specific Diagnostics

The strongest original regime-specific daily diagnostic appears in the bear regime at horizon 20. Daily LightGBM h=20 bear-regime accuracy is 69.59% over 444 observations. Daily XGBoost h=20 bear-regime accuracy is 69.14% over 444 observations. These are diagnostic, not global.

**Table 6. Regime-Specific Diagnostics**

Source: `reports/generated/paper_tables/table6_regime_specific_diagnostics.md`

| frequency | regime | best_model | horizon | n_obs | accuracy | passed_60pct | reliable | top_ticker_by_contribution | top_ticker_contribution_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| daily | bear | lightgbm | 20 | 444 | 69.59% | true | true | BVH | 19.37% |
| daily | high_volatility | random_forest | 20 | 622 | 60.93% | true | true |  |  |
| hourly | high_volatility | stacking | 1 | 2,012 | 56.51% | false | true | BID | 16.30% |
| hourly | sideways | stacking | 1 | 4,760 | 55.76% | false | true |  |  |

![Figure 5. Regime-specific accuracy](reports/generated/paper_figures/figure5_regime_specific_accuracy.png)

The bear-regime h=20 result should not be described as a stable full-market 63% method. It is conditional on regime, window, and the seven-ticker benchmark-usable set.

## 4.6 Lagged Ex-Ante Regime Validation

The V2 evidence package adds a lagged ex-ante regime proxy. The proxy uses prior realized target-return information only within each ticker/frequency/model/horizon sequence. It does not use the current row's future return to label the current prediction row.

The lagged ex-ante proxy supports the 2025 daily bear-regime diagnostic for h=20.

**Table 6B. Lagged Ex-Ante Bear-Regime Diagnostics**

Source: `reports/generated/evidence_gap_closure/vn100_exante_regime_accuracy_summary.csv`

| frequency | exante_regime | model | horizon | n_obs | accuracy | passed_60pct | passed_63pct | reliable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| daily | bear | lightgbm | 20 | 309 | 66.34% | true | true | true |
| daily | bear | xgboost | 20 | 309 | 65.05% | true | true | true |

This reduces post-hoc concern for the 2025 bear-regime finding only. It does not establish multi-window regime stability and does not establish a stable 63% full-market method.

## 4.7 Statistical Significance

The official significance summaries include binomial p-values against a 50% null and bootstrap confidence intervals. Several model/horizon rows are statistically above the 50% null. For instance, hourly stacking h=1 records 55.59% accuracy with a binomial p-value of 4.771490994537493e-22.

Statistical significance should be interpreted as directional-accuracy evidence only. It does not prove profitability or trading readiness. Multiple-testing and selected-slice issues remain limitations.

**Table 7. Statistical Significance Summary**

Source: `reports/generated/paper_tables/table7_statistical_significance_summary.md`

| frequency | model | horizon | n_obs | accuracy | null_accuracy | binomial_p_value | bootstrap_ci_low | bootstrap_ci_high | significant_at_5pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| daily | lightgbm | 20 | 1,592 | 55.97% | 50.00% | 1.0570499989873873e-06 | 53.64% | 58.23% | true |
| daily | random_forest | 20 | 1,592 | 56.97% | 50.00% | 1.4486683327839042e-08 | 54.52% | 59.17% | true |
| daily | xgboost | 20 | 1,592 | 56.97% | 50.00% | 1.4486683327839042e-08 | 54.52% | 59.55% | true |
| hourly | stacking | 1 | 7,339 | 55.59% | 50.00% | 4.771490994537493e-22 | 54.42% | 56.74% | true |

The complete generated Table 7 is retained in `reports/generated/paper_tables/table7_statistical_significance_summary.md`.

## 4.8 Ticker Concentration and Representativeness

Ticker concentration diagnostics show that the global daily and global hourly prediction rows are not dominated by a single ticker by prediction count. However, the original selected hourly confidence slice is concentrated: it contains five tickers, and the top three tickers contribute 79.49% of selected rows. In that selected slice, BID contributes 31.17% of rows, CII contributes 28.73%, and BCM contributes 19.59%.

The V2 sweep adds a more nuanced view. Higher-coverage daily XGBoost h=20 candidates show lower prediction-count concentration than the original selected hourly slice. The best >=50% daily XGBoost h=20 candidate has low concentration, with BWE as the top ticker at 21.68% and the top three tickers at 54.50%. The best >=40% daily XGBoost h=20 candidate is also low-concentration, with BWE at 24.37% and the top three at 58.96%.

This improvement does not establish full VN100 representativeness. The benchmark-usable universe remains seven tickers.

## 4.9 Cost/Slippage Proxy Diagnostics

Cost/slippage proxy diagnostics now exist. The proxy grid crosses transaction cost bps of 5, 10, 15, and 20 with slippage bps of 5, 10, 15, and 20. It includes buy-and-hold, flat/no-trade, always-up, moving-average signal, and previous-direction signal baselines. It reports gross return, net return, turnover, drawdown, profit factor, win rate, trade count, exposure, and equity curve.

The hourly stacking h=1 threshold 0.57 slice is positive at the 10/10 bps diagnostic grid but negative at the 20/20 bps grid in the proxy. This is an important caution: cost assumptions can materially change the interpretation of selected signal slices.

This is not execution-ready because it uses target-return proxies, not real entry/exit prices, liquidity filters, or fill assumptions.

![Appendix Figure B1. Cost/slippage proxy equity curve](reports/generated/evidence_gap_closure/vn100_equity_curve.png)

Caption: Appendix diagnostic proxy, not executable trading backtest.

## 4.10 Evidence Gap Closure Summary

**Table 8. Evidence Gap Closure Status**

Source: `reports/NCKH_EVIDENCE_GAP_CLOSURE_REGISTER.md`

| Original gap | Status | Evidence artifact | Remaining limitation |
| --- | --- | --- | --- |
| Seven evaluated tickers | Not closed | `vn100_cache_coverage_audit.md` | Still only 7 benchmark-usable 2025 tickers. |
| Missing daily confidence sweep rows | Partially closed | `vn100_full_confidence_sweep_summary.csv` | Derived from predictions; official daily sweep remains header-only. |
| Confidence sweep only hourly stacking h=1 | Partially closed | `vn100_full_confidence_sweep_report.md` | Broader derived sweep, not a new official rerun. |
| Selected slice concentration | Partially closed | `vn100_full_confidence_sweep_report.md`; `vn100_ticker_concentration_summary.md` | Original hourly slice remains concentrated. |
| No cost/slippage trading artifacts | Partially closed | `vn100_cost_slippage_validation_report.md` | Diagnostic proxy only, not executable trading backtest. |
| Single 2025 evaluation window | Not closed | `vn100_multiwindow_validation_report.md` | 2022-2024 official artifacts unavailable. |
| No ex-ante regime validation | Partially closed | `vn100_exante_regime_validation_report.md` | Supports 2025 only, not multi-window stability. |

# Chapter 5: Conclusion and Recommendations

## 5.1 Main Findings

The official 2025 VN100 walk-forward benchmark does not pass the global 60% directional-accuracy threshold. Daily overall accuracy is 53.19% over 26,104 predictions, and hourly overall accuracy is 51.29% over 127,944 predictions.

Conditional confidence evidence is stronger after the derived V2 sweep. The original selected hourly stacking h=1 confidence slice reaches 60.03% accuracy with 31.30% coverage, and the derived V2 sweep identifies daily h=20 confidence-filtered candidates above 60% at coverage floors from 20% to 50%.

Regime evidence is stronger for the 2025 window after lagged ex-ante proxy testing. Daily LightGBM h=20 in lagged ex-ante bear regime reaches 66.34% over 309 rows, and daily XGBoost h=20 reaches 65.05% over 309 rows.

These findings remain bounded. Full VN100 representativeness is not established. A stable multi-window 63% method is not established. Trading readiness is not established.

## 5.2 Academic Contribution

The study contributes a reproducible and leakage-aware VN100 benchmark design. It also contributes a claim-governance approach that separates global benchmark results from selected confidence diagnostics, regime diagnostics, statistical evidence, concentration checks, cost/slippage proxy diagnostics, and practical-readiness gaps.

This contribution is useful because it prevents attractive diagnostic slices from being overstated as full-market proof. It also creates a clear roadmap for future experiments.

## 5.3 Practical Implications

The results suggest that conditional directional signal may exist in selected VN100 contexts. However, this implication is research-oriented rather than deployment-oriented. The current artifacts should not be used to justify live trading, portfolio allocation, or investment recommendations.

Before practical use can be considered, selected signal slices must be converted into executable strategies and tested with transaction costs, slippage, turnover, drawdown, profit factor, liquidity assumptions, fill assumptions, and out-of-sample multi-window robustness.

## 5.4 Recommendations for Future Work

Future work should proceed in five steps:

1. Expand benchmark-usable ticker coverage.
2. Rerun the official 2025 benchmark after cache expansion.
3. Generate official 2022-2024 walk-forward artifacts.
4. Validate ex-ante regimes across windows.
5. Convert the cost/slippage proxy into an execution-aware backtest with entry/exit prices, liquidity filters, and fill assumptions.

## 5.5 Final Claim Boundary

The final V2 claim boundary is:

- Global benchmark pass: no.
- Conditional confidence evidence: stronger after derived V2 sweep.
- Regime evidence: stronger for 2025 after lagged ex-ante proxy.
- Full VN100 representativeness: not established.
- Stable multi-window 63% method: not established.
- Trading readiness: not established.
- Profitability: not established.

# References

Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk, 3*(2), 5-39. https://doi.org/10.21314/JOR.2001.041

Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences, 191*, 192-213. https://doi.org/10.1016/j.ins.2011.12.028

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5-32. https://doi.org/10.1023/A:1010933404324

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785-794). https://arxiv.org/abs/1603.02754

Christoffersen, P. F., & Diebold, F. X. (2006). Financial asset returns, direction-of-change forecasting, and volatility dynamics. *Management Science, 52*(8), 1273-1287. https://doi.org/10.1287/mnsc.1060.0520

Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics, 13*, 253-265. https://www.nber.org/papers/t0169

Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3rd ed.). OTexts / Monash University. https://research.monash.edu/en/publications/forecasting-principles-and-practice-3

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems 30*. https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree

Wolpert, D. H. (1992). Stacked generalization. *Neural Networks, 5*(2), 241-259. https://doi.org/10.1016/S0893-6080(05)80023-1

# Appendix A: Artifact Map

| Artifact | Role in V2 manuscript |
| --- | --- |
| `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff` | Official 2025 benchmark artifact family. |
| `reports/generated/paper_tables/` | V1 paper-ready tables reused and selectively summarized in V2. |
| `reports/generated/paper_figures/` | Paper-ready methodology and result figures. |
| `reports/generated/evidence_gap_closure/vn100_cache_coverage_audit.md` | Cache coverage audit and seven-ticker limitation. |
| `reports/generated/evidence_gap_closure/vn100_full_confidence_sweep_report.md` | Derived V2 confidence-sweep expansion. |
| `reports/generated/evidence_gap_closure/vn100_exante_regime_validation_report.md` | Lagged ex-ante regime validation. |
| `reports/generated/evidence_gap_closure/vn100_cost_slippage_validation_report.md` | Cost/slippage proxy diagnostics. |
| `reports/NCKH_RESULTS_CLAIM_REGISTER_V2.md` | V2 claim-governance boundary. |

# Appendix B: Cost/Slippage Proxy Diagnostics

The cost/slippage proxy should be read as a diagnostic appendix, not as a trading backtest. It applies a long/flat signal mapping to selected prediction rows, uses the official target-return proxy divided by horizon, and evaluates cost/slippage grids.

**Appendix Table B1. Model-Signal Proxy Results**

| slice | cost/slippage | net return | max drawdown | profit factor | note |
| --- | --- | --- | --- | --- | --- |
| hourly stacking h=1 threshold 0.57 | 10/10 bps | 12.07% | -11.68% | 1.176 | Positive in proxy. |
| hourly stacking h=1 threshold 0.57 | 20/20 bps | -8.71% | -16.46% | 0.8455 | Negative in proxy. |
| daily LightGBM h=20 post-hoc bear | 10/10 bps | 33.57% | -5.51% | 3.992 | Proxy only. |
| daily XGBoost h=20 post-hoc bear | 10/10 bps | 36.20% | -5.50% | 4.182 | Proxy only. |

![Appendix Figure B1. Cost/slippage proxy equity curve](reports/generated/evidence_gap_closure/vn100_equity_curve.png)

Caption: Appendix diagnostic proxy, not executable trading backtest.

# Appendix C: Evidence Gap Closure Status

| Gap | V2 status | Claim impact |
| --- | --- | --- |
| Seven evaluated tickers | Not closed | Full VN100 representativeness remains unsafe. |
| Missing daily confidence sweep rows | Partially closed | Derived V2 diagnostics can be discussed with caution. |
| Sweep limited to hourly stacking h=1 | Partially closed | Confidence-filtering evidence is broader but still derived. |
| Selected-slice concentration | Partially closed | Some V2 candidates are less concentrated, but seven-ticker limitation remains. |
| No cost/slippage artifacts | Partially closed | Proxy diagnostics exist; trading readiness remains unsafe. |
| Single 2025 window | Not closed | Stable multi-window claims remain unsafe. |
| No ex-ante regime validation | Partially closed | Lagged ex-ante 2025 evidence supports regime discussion only. |
