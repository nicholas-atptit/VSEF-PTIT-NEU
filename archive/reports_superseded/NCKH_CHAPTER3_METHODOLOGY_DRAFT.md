# Chapter 3: Methodology Draft

## 3.1 Data Scope

This study evaluates VN100 stock-direction forecasting using the official artifact family at `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff`. The benchmark is configured for the VN100 universe, classification target mode, and a 2025 held-out evaluation window. The official configuration records a raw daily cache request range from 2006-01-01 to 2015-12-31 and a raw hourly cache request range from 2016-01-01 to 2025-12-31.

The daily benchmark should not be described as using only daily rows from 2006-2015. The manifest states that the daily benchmark method combines daily OHLCV for 2006-2015 with hourly OHLCV from 2016 onward resampled to daily. The official daily benchmark summary records effective training from 2006-01-03 to 2024-12-31 and effective evaluation from 2025-01-02 to 2025-12-31. The hourly benchmark summary records effective training from 2024-01-02 to 2024-12-31 and effective evaluation from 2025-01-02 to 2025-12-31.

The official evaluated tickers are ANV, BCM, BID, BMP, BVH, BWE, and CII. Table 1 in `reports/generated/paper_tables/table1_dataset_evaluation_scope.md` provides the paper-ready dataset and evaluation-scope summary.

## 3.2 Data Coverage and Cache Usability

The official run is a limited-cache VN100 benchmark rather than a definitive full-market VN100 evaluation. The artifact `usable_cache_summary.csv` shows no standalone daily cache rows marked benchmark-usable, because standalone daily rows end before the 2025 evaluation window. The benchmark-usable rows are seven partial hourly pairs, covering ANV, BCM, BID, BMP, BVH, BWE, and CII with actual/effective range from 2024-01-02 to 2025-12-31.

This coverage limitation affects interpretation. The empirical results are useful for testing the repository's walk-forward framework and conditional diagnostics, but they should not be generalized to all VN100 constituents. The paper should explicitly state that cache usability restricts representativeness.

## 3.3 Model Groups and Baselines

The official model set consists of LightGBM, XGBoost, random forest, and stacking. These models are already supported by the benchmark runner and no new model family is introduced in this evidence pack. The benchmark also evaluates simple directional baselines: always-up, previous-direction, seeded random direction, and moving-average signal.

The model and baseline inventory is summarized in Table 2 at `reports/generated/paper_tables/table2_model_baseline_list.md`. The role of the baselines is methodological: they provide directional comparison points so that model accuracy can be interpreted against simple non-ML policies.

## 3.4 Walk-Forward Validation Design

The benchmark uses walk-forward out-of-sample evaluation with a strict training-label cutoff. The official configuration records `train_cutoff = 2024-12-31` and `training_label_cutoff_rule = target_timestamp <= train_cutoff`. This rule is central to the study because it prevents target labels from the 2025 evaluation window from entering training labels.

The official evaluation window is 2025-01-01 to 2025-12-31, with effective evaluation rows beginning on 2025-01-02 in both daily and hourly summaries. Figure 2 in `reports/generated/paper_figures/figure2_walk_forward_design.md` provides the paper-ready schematic of the cutoff and evaluation design.

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

The artifact verification note at `reports/generated/vn100_artifact_date_schema_verification.md` records the relevant date fields and confirms that the raw daily, raw hourly, effective training, and effective evaluation ranges should be described separately. The paper should preserve this distinction to avoid implying that all 2025 information was available during training.

## 3.7 Limitations of the Current Dataset

The current dataset has four material limitations. First, the official evaluated ticker set contains only seven tickers, which weakens full-market representativeness. Second, daily standalone cache rows are not benchmark-usable for the 2025 evaluation window, so the daily benchmark depends on the documented hybrid daily construction. Third, confidence-threshold sweep evidence is partial: the combined sweep artifact covers hourly stacking h=1, while the daily threshold-sweep file contains no data rows. Fourth, the current official artifacts do not include cost-adjusted returns, turnover, drawdown, profit factor, slippage-applied execution, or trade-level PnL for selected diagnostic slices.

These limitations do not invalidate the benchmark as a research framework, but they constrain the claims. The paper should present the results as evidence of leakage-aware VN100 diagnostic forecasting under limited cache coverage, not as proof of deployable trading performance or full-market stability.
