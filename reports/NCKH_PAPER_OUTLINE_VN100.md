# NCKH Paper Outline - VN100 Walk-Forward Benchmark

## Chapter 1: Introduction

### 1.1 Research Context

Write about the need for transparent stock-direction forecasting evaluation in Vietnam. Emphasize that forecasting claims must be tested with chronological splits and held-out outcomes.

Supporting artifacts:
- `reports/WALK_FORWARD_BENCHMARKING_ENSEMBLE_BASELINE_MODELS_VN_STOCK_MARKET.md`
- `reports/VN100_HYBRID_BENCHMARK_CLOSEOUT.md`

Safe claims:
- The repository implements a leakage-aware VN100 benchmark.
- The official benchmark uses 2025 as a held-out evaluation window.

Unsafe claims:
- Do not claim practical trading readiness.
- Do not claim a global 60% benchmark pass.

### 1.2 Problem Statement

Explain that the main research problem is whether machine learning and ensemble models show directional forecasting value under a strict walk-forward design.

Supporting artifacts:
- Official benchmark summaries under `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff`.
- `tests/research/test_vn100_hybrid_60pct_accuracy_gate.py`.

Safe claims:
- The official benchmark did not pass 60%.
- Conditional diagnostics show signal that requires further validation.

### 1.3 Research Objectives and Questions

Use the objectives and research questions from `reports/NCKH_RESEARCH_DESIGN_VN100.md`.

Supporting artifacts:
- Research design document.
- Claim-status table in the benchmark research report.

### 1.4 Scope and Limitations

State that the official run currently evaluated only seven usable tickers: ANV, BCM, BID, BMP, BVH, BWE, CII.

Supporting artifacts:
- Official `benchmark_summary.json`, field `evaluated_tickers`.
- Evidence map in the benchmark research report.

Safe claims:
- This is an initial VN100 benchmark implementation with limited usable-cache coverage.

Unsafe claims:
- Do not present the result as a definitive full-market VN100 evaluation.

## Chapter 2: Literature Review and Theoretical Background

### 2.1 Time-Series and Walk-Forward Validation

Discuss why chronological evaluation is required for time-dependent stock forecasting.

Supporting references:
- Bergmeir and Benitez (2012).
- Hyndman and Athanasopoulos (2021).

Supporting artifacts:
- `run_config.json`, fields `eval_start`, `eval_end`, `train_cutoff`.
- Benchmark metadata field `evaluation_type`.

### 2.2 Machine Learning Models

Discuss random forests, XGBoost, LightGBM, and stacking.

Supporting references:
- Breiman (2001).
- Chen and Guestrin (2016).
- Ke et al. (2017).
- Wolpert (1992).

Supporting artifacts:
- Official `benchmark_summary.json`, fields `available_models`, `requested_models`.

### 2.3 Directional Accuracy and Forecast Evaluation

Explain directional accuracy, a 50% null, binomial tests, and bootstrap confidence intervals.

Supporting references:
- Christoffersen and Diebold (2006).
- Diebold and Mariano (1995).

Supporting artifacts:
- `daily/significance_summary.csv`.
- `hourly/significance_summary.csv`.
- `src/ml/metrics.py`.
- `tests/ml/test_directional_accuracy_metrics.py`.

### 2.4 Trading Practicality and Transaction Costs

Explain that directional accuracy alone is not enough for trading readiness.

Supporting reference:
- Almgren and Chriss (2001).

Supporting artifacts:
- Missing practical-readiness diagnostics listed in `reports/NCKH_EXPERIMENT_INVENTORY.md`.

Safe claims:
- Practical trading readiness is not established.

Unsafe claims:
- Do not claim profitability or deployment readiness.

## Chapter 3: Data and Methodology

### 3.1 Data Scope

Describe VN100, raw daily cache request range, raw hourly cache request range, hybrid daily benchmark construction, train cutoff, and official 2025 evaluation.

Supporting artifacts:
- `run_config.json`.
- `manifest.json`.
- `usable_cache_summary.csv`.
- Official benchmark research report evidence map.

### 3.2 Data Coverage and Cache Usability

Document the limited usable-cache coverage and seven evaluated tickers.

Supporting artifacts:
- `usable_cache_summary.csv`.
- `benchmark_summary.json`, fields `evaluated_tickers`, `partial_usable_pairs`, `full_range_valid_pairs`.

Safe claims:
- Coverage limits representativeness.

Unsafe claims:
- Do not generalize to the full VN100 market yet.

### 3.3 Model Groups and Baselines

Define the model groups and baseline strategies.

Supporting artifacts:
- `accuracy_summary.csv`.
- `baseline_summary.csv`.
- `baseline_delta_summary.csv`.

### 3.4 Walk-Forward Validation Design

Explain train-cutoff enforcement and held-out 2025 labels.

Supporting artifacts:
- `run_config.json`, field `training_label_cutoff_rule`.
- `tests/ml/test_directional_accuracy_metrics.py` train-cutoff tests.

### 3.5 Evaluation Metrics

Define overall accuracy, per-horizon accuracy, baseline delta, confidence-filtered accuracy, regime-specific accuracy, p-values, and bootstrap confidence intervals.

Supporting artifacts:
- `classification_accuracy_summary.csv`.
- `confidence_filter_summary.csv`.
- `confidence_threshold_sweep_summary.csv`.
- `regime_accuracy_summary.csv`.
- `significance_summary.csv`.

## Chapter 4: Empirical Results and Discussion

### 4.1 Official Global Benchmark Results

Report official daily and hourly results:
- Daily accuracy: 0.5318725099601593 over 26,104 predictions.
- Hourly accuracy: 0.5128571875195398 over 127,944 predictions.
- Global benchmark pass: no.

Supporting artifacts:
- Daily and hourly `benchmark_summary.json`.

Safe claims:
- Official benchmark produced nonzero predictions.
- Official benchmark did not pass 60%.

### 4.2 Baseline Comparison

Report model improvements over simple baselines for the stronger slices.

Supporting artifacts:
- `baseline_delta_summary.csv`.

Safe claims:
- Some model/horizon slices beat simple baselines.

Unsafe claims:
- Baseline outperformance is not a global pass.

### 4.3 Confidence-Filtered Strategy Diagnostics

Report the selected hourly stacking h=1 diagnostic:
- Threshold: 0.57.
- Accuracy: 0.6003482803656944.
- Coverage: 0.3129854203569969.
- Evaluated rows: 2,297.

Supporting artifacts:
- `hourly/confidence_threshold_sweep_summary.csv`.
- `strategy_selection_summary.csv` when generated by the current runner.

Safe claims:
- Strategy-level diagnostic pass at 60.03% with 31.30% coverage.

Unsafe claims:
- Do not call it a global benchmark pass.
- Do not claim a stable 63% method.

### 4.4 Regime-Specific Diagnostics

Report daily bear-regime h=20 results:
- LightGBM h=20: 0.6959459459459459.
- XGBoost h=20: 0.6914414414414415.
- n_obs: 444 for the focal xgboost bear-regime slice.

Supporting artifacts:
- `daily/regime_accuracy_summary.csv`.

Safe claims:
- Bear-regime diagnostic signal exists in the official artifact.

Unsafe claims:
- Do not treat this as full-market performance.
- Do not treat this as selected confidence-sweep performance.

### 4.5 Statistical Significance

Report p-values and bootstrap confidence intervals for unfiltered model/horizon rows.

Supporting artifacts:
- `significance_summary.csv`.

Safe claims:
- Several slices are statistically above a 50% null.

Unsafe claims:
- Statistical significance alone does not prove trading readiness.

### 4.6 Discussion of Data and Methodological Limits

Discuss the seven-ticker coverage issue, selected-slice risk, regime-slice risk, and missing trading-cost evidence.

Supporting artifacts:
- `usable_cache_summary.csv`.
- `model_error_summary.csv`.
- `source_health_summary.csv`.
- `reports/generated/vn100_ticker_concentration_summary.md`.
- `reports/generated/vn100_confidence_coverage_review.md`.
- `reports/generated/vn100_cost_slippage_readiness_review.md`.

## Chapter 5: Conclusion and Recommendations

### 5.1 Main Findings

Summarize:
- Global benchmark pass: no.
- Strategy-level pass: yes, hourly stacking h=1 at 60.03% and 31.30% coverage.
- Stable 63% method: no.
- Regime-specific 63%+ diagnostic: yes, bear-regime only.
- Practical trading readiness: not established.

### 5.2 Academic Contribution

State that the work contributes a reproducible, leakage-aware VN100 benchmark and a disciplined claim boundary.

### 5.3 Practical Implications

Explain that conditional signal may guide future research, but not deployment.

### 5.4 Recommendations

Recommend:
- Expand usable cache coverage.
- Add multi-window validation.
- Add ticker concentration diagnostics.
- Add transaction-cost and slippage backtests.
- Generate paper-ready tables and charts from artifacts.

### 5.5 Final Claim Boundary

Safe final claim:

The official 2025 VN100 walk-forward benchmark did not pass the global 60% directional-accuracy threshold, but confidence-filtered and bear-regime diagnostics show conditional predictive signal that merits further validation.

Unsafe final claims:
- The model is ready for live trading.
- The benchmark proves a stable 63% method.
- The current evidence is a definitive full-market VN100 evaluation.
