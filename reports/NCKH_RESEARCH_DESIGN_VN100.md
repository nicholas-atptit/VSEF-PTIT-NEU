# NCKH Research Design - VN100 Walk-Forward Stock Direction Forecasting

## Proposed Vietnamese Research Title

Đánh giá thực nghiệm các mô hình học máy và ensemble trong dự báo xu hướng cổ phiếu VN100 theo phương pháp walk-forward ngoài mẫu.

## Proposed English Research Title

Walk-Forward Evaluation of Machine Learning and Ensemble Models for VN100 Stock Direction Forecasting.

## Research Background

Vietnamese equity forecasting research often reports model accuracy without a strict separation between training data and future evaluation labels. The current repository now contains a VN100 hybrid-frequency benchmark that enforces a 2024-12-31 training-label cutoff and evaluates 2025 held-out outcomes through walk-forward out-of-sample artifacts. This provides a starting point for a formal NCKH study, but the evidence must be framed carefully because the official run did not pass the global 60% threshold and used limited benchmark-usable cache coverage.

## Research Gap

The current gap is not simply model selection. The stronger gap is methodological: a defensible VN100 study needs transparent data coverage, leakage-safe validation, baseline comparison, conditional strategy diagnostics, regime analysis, and clear separation between benchmark-wide performance and selected diagnostic slices. Existing repository evidence shows useful conditional signal, but not yet a definitive full-market VN100 conclusion.

## Research Objectives

1. Build a reproducible VN100 walk-forward benchmark with explicit train/evaluation separation.
2. Compare machine learning models with baseline directional strategies.
3. Measure whether confidence filtering or regime segmentation reveals conditional predictive signal.
4. Document data-coverage limitations and prevent overclaiming.
5. Define the experiments needed before making practical trading-readiness claims.

## Research Questions

1. Do supported VN100 models exceed the official 60% global directional-accuracy benchmark under a 2025 held-out evaluation window?
2. Do ensemble methods, especially stacking, outperform simple directional baselines?
3. Does confidence filtering produce a coverage-qualified strategy-level diagnostic pass?
4. Are predictive signals concentrated in specific market regimes such as bear regimes or high volatility?
5. Does current usable-cache coverage support a full-market VN100 conclusion?

## Research Hypotheses

- H1: The global VN100 benchmark accuracy of the implemented models does not exceed the 60% acceptance threshold under the official 2025 train-cutoff evaluation.
- H2: Some model/horizon slices outperform simple directional baselines even when the global benchmark fails.
- H3: Confidence filtering can identify a strategy-level diagnostic slice above 60% accuracy, but the coverage requirement materially limits the claim.
- H4: Regime-specific slices can exceed 63% accuracy, but these findings are diagnostic and not benchmark-wide passes.
- H5: Limited usable-cache coverage weakens generalization to the full VN100 universe.

## Dataset Scope

- Universe: VN100.
- Raw daily cache request range: 2006-01-01 to 2015-12-31; official metadata records effective raw daily data starting on 2006-01-03.
- Raw hourly cache request range: 2016-01-01 to 2025-12-31; the benchmark-usable official cache rows are the seven partial hourly pairs with actual/effective range 2024-01-02 to 2025-12-31.
- Official daily benchmark method: daily OHLCV for 2006-2015 plus hourly OHLCV from 2016 onward resampled to daily. The standalone daily cache rows in `usable_cache_summary.csv` are not benchmark-usable because they end before the 2025 evaluation window.
- Training-label cutoff: 2024-12-31.
- Official evaluation window: 2025-01-01 to 2025-12-31.
- Current official evaluated tickers: ANV, BCM, BID, BMP, BVH, BWE, CII.
- Interpretation: initial VN100 benchmark implementation with limited usable-cache coverage, not a definitive full-market evaluation.

## Model Groups

- Machine learning classifiers/regressors used by the VN100 benchmark runner: LightGBM, XGBoost, random forest, and stacking.
- Baselines: always-up, previous-direction, seeded random direction, and moving-average signal.
- Conditional diagnostic layers: confidence-threshold filtering and regime-aware segmentation.
- Deferred model families: sequence models and additional tabular classifiers should be added only after leakage and coverage checks are stable.

## Walk-Forward Validation Design

The official design uses walk-forward out-of-sample evaluation with raw/cache data available through 2025 actuals while training labels are capped at 2024-12-31. The benchmark metadata records `training_label_cutoff_rule = target_timestamp <= train_cutoff`, which prevents 2025 labels from entering model training. Predictions are evaluated only against held-out 2025 target outcomes.

## Evaluation Metrics

- Overall directional accuracy.
- Per-frequency and per-horizon directional accuracy.
- Baseline delta by model, horizon, and baseline.
- Confidence-filtered accuracy.
- Coverage ratio after filtering.
- Regime-specific accuracy.
- Binomial p-values against a 50% null.
- Bootstrap confidence intervals.
- Model error count.
- Benchmark pass/fail status against the 60% global threshold.

## Robustness Checks

- Re-run official 2025 benchmark from the same command and compare artifact fields.
- Repeat confidence sweeps with stricter coverage thresholds such as 50%, 40%, and 30%.
- Validate regime-specific findings across additional evaluation windows.
- Use ticker concentration diagnostics to avoid small-subset overinterpretation; the current selected hourly confidence slice is concentrated in five tickers, with the top three tickers contributing most selected rows.
- Add transaction-cost, slippage, turnover, drawdown, and profit-factor tests before practical readiness claims.
- Expand usable cache coverage before presenting a full-market VN100 conclusion.

## Academic Contribution

The expected contribution is a leakage-safe VN100 benchmarking framework and an evidence-backed distinction between global benchmark performance, strategy-level diagnostics, regime-specific diagnostics, and extended monitoring. This is valuable because it prevents treating attractive conditional slices as full-market proof.

## Limitations

- The official benchmark did not pass the global 60% threshold.
- The official run currently evaluates only seven usable tickers.
- The strategy-level pass has limited and concentrated coverage: hourly stacking h=1 reaches 60.03% with 31.30% coverage, and selected rows are concentrated in five tickers.
- The 63%+ results are regime-specific diagnostics, not stable benchmark-wide methods.
- Practical trading readiness is not established because cost-adjusted PnL, slippage, turnover, and drawdown are not yet verified.

## Next-Step Research Roadmap

1. Freeze the official benchmark command and artifact schema.
2. Expand VN100 cache coverage and rerun the official 2025 benchmark.
3. Add ticker concentration and per-ticker contribution diagnostics.
4. Run coverage-constrained confidence sweeps.
5. Validate bear-regime findings across additional windows.
6. Add cost/slippage-aware strategy backtests for selected diagnostic slices.
7. Prepare paper tables from benchmark summaries, baseline deltas, confidence sweeps, regime summaries, and significance summaries.
8. Only after these steps, evaluate new model families or sequence models.
