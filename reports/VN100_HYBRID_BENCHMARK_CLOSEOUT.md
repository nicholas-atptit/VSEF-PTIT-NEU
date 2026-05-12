# VN100 Hybrid Benchmark Closeout

## 1. Executive Summary

The VN100 hybrid-frequency benchmark execution succeeded in cache-only partial-usable mode and produced nonzero daily and hourly predictions. The benchmark did not meet the 60% directional accuracy threshold, so this is a completed research run, not a passing benchmark.

## 2. Benchmark Configuration

- Universe: VN100
- Daily raw request range: 2006-01-01 to 2015-12-31
- Hourly raw request range: 2016-01-01 to 2026-05-11
- Evaluation range: 2025-01-01 to 2026-05-11
- Models: lightgbm, xgboost, random_forest, stacking
- Daily horizons: 1, 5, 10, 20
- Hourly horizons: 1, 4, 8, 20
- Directional accuracy threshold: 0.60
- Minimum observations per group: 50

## 3. Data Availability

Data availability remains a key limitation. Strict full-range cache validation found only a small number of fully valid daily cache pairs, while many provider/cache responses were partial or unavailable. The completed run used benchmark-usable partial hourly caches when they had enough pre-evaluation training rows and enough evaluation-window rows.

## 4. Cache-only Partial-usable Mode

The cache-only partial-usable run used cached data only. It did not call the provider, lower thresholds, or synthesize missing history. Partial cache inputs were eligible only when they covered the evaluation window within tolerance, started before evaluation, and met the configured minimum pre-evaluation and evaluation row counts.

## 5. Model Results

- Daily predictions: 35,228
- Hourly predictions: 172,016
- Daily overall accuracy: 0.5026
- Hourly overall accuracy: 0.4957
- Best daily model: xgboost 0.512660
- Best hourly model: xgboost 0.503814
- Model errors: 0

## 6. Threshold Result

The 60% accuracy threshold was not met. No benchmark pass is claimed.

## 7. Limitations

- Partial provider/cache coverage remains the main blocker.
- The usable cache set is small relative to the VN100 universe.
- Model results are below the 60% research gate despite successful execution.
- Cache-only mode validates reproducibility from local data, but it cannot recover missing provider history.

## 8. Next Phase Recommendations

- Improve provider coverage and source fallback diagnostics before model changes.
- Expand usable cache coverage for the evaluation window.
- Keep the 60% threshold unchanged for the research gate.
- Re-run the same benchmark configuration after improving data availability.
- Investigate feature and model changes only after the data coverage bottleneck is reduced.
