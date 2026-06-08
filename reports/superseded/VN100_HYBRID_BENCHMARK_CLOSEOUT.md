# VN100 Hybrid Benchmark Closeout

## 1. Executive Summary

The official VN100 hybrid-frequency benchmark uses training labels/features through 2024-12-31 and walk-forward out-of-sample evaluation from 2025-01-01 to 2025-12-31. The model must not train on 2025 data before predicting 2025 targets; 2025 rows are used only as held-out actuals for evaluation.

The cache-only partial-usable benchmark execution succeeded and produced nonzero daily and hourly predictions. The benchmark did not meet the 60% directional accuracy threshold, so this is a completed research run, not a passing benchmark. Values in this closeout that reference data through 2026-05-11 are extended monitoring diagnostics only and are not the official 2006-2024 training / 2025 evaluation benchmark.

## 2. Benchmark Configuration

- Universe: VN100
- Official historical/training label cutoff: 2024-12-31
- Daily data range: 2006-01-01 to 2015-12-31
- Hourly raw/cache actual data range: 2016-01-01 to 2025-12-31
- Out-of-sample evaluation range: 2025-01-01 to 2025-12-31
- Evaluation method: walk-forward out-of-sample
- Training rule: do not train on 2025 data before predicting 2025 targets
- Models: lightgbm, xgboost, random_forest, stacking
- Daily horizons: 1, 5, 10, 20
- Hourly horizons: 1, 4, 8, 20
- Directional accuracy threshold: 0.60
- Minimum observations per group: 50

Official date arguments:

```powershell
python scripts/run_vn100_hybrid_frequency_accuracy_benchmark.py `
  --daily-start 2006-01-01 `
  --daily-end 2015-12-31 `
  --hourly-start 2016-01-01 `
  --hourly-end 2025-12-31 `
  --train-cutoff 2024-12-31 `
  --eval-start 2025-01-01 `
  --eval-end 2025-12-31
```

Raw/cache actual data may extend to 2025-12-31 so 2025 held-out labels and actuals can be computed. Training labels/features are cut off at 2024-12-31. The date 2026-05-11 is reserved for extended monitoring runs only and should not be used in the official 2006-2024 training / 2025 evaluation benchmark.

## 3. Data Availability

Data availability remains a key limitation. Strict full-range cache validation found only a small number of fully valid daily cache pairs, while many provider/cache responses were partial or unavailable. The completed run used benchmark-usable partial hourly caches when they had enough pre-evaluation training rows and enough evaluation-window rows.

## 4. Cache-only Partial-usable Mode

The cache-only partial-usable run used cached data only. It did not call the provider, lower thresholds, or synthesize missing history. Partial cache inputs were eligible only when they covered the evaluation window within tolerance, started before evaluation, and met the configured minimum pre-evaluation and evaluation row counts.

## 5. Extended Monitoring Model Results

The following recorded results came from the extended monitoring run that included data through 2026-05-11. They are useful diagnostics, but they are not the official 2025-only out-of-sample benchmark result.

- Daily predictions: 35,228
- Hourly predictions: 172,016
- Daily overall accuracy: 0.5026
- Hourly overall accuracy: 0.4957
- Best daily model: xgboost 0.512660
- Best hourly model: xgboost 0.503814
- Model errors: 0

## 6. Threshold Result

The 60% accuracy threshold was not met in the recorded run. No benchmark pass is claimed. Official acceptance must be based on the 2006-2024 training / 2025 out-of-sample evaluation period above.

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

## 9. Phase 1-4 Diagnostic Extension

The Phase 1-4 diagnostics below were produced from extended monitoring artifacts that included data through 2026-05-11. They should be treated as diagnostic evidence only until rerun on the official 2025 out-of-sample evaluation window.

### Phase 1–3 Full Diagnostic Result

Daily:
- Overall accuracy: 0.5111
- Predictions: 35,228
- Best model/horizon: xgboost h=20, accuracy 0.547509
- Best confidence-filtered result: xgboost h=20, filtered accuracy 0.553181, coverage 0.949723
- Best regime: bear, xgboost h=20, accuracy 0.610123, n=731
- Overall 60% threshold: not met

Hourly:
- Overall accuracy: 0.5175
- Predictions: 172,016
- Best model/horizon: stacking h=1, accuracy 0.569767
- Best confidence-filtered result: stacking h=1, filtered accuracy 0.599915, coverage 0.477654

This result is close to 0.60, but it remains below the threshold and is not counted as a benchmark pass.

- Best regime: high_volatility, stacking h=1, accuracy 0.586131, n=2711
- Overall 60% threshold: not met

### Phase 4 Focused Tuning

- Backend: grid
- Requested trials: 20
- Actual candidates per model/frequency: 5
- xgboost h=1 accuracy: 0.567745
- lightgbm h=1 accuracy: 0.557230
- xgboost filtered accuracy: 0.590643, coverage 0.691608
- lightgbm filtered accuracy: 0.569135, coverage 0.761982
- xgboost p-value: 9.60e-42
- lightgbm p-value: 2.48e-30
- Filtered 60% threshold: not met

### Conclusion

Diagnostics reveal meaningful signal, especially hourly h=1 and daily bear-regime h=20. However, the benchmark still does not pass the 60% overall threshold. No pass is claimed.

## 10. Official 2025 Train-cutoff Diagnostic Result

### Configuration

- train_cutoff: 2024-12-31
- data_end: 2025-12-31
- eval_start: 2025-01-01
- eval_end: 2025-12-31
- training_label_cutoff_rule: target_timestamp <= train_cutoff
- actual_rows_allowed_after_train_cutoff: true

### Main Benchmark

- Daily n_predictions: 26,104
- Daily overall accuracy: 0.5318725099601593
- Hourly n_predictions: 127,944
- Hourly overall accuracy: 0.5128571875195398
- Global 60% benchmark passed: no
- Model errors: 0

### Strategy-level Diagnostic Pass

- Frequency/model/horizon: hourly stacking h=1
- Confidence threshold: 0.57
- Filtered accuracy: 0.6003482803656944
- Coverage ratio: 0.3129854203569969
- Evaluated rows: 2,297
- p-value: 2.98e-22
- Interpretation: strategy-level diagnostic pass, not global benchmark pass

### 63% Check

- No covered confidence-sweep candidate reached 0.63.
- Daily xgboost h=20 bear-regime diagnostic slice reached 0.6914414414414415 with n=444 and p-value 2.31e-16.
- Interpretation: regime-specific diagnostic finding, not global benchmark pass and not confidence-sweep selected strategy.

### Leakage Confirmation

- 2025 actual rows were used only for held-out evaluation.
- Training labels were capped at 2024-12-31.
- Synthetic test verifies 2025 predictions can be generated while training labels remain capped at 2024-12-31.
