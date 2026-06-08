# Walk-Forward Benchmarking of Ensemble and Baseline Models for Stock Direction Forecasting in Vietnam

## Đánh giá theo phương pháp Walk-Forward các mô hình Ensemble và Baseline trong dự báo xu hướng cổ phiếu tại Việt Nam

## 1. Abstract

This report audits the implemented VN100 walk-forward benchmark system in this repository and converts the current code, tests, and verified artifacts into a research-oriented technical narrative. The official benchmark design is a walk-forward out-of-sample evaluation on VN100 with training labels capped at 2024-12-31 and a held-out evaluation window from 2025-01-01 to 2025-12-31. Under this official configuration, the benchmark produced nonzero predictions at both daily and hourly frequencies, but the official 2025 global benchmark did **not** pass the 60% directional-accuracy gate. The verified official results are daily overall accuracy of 0.5318725099601593 over 26,104 predictions and hourly overall accuracy of 0.5128571875195398 over 127,944 predictions (`outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/daily/benchmark_summary.json`, fields `overall_accuracy`, `n_predictions`; `.../hourly/benchmark_summary.json`, same fields). The official run should be interpreted as an initial VN100 benchmark implementation with limited usable-cache coverage rather than a definitive full-market evaluation.

The same official artifact family also contains two conditional findings that differ materially from benchmark-wide acceptance. First, the selected confidence-sweep candidate is hourly stacking at horizon 1 with threshold 0.57, filtered accuracy 0.6003482803656944, coverage ratio 0.3129854203569969, and 2,297 evaluated rows (`outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/hourly/confidence_threshold_sweep_summary.csv`, columns `threshold`, `coverage_ratio`, `filtered_accuracy`, `evaluated_rows`, `selected_candidate`). Second, the official daily bear-regime diagnostics exceed 0.63 for xgboost h=20 at 0.6914414414414415 and for lightgbm h=20 at 0.6959459459459459 (`outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/daily/regime_accuracy_summary.csv`, columns `regime`, `model`, `horizon`, `n_obs`, `accuracy`).

The narrow, defensible conclusion is therefore: the global benchmark did not pass 60%, but selected confidence-filtered and regime-specific diagnostics showed statistically significant conditional predictive signals. This evidence does not establish coverage-qualified 63% overall performance, benchmark-wide acceptance, or practical trading readiness.

## 2. Introduction

Benchmarking directional stock-forecasting systems is vulnerable to leakage, regime dependence, selection effects, and over-interpretation of attractive slices. These risks are especially important when the research objective is to compare model classes against simple baselines under a reproducible walk-forward design rather than to showcase a single favorable result. Time-series evaluation should preserve chronological order and prevent training on future labels (Bergmeir & Benítez, 2012; Hyndman & Athanasopoulos, 2021).

The repository audited here has evolved into a substantial VN100 benchmark framework with train-cutoff enforcement, baseline comparison, significance testing, confidence filtering, regime-aware diagnostics, and data-governance controls. This report therefore takes a strict research-auditor posture: it states what was implemented, what the verified artifacts show, what they do not show, and which next experiments are required before stronger claims would be justified.

## 3. Research Problem and Objectives

The research problem is to assess whether the implemented ensemble and baseline models demonstrate directional forecasting signal on VN100 under a walk-forward out-of-sample design that respects a strict pre-2025 training cutoff.

The objectives are:

1. To document the benchmark design actually implemented in the repository.
2. To verify whether the official 2025 benchmark passed the configured 60% global directional-accuracy gate.
3. To compare the supported ML models against simple directional baselines.
4. To identify whether confidence filtering or regime partitioning reveals conditional signal that is not visible in the global aggregate.
5. To identify the evidential limits of the current system and define a defensible next research phase.

## 4. Literature and Practical Background

Walk-forward and rolling-origin evaluation are standard approaches for time-dependent forecasting because they preserve chronology and better approximate deployment conditions than shuffled cross-validation (Bergmeir & Benítez, 2012; Hyndman & Athanasopoulos, 2021). The model families implemented in the repository are also well established in the predictive-modeling literature. Random Forests reduce variance through bagging and randomized tree construction (Breiman, 2001). XGBoost is a scalable gradient-boosted tree system optimized for predictive performance and computation (Chen & Guestrin, 2016). LightGBM emphasizes efficient histogram-based boosting (Ke et al., 2017). The broader forecast-comparison literature also motivates formal predictive-accuracy testing beyond raw score differences (Diebold & Mariano, 1995).

Stacking is grounded in Wolpert's original stacked-generalization framework (Wolpert, 1992). Directional-accuracy claims in financial markets should be interpreted against the direction-of-change literature, which distinguishes sign predictability from mean-return predictability (Christoffersen & Diebold, 2006). Practical trading-readiness claims require cost-aware execution evidence because market impact and transaction costs can materially alter realized performance (Almgren & Chriss, 2001).

For market context, the benchmark universe is VN100. Official HOSE ground rules describe VN100 as a constituent set drawn from VN30 and VNMidcap under the HOSE index family (`Ground Rules for Management of the HOSE-Index Series`, Version 4.0).


## 5. Data and Benchmark Design

### Table 1. Benchmark configuration table

| Item | Value | Evidence |
|---|---:|---|
| Universe | VN100 | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/run_config.json`, field `universe` |
| Daily historical range | 2006-01-01 to 2015-12-31 | `run_config.json`, fields `daily_start`, `daily_end` |
| Hourly raw/cache range | 2016-01-01 to 2025-12-31 | `run_config.json`, fields `hourly_start`, `hourly_end` |
| Train cutoff | 2024-12-31 | `run_config.json`, field `train_cutoff` |
| Evaluation window | 2025-01-01 to 2025-12-31 | `run_config.json`, fields `eval_start`, `eval_end` |
| Evaluation type | walk_forward_out_of_sample | official `benchmark_summary.json`, field `evaluation_type` |
| Target mode | classification | `run_config.json`, field `target_mode` |
| Global threshold | 0.60 | `run_config.json`, field `threshold` |
| Regime evaluation | enabled | `run_config.json`, field `enable_regime_evaluation` |
| Confidence filter | enabled | `run_config.json`, field `enable_confidence_filter` |
| Confidence threshold sweep | enabled | `run_config.json`, field `enable_confidence_threshold_sweep` |
| Minimum sweep coverage | 0.30 | `run_config.json`, field `min_sweep_coverage` |
| Cache-only mode | true | `run_config.json`, field `cache_only` |
| Partial cache allowed | true | `run_config.json`, field `allow_partial_cache_for_benchmark` |
| Leakage rule | `target_timestamp <= train_cutoff` | `run_config.json`, field `training_label_cutoff_rule` |
| Post-cutoff actual rows | allowed for evaluation | `run_config.json`, field `actual_rows_allowed_after_train_cutoff` |

### Table 2. Data split table

| Split role | Date range / rule | Interpretation | Evidence |
|---|---|---|---|
| Daily historical inputs | 2006-01-01 to 2015-12-31 | long-span daily training history | `run_config.json` |
| Hourly raw/cache actual data | 2016-01-01 to 2025-12-31 | needed to compute held-out 2025 actuals | `run_config.json` |
| Training-label end | 2024-12-31 | 2025 labels excluded from training | `run_config.json`; `scripts/run_vn100_hybrid_frequency_accuracy_benchmark.py:1244-1261` |
| Official held-out evaluation | 2025-01-01 to 2025-12-31 | walk-forward OOS test period | official `benchmark_summary.json` |
| Extended monitoring | through 2026-05-11 | non-official diagnostics only | `reports/superseded/VN100_HYBRID_BENCHMARK_CLOSEOUT.md:7`, `:37`, `:80` |

Official command structure:

```powershell
python scripts/run_vn100_hybrid_frequency_accuracy_benchmark.py `
  --cache-only `
  --allow-partial-cache-for-benchmark `
  --universe VN100 `
  --daily-start 2006-01-01 `
  --daily-end 2015-12-31 `
  --hourly-start 2016-01-01 `
  --hourly-end 2025-12-31 `
  --train-cutoff 2024-12-31 `
  --eval-start 2025-01-01 `
  --eval-end 2025-12-31 `
  --models lightgbm,xgboost,random_forest,stacking `
  --daily-horizons 1,5,10,20 `
  --hourly-horizons 1,4,8,20 `
  --threshold 0.60 `
  --min-obs-per-group 50 `
  --max-daily-gap-days 30 `
  --coverage-start-tolerance-days 7 `
  --coverage-end-tolerance-days 3 `
  --min-coverage-ratio 0.80 `
  --target-mode classification `
  --enable-regime-evaluation `
  --enable-confidence-filter `
  --confidence-threshold 0.55 `
  --enable-confidence-threshold-sweep `
  --confidence-threshold-grid "0.50,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.60,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.70,0.72,0.74,0.76,0.78,0.80,0.82,0.84,0.86,0.88,0.90" `
  --min-sweep-coverage 0.30 `
  --bootstrap-samples 1000 `
  --bootstrap-seed 42 `
  --output-dir outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff
```

## 6. Methodology

### 6.1 Walk-forward validation

The benchmark uses `walk_forward_out_of_sample` rather than shuffled splits (`outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/daily/benchmark_summary.json`, field `evaluation_type`; hourly equivalent). This is consistent with time-series evaluation guidance (Bergmeir & Benítez, 2012; Hyndman & Athanasopoulos, 2021).

### 6.2 Model groups

The verified ML models are `lightgbm`, `xgboost`, `random_forest`, and `stacking` (official `benchmark_summary.json`, fields `available_models` and `requested_models`).

### 6.3 Baseline models

The benchmark compares against `always_up`, `previous_direction`, `random_seeded_direction`, and `moving_average_signal`, verified from the baseline artifacts.

### 6.4 Directional accuracy

The canonical helper `compute_directional_accuracy_from_returns()` ignores rows with missing or non-finite values and rows where the actual return is exactly zero, then maps positive predicted return to class 1 and non-positive predicted return to class 0 (`src/ml/metrics.py:101-123`).

### 6.5 Confidence filtering

The benchmark can filter predictions by confidence threshold and can also sweep a threshold grid. In the official run, both were enabled (`run_config.json`, fields `enable_confidence_filter`, `enable_confidence_threshold_sweep`, `confidence_threshold_grid`). Accuracy under filtering is paired with coverage.

### 6.6 Regime-aware evaluation

Regime evaluation is enabled in the official run (`run_config.json`, field `enable_regime_evaluation`) and reported in `regime_accuracy_summary.csv`. These are diagnostic slices, not substitutes for global benchmark performance.

### 6.7 Statistical significance

The artifact set reports binomial p-values and bootstrap confidence intervals in `significance_summary.csv`. The benchmark therefore distinguishes raw accuracy from evidence that a slice is above a 50% null.

## 7. Implementation and Reproducibility

### Table 3. Model and baseline table

| Category | Members | Evidence |
|---|---|---|
| ML models | lightgbm, xgboost, random_forest, stacking | official `benchmark_summary.json`, field `available_models` |
| Baselines | always_up, previous_direction, random_seeded_direction, moving_average_signal | official daily/hourly `baseline_delta_summary.csv`, column `baseline` |
| Daily horizons | 1, 5, 10, 20 | `run_config.json`, field `daily_horizons` |
| Hourly horizons | 1, 4, 8, 20 | `run_config.json`, field `hourly_horizons` |

### Table 4. Implemented benchmark components table

| Component | Verified status | Evidence |
|---|---|---|
| Train-cutoff support | Implemented | `scripts/run_vn100_hybrid_frequency_accuracy_benchmark.py:441-442`, `:507-509`, `:1244-1261` |
| 2025 leakage prevention | Implemented and tested | `tests/ml/test_directional_accuracy_metrics.py:210-279` |
| Cache-only mode | Implemented | `run_config.json`, field `cache_only`; CLI in `scripts/...:449-451` |
| Fetch-only / resume-fetch | Implemented | `scripts/...:451-452` |
| Partial benchmark-usable cache mode | Implemented | `run_config.json`, field `allow_partial_cache_for_benchmark`; `scripts/...:1264+` |
| Coverage diagnostics | Implemented | `usable_cache_summary.csv`, columns `benchmark_usable`, `benchmark_usable_reason`, `pre_eval_rows`, `eval_rows` |
| Confidence filtering | Implemented | official `hourly/confidence_filter_summary.csv` |
| Confidence threshold sweep | Implemented | official `hourly/confidence_threshold_sweep_summary.csv` |
| Regime-aware evaluation | Implemented | official `daily/regime_accuracy_summary.csv`; `hourly/benchmark_summary.json`, regime fields |
| Binomial significance + bootstrap CI | Implemented | official daily/hourly `significance_summary.csv` |
| Model-error tracking | Implemented | official `model_error_summary.csv` |
| Source health tracking | Implemented | official `source_health_summary.csv` |
| Fetch visibility | Implemented | official `fetch_summary.csv` |
| Fail-by-default benchmark gate | Implemented | `tests/research/test_vn100_hybrid_60pct_accuracy_gate.py:52-72` |

The gate is strict: the daily research test fails unless `overall_accuracy >= 0.60` and `n_predictions >= 5000` with the correct walk-forward evaluation type (`tests/research/test_vn100_hybrid_60pct_accuracy_gate.py:13-15`, `:64-71`).

## 8. Empirical Results

### 8.1 Official 2025 benchmark results

### Table 5. Official 2025 benchmark result table

| Frequency | n_predictions | overall_accuracy | best_model_accuracy | best_baseline_accuracy | passed | Source |
|---|---:|---:|---:|---:|---|---|
| Daily | 26104 | 0.5318725099601593 | 0.5697236180904522 | 0.5452261306532663 | no | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/daily/benchmark_summary.json` |
| Hourly | 127944 | 0.5128571875195398 | 0.5559340509606213 | 0.5054466230936819 | no | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/hourly/benchmark_summary.json` |

The official 2025 global benchmark therefore did **not** pass the 60% directional-accuracy gate.

### 8.2 Baseline comparison

### Table 6. Baseline comparison table

| Frequency | Model / horizon | Baseline | Model accuracy | Baseline accuracy | Accuracy delta | Source |
|---|---|---|---:|---:|---:|---|
| Daily | xgboost h=20 | always_up | 0.5697236180904522 | 0.5452261306532663 | 0.02449748743718594 | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/daily/baseline_delta_summary.csv` |
| Daily | xgboost h=20 | previous_direction | 0.5697236180904522 | 0.5043969849246231 | 0.0653266331658291 | same file |
| Hourly | stacking h=1 | always_up | 0.5559340509606213 | 0.45305899986374165 | 0.10287505109687967 | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/hourly/baseline_delta_summary.csv` |
| Hourly | stacking h=1 | previous_direction | 0.5559340509606213 | 0.48712358631966207 | 0.06881046464095925 | same file |
| Hourly | xgboost h=1 | random_seeded_direction | 0.5493936503610846 | 0.5002043875187355 | 0.049189262842349035 | same file |

The stronger daily and hourly model slices beat simple baselines, but that outperformance is still insufficient to convert the aggregate benchmark into benchmark-wide acceptance.

### 8.3 Strategy-level confidence-filtered diagnostics

### Table 7. Strategy-level confidence-filtered diagnostic table

| Frequency | Model | Horizon | Threshold | Evaluated rows | Coverage ratio | Filtered accuracy | Passed 60%? | Selected candidate? | Source |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| Hourly | stacking | 1 | 0.57 | 2297 | 0.3129854203569969 | 0.6003482803656944 | yes | True | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/hourly/confidence_threshold_sweep_summary.csv` |
| Hourly | xgboost | 1 | 0.55 | 5744 | 0.7826679384112277 | 0.5631963788300836 | no | n/a | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/hourly/confidence_filter_summary.csv` |
| Hourly | lightgbm | 1 | 0.55 | 5970 | 0.8134623245673798 | 0.5582914572864321 | no | n/a | same file |
| Hourly | random_forest | 1 | 0.55 | 4696 | 0.6398691919880093 | 0.5724020442930153 | no | n/a | same file |
| Daily | xgboost | 20 | 0.55 | 1497 | 0.9403266331658292 | 0.5704742818971276 | no | n/a | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/daily/confidence_filter_summary.csv` |

This is the main official conditional-positive finding. The selected strategy-level slice did pass 60%, but only after filtering reduced coverage to 0.3129854203569969.

No coverage-qualified official confidence-sweep candidate reached 0.63. The same official sweep file shows 0.6294642857142857 at threshold 0.62 and 0.6307420494699647 at threshold 0.63, but coverage there is only 0.0915656083935141 and 0.07712222373620384, with `coverage_ok=False`; those rows are not selected candidates.

### 8.4 Regime-specific diagnostics

### Table 8. Regime-specific diagnostic result table

| Regime | Frequency | Model | Horizon | n_obs | Accuracy | Passed 60%? | Interpretation | Source |
|---|---|---|---:|---:|---:|---|---|---|
| bear | Daily | lightgbm | 20 | 444 | 0.6959459459459459 | yes | Artifact-level best bear-regime result | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/daily/regime_accuracy_summary.csv` |
| bear | Daily | xgboost | 20 | 444 | 0.6914414414414415 | yes | Requested focal bear-regime slice | same file |
| high_volatility | Daily | random_forest | 20 | 622 | 0.6093247588424437 | yes | Secondary daily regime-specific pass | same file |
| high_volatility | Hourly | stacking | 1 | n/a in summary row | 0.5651093439363817 | no | Best hourly regime in official aggregate summary | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/hourly/benchmark_summary.json` |

These 63%+ daily results are regime-specific diagnostics only. They are not global benchmark passes and they are not selected confidence-sweep strategies.

### 8.5 Extended monitoring diagnostics

### Table 9. Extended monitoring diagnostics table

| Artifact set | Frequency | overall_accuracy | n_predictions | Best model/horizon | Best regime diagnostic | Evidence type |
|---|---|---:|---:|---|---|---|
| `outputs/vn100_hybrid_accuracy_benchmark_cache_partial_usable` | Daily | 0.5025831724764392 | 35228 | daily h=20 best model accuracy 0.540590405904059 | not the official benchmark | extended monitoring |
| `outputs/vn100_hybrid_accuracy_benchmark_cache_partial_usable` | Hourly | 0.4956980745977118 | 172016 | hourly h=1 best model accuracy 0.5383215369059656 | not the official benchmark | extended monitoring |
| `outputs/vn100_hybrid_accuracy_benchmark_phase123_full` | Daily | 0.5110707391847394 | 35228 | xgboost h=20, 0.547509225092251 | bear xgboost h=20, 0.6101231190150479 | extended monitoring |
| `outputs/vn100_hybrid_accuracy_benchmark_phase123_full` | Hourly | 0.5175216258952655 | 172016 | stacking h=1, 0.5697674418604651 | high_volatility stacking h=1, 0.5861305791220952 | extended monitoring |
| `outputs/vn100_hybrid_confidence_sweep_smoke` | Hourly strategy slice | 0.6304654442877292 filtered | 709 filtered rows | stacking h=1 at threshold 0.55 | smoke diagnostic only | extended monitoring |
| `outputs/vn100_hybrid_accuracy_benchmark_hourly_h1_tuned` | Hourly tuning run | overall_accuracy 0.5624873609706774 | 19780 | xgboost h=1 unfiltered 0.5677451971688574 | not an official benchmark outcome | extended monitoring |

These artifacts are useful diagnostics, but they are not official 2025 train-cutoff benchmark evidence and should not be merged into the official claim base.

### 8.6 Significance testing

### Table 10. Significance result table

| Frequency | Model | Horizon | n_obs | Accuracy | Binomial p-value | Bootstrap CI | Significant at 5%? | Source |
|---|---|---:|---:|---:|---:|---|---|---|
| Hourly | stacking | 1 | 7339 | 0.5559340509606213 | 4.771490994537493e-22 | [0.5442090203024935, 0.5673831584684562] | yes | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/hourly/significance_summary.csv` |
| Hourly | xgboost | 1 | 7339 | 0.5493936503610846 | 1.3598274636843495e-17 | [0.5382204660035427, 0.5611152745605669] | yes | same file |
| Daily | xgboost | 20 | 1592 | 0.5697236180904522 | 1.4486683327839042e-08 | [0.5452261306532663, 0.5954773869346733] | yes | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/daily/significance_summary.csv` |
| Daily | lightgbm | 20 | 1592 | 0.5596733668341709 | 1.0570499989873873e-06 | [0.5364164572864322, 0.582286432160804] | yes | same file |
| Tuning run hourly | xgboost | 1 | 9890 | 0.5677451971688574 | 9.603101920845109e-42 | [0.5579347826086957, 0.5773559150657229] | yes | `outputs/vn100_hybrid_accuracy_benchmark_hourly_h1_tuned/hourly/significance_summary.csv` |
| Tuning run hourly | lightgbm | 1 | 9890 | 0.5572295247724974 | 2.481151125371174e-30 | [0.5474191102123357, 0.5674418604651162] | yes | same file |

The significance evidence supports the narrower claim that several slices are statistically above a 50% null. It does not support the stronger claim that benchmark-wide acceptance was achieved.

## 9. Claim-status table

### Table 11. Claim-status and forbidden-claim audit table

| Claim | Status | Exact basis |
|---|---|---|
| Global benchmark pass | **no** | Official daily and hourly overall accuracies are 0.5318725099601593 and 0.5128571875195398, both below 0.60 |
| Strategy-level pass | **yes** | Hourly stacking h=1 reached 60.03% with 31.30% coverage under confidence threshold 0.57 |
| Stable 63% method | **no** | No coverage-qualified official confidence-sweep candidate reached 0.63 |
| Regime-specific 63%+ diagnostic | **yes** | Daily bear-regime only: lightgbm h=20 at 0.6959459459459459 and xgboost h=20 at 0.6914414414414415 |
| Practical trading readiness | **not established** | No verified cost-adjusted PnL, slippage, turnover, drawdown, profit factor, concentration, or multi-window robustness evidence |

## 10. Discussion

The repository evidence supports a nuanced interpretation. First, the benchmark framework itself is substantial and real: the codebase implements train-cutoff controls, cache/fetch governance, benchmark-usable cache diagnostics, source-health reporting, baseline comparison, regime labeling, significance testing, and confidence-threshold sweeps. Second, the official 2025 train-cutoff benchmark generates nonzero predictions in both daily and hourly modes, so the project has clearly moved beyond toy-output evidence. Third, the official global benchmark still fails the 60% gate.

The most interesting empirical result is conditional rather than universal. The official 2025 artifact family identifies one selected coverage-qualified confidence-filtered strategy slice—hourly stacking h=1 at threshold 0.57—with filtered accuracy slightly above 60%. Separately, the bear-regime daily h=20 slices show much higher accuracies. This pattern is consistent with the hypothesis that signal may be conditional on participation rules or market state rather than broadly stable across the entire VN100 evaluation surface.

A conditional slice can be statistically interesting without proving a robust, benchmark-wide forecasting capability. That is the key distinction that the report must preserve.

The official run should be interpreted as an initial VN100 benchmark implementation with limited usable-cache coverage rather than a definitive full-market evaluation.

## 11. Limitations

The official run should be interpreted as an initial VN100 benchmark implementation with limited usable-cache coverage rather than a definitive full-market evaluation.

### Table 12. Limitations and mitigation table

| Limitation | Evidence | Why it matters | Mitigation |
|---|---|---|---|
| Global benchmark below 60% | official `benchmark_summary.json` | main research gate not met | re-run after code/data stabilization without lowering threshold |
| Sparse benchmark-usable coverage outside a small usable subset | `usable_cache_summary.csv` shows many `cache_missing` / `cache_partial` / insufficient-row states | limits representativeness of the VN100 universe | improve usable cache coverage and provider completeness |
| Official run uses only seven evaluated tickers | official `benchmark_summary.json`, fields `evaluated_tickers`, `partial_usable_pairs` | weakens generalization to the full VN100 universe; the official run should be interpreted as an initial VN100 benchmark implementation with limited usable-cache coverage rather than a definitive full-market evaluation | expand benchmark-usable ticker coverage |
| Strategy-level pass has 31.30% coverage | official hourly sweep summary | conditional slice may not scale to broader participation | run coverage-constrained sweeps at >=0.50, >=0.40, >=0.30 |
| Regime-specific passes are narrow slices | official `daily/regime_accuracy_summary.csv` | regime-specific signal is not benchmark-wide acceptance | validate across more windows and ex-ante regime rules |
| Transaction costs and slippage are not tested | no official cost-adjusted PnL artifact | direction accuracy may not survive execution frictions | add transaction-cost, slippage, turnover, and drawdown backtests |

## 12. Practical readiness limitations

Practical readiness is **not established**. The current benchmark shows directional-accuracy diagnostics only. It does **not** yet provide:

- transaction-cost-adjusted PnL,
- slippage analysis,
- turnover evidence,
- drawdown evidence,
- profit factor,
- ticker-concentration stability metrics,
- or robustness across multiple walk-forward windows.

Accordingly, the current system must not be described as live-trading-ready.

## 13. Next Research Phase

### Table 13. Next experiment roadmap table

| Priority | Experiment | Objective | Command | Expected outputs | Acceptance criteria | What supports the claim | What weakens/falsifies the claim |
|---|---|---|---|---|---|---|---|
| 1 | Official 2025 rerun after latest code changes | Confirm that the current codebase still reproduces the official artifact family | Use the official command block in Section 5 | Full official artifact set under `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff` | nonzero predictions; zero model errors; same broad pattern or justified differences | reproduces global fail + conditional signal pattern | contradictory results without code/data explanation |
| 2 | Coverage-constrained confidence sweeps | Test whether 63% is reachable with meaningful participation | Re-run official command with `--min-sweep-coverage 0.50`, then `0.40`, then `0.30` | new `confidence_threshold_sweep_summary.csv` files | coverage-qualified candidate at or above target threshold | >=0.63 candidate at meaningful coverage | no >=0.60 candidate once coverage is tightened |
| 3 | Regime-filtered strategy selection | Test whether bear-regime daily h=20 slices are stable | Re-run official command and inspect `daily/regime_accuracy_summary.csv` | regime tables and significance tables | consistent bear-regime accuracy with sufficient n | bear-regime passes recur with significant p-values | bear-regime signal disappears on rerun or in other windows |
| 4 | Expanded tabular classifier search | Test whether broader tabular families improve global or conditional accuracy | **After adding support**: rerun with `--models lightgbm,xgboost,random_forest,stacking,extra_trees,hist_gradient_boosting,logistic_regression,ridge_classifier,linear_svm,catboost` | benchmark summaries, baseline deltas, significance tables | any new model must beat current best on like-for-like evidence | higher global or coverage-qualified conditional accuracy | no improvement over current best models |
| 5 | Ticker concentration diagnostics | Determine whether wins are concentrated in a few names | add/report `number_of_tickers_used`, `top_ticker_share`, min/max ticker count | new diagnostics table or CSV | low concentration and stable participation | signal is broadly distributed | results are dominated by a small subset of tickers |
| 6 | Transaction-cost/slippage backtest | Test economic viability of diagnostic slices | backtest hourly stacking h=1 filtered and daily bear-regime h=20 slices with costs | PnL, turnover, drawdown, profit factor | positive net performance after costs | signal survives costs and slippage | edge vanishes after costs |
| 7 | Multi-window walk-forward robustness | Test temporal stability beyond one focal window | shift evaluation window and repeat the official design | multiple artifact families | similar conclusions across windows | conditional signal survives in several windows | signal is unique to one window |
| 8 | Sequence-model exploration only after leakage tests | Avoid premature complexity | do not add LSTM/BiLSTM yet; defer until sequence-specific leakage controls exist | future model artifacts | leakage controls and reproducible gains | sequence models add robust value after safeguards | complexity rises without reliable improvement |

## 14. Conclusion

The audited repository supports five firm conclusions.

First, the VN100 walk-forward benchmark framework was implemented and executed, with verifiable support for train-cutoff enforcement, cache-usable diagnostics, baseline comparison, significance testing, confidence filtering, regime-aware evaluation, and benchmark gating.

Second, the official 2025 train-cutoff benchmark produced nonzero daily and hourly predictions, but the **global benchmark did not pass** the 60% threshold.

Third, the official artifact family contains a **strategy-level conditional pass**: hourly stacking at horizon 1 reached 0.6003482803656944 under threshold 0.57 with 0.3129854203569969 coverage. This is important evidence of conditional predictive signal, but it is not benchmark-wide acceptance.

Fourth, the official artifact family contains **regime-specific conditional signal**: the requested daily xgboost h=20 bear-regime slice reached 0.6914414414414415, while the artifact-level best bear slice was lightgbm h=20 at 0.6959459459459459. These are regime-specific findings only.

Fifth, the current evidence does **not** support claims of coverage-qualified 63% overall performance, benchmark-wide acceptance, practical trading readiness, or economic profitability.

The defensible research statement is therefore: **the global benchmark did not pass the 60% threshold, but selected confidence-filtered and regime-specific diagnostics showed statistically significant conditional predictive signals.**

Vietnamese summary sentence: **Benchmark tổng thể chưa đạt ngưỡng 60%, nhưng một số chẩn đoán theo confidence filtering và regime-specific cho thấy tín hiệu dự báo có điều kiện và có ý nghĩa thống kê.**

## 15. References

Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk, 3*(2), 5-39. https://doi.org/10.21314/JOR.2001.041

Bergmeir, C., & Benítez, J. M. (2012). *On the use of cross-validation for time series predictor evaluation*. *Information Sciences, 191*, 192-213. https://doi.org/10.1016/j.ins.2011.12.028

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5-32. https://doi.org/10.1023/A:1010933404324

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785-794). https://arxiv.org/abs/1603.02754

Christoffersen, P. F., & Diebold, F. X. (2006). Financial asset returns, direction-of-change forecasting, and volatility dynamics. *Management Science, 52*(8), 1273-1287. https://doi.org/10.1287/mnsc.1060.0520

Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics, 13*, 253-265. Verified via NBER record: https://www.nber.org/papers/t0169

Ho Chi Minh City Stock Exchange. (2024). *Ground Rules for Management of the HOSE-Index Series* (Version 4.0; Decision No. 747/QD-SGDHCM dated 2024-12-30). https://staticfile.hsx.vn/Uploads/LocalFiles/ef15ff11e799483abd11677ad0443887/20250114_20241230_QD%20747%20HOSE%20Index%20Ground%20Rules.pdf

Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts / Monash University. Verified source page: https://research.monash.edu/en/publications/forecasting-principles-and-practice-3

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems 30*. https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree

Wolpert, D. H. (1992). Stacked generalization. *Neural Networks, 5*(2), 241-259. https://doi.org/10.1016/S0893-6080(05)80023-1

## Appendix A. Evidence map

### Table 14. Evidence map table

| Claim | Source file/artifact | JSON field or CSV column | Value | Evidence type |
|---|---|---|---|---|
| Official daily predictions were nonzero | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/daily/benchmark_summary.json` | `n_predictions` | 26104 | official 2025 |
| Official daily overall accuracy | same file | `overall_accuracy` | 0.5318725099601593 | official 2025 |
| Official hourly predictions were nonzero | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/hourly/benchmark_summary.json` | `n_predictions` | 127944 | official 2025 |
| Official hourly overall accuracy | same file | `overall_accuracy` | 0.5128571875195398 | official 2025 |
| Official benchmark pass | both benchmark summaries | `passed` | false / false | official 2025 |
| Official daily best model accuracy | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/daily/benchmark_summary.json` | `best_model_accuracy` | 0.5697236180904522 | official 2025 |
| Official daily best baseline accuracy | same file | `best_baseline_accuracy` | 0.5452261306532663 | official 2025 |
| Official hourly best model accuracy | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/hourly/benchmark_summary.json` | `best_model_accuracy` | 0.5559340509606213 | official 2025 |
| Official hourly best baseline accuracy | same file | `best_baseline_accuracy` | 0.5054466230936819 | official 2025 |
| Daily xgboost h=20 baseline deltas | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/daily/baseline_delta_summary.csv` | `model_accuracy`, `baseline_accuracy`, `accuracy_delta` | always_up: 0.5697236180904522 / 0.5452261306532663 / 0.02449748743718594; previous_direction: 0.5697236180904522 / 0.5043969849246231 / 0.0653266331658291 | official 2025 |
| Hourly stacking h=1 baseline deltas | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/hourly/baseline_delta_summary.csv` | `model_accuracy`, `baseline_accuracy`, `accuracy_delta` | always_up: 0.5559340509606213 / 0.45305899986374165 / 0.10287505109687967; previous_direction: 0.5559340509606213 / 0.48712358631966207 / 0.06881046464095925 | official 2025 |
| Hourly xgboost h=1 random baseline delta | same file | `model_accuracy`, `baseline_accuracy`, `accuracy_delta` | 0.5493936503610846 / 0.5002043875187355 / 0.049189262842349035 | official 2025 |
| Official evaluated ticker coverage | official daily/hourly `benchmark_summary.json` | `evaluated_tickers` | 7 tickers: ANV, BCM, BID, BMP, BVH, BWE, CII | data coverage |
| Official partial usable hourly pairs | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/hourly/benchmark_summary.json` | `partial_usable_pairs` | 7 | data coverage |
| Selected strategy candidate | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/hourly/confidence_threshold_sweep_summary.csv` | `selected_candidate` | True for hourly stacking h=1 at threshold 0.57 | strategy-level diagnostic |
| Strategy filtered accuracy | same file | `filtered_accuracy` | 0.6003482803656944 | strategy-level diagnostic |
| Strategy coverage ratio | same file | `coverage_ratio` | 0.3129854203569969 | strategy-level diagnostic |
| Strategy evaluated rows | same file | `evaluated_rows` | 2297 | strategy-level diagnostic |
| Strategy display claim | same file | `filtered_accuracy`, `coverage_ratio` | 60.03% accuracy, 31.30% coverage | strategy-level diagnostic |
| Hourly h=1 model filter diagnostics | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/hourly/confidence_filter_summary.csv` | `filtered_accuracy`, `coverage_ratio`, `evaluated_rows` | xgboost: 0.5631963788300836 / 0.7826679384112277 / 5744; lightgbm: 0.5582914572864321 / 0.8134623245673798 / 5970; random_forest: 0.5724020442930153 / 0.6398691919880093 / 4696 | strategy-level diagnostic |
| Daily xgboost h=20 confidence filter | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/daily/confidence_filter_summary.csv` | `filtered_accuracy`, `coverage_ratio`, `evaluated_rows` | 0.5704742818971276 / 0.9403266331658292 / 1497 | strategy-level diagnostic |
| Artifact-level best bear regime | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/daily/regime_accuracy_summary.csv` | `accuracy` | 0.6959459459459459 for daily bear lightgbm h=20 | regime-specific diagnostic |
| Requested focal bear regime | same file | `accuracy` | 0.6914414414414415 for daily bear xgboost h=20 | regime-specific diagnostic |
| Requested focal bear regime sample size | same file | `n_obs` | 444 for daily bear xgboost h=20 | regime-specific diagnostic |
| Daily high-volatility random_forest h=20 regime | same file | `accuracy`, `n_obs` | 0.6093247588424437, n=622 | regime-specific diagnostic |
| Hourly high-volatility stacking h=1 regime | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/hourly/regime_accuracy_summary.csv` | `accuracy`, `n_obs` | 0.5651093439363817, n=2012 | regime-specific diagnostic |
| Official hourly stacking h=1 unfiltered accuracy | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/hourly/significance_summary.csv` | `accuracy` | 0.5559340509606213 | official 2025 |
| Official hourly stacking h=1 p-value and CI | same file | `binomial_p_value`, `bootstrap_ci_low`, `bootstrap_ci_high` | 4.771490994537493e-22; [0.5442090203024935, 0.5673831584684562] | official 2025 |
| Official hourly xgboost h=1 p-value and CI | same file | `binomial_p_value`, `bootstrap_ci_low`, `bootstrap_ci_high` | 1.3598274636843495e-17; [0.5382204660035427, 0.5611152745605669] | official 2025 |
| Official daily xgboost h=20 p-value | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/daily/significance_summary.csv` | `binomial_p_value` | 1.4486683327839042e-08 | official 2025 |
| Official daily xgboost h=20 CI | same file | `bootstrap_ci_low`, `bootstrap_ci_high` | [0.5452261306532663, 0.5954773869346733] | official 2025 |
| Official daily lightgbm h=20 p-value and CI | same file | `binomial_p_value`, `bootstrap_ci_low`, `bootstrap_ci_high` | 1.0570499989873873e-06; [0.5364164572864322, 0.582286432160804] | official 2025 |
| Extended daily overall accuracy | `outputs/vn100_hybrid_accuracy_benchmark_phase123_full/daily/benchmark_summary.json` | `overall_accuracy` | 0.5110707391847394 | extended monitoring |
| Extended daily best model accuracy | same file | `best_model_accuracy` | 0.547509225092251 | extended monitoring |
| Extended daily bear xgboost h=20 accuracy | `outputs/vn100_hybrid_accuracy_benchmark_phase123_full/daily/regime_accuracy_summary.csv` | `accuracy` | 0.6101231190150479 | extended monitoring |
| Extended hourly overall accuracy | `outputs/vn100_hybrid_accuracy_benchmark_phase123_full/hourly/benchmark_summary.json` | `overall_accuracy` | 0.5175216258952655 | extended monitoring |
| Extended hourly best model accuracy | same file | `best_model_accuracy` | 0.5697674418604651 | extended monitoring |
| Smoke-sweep filtered accuracy | `outputs/vn100_hybrid_confidence_sweep_smoke/confidence_threshold_sweep_summary.csv` | `filtered_accuracy` | 0.6304654442877292 | extended monitoring |
| Smoke-sweep coverage | same file | `coverage_ratio` | 0.5126536514822849 | extended monitoring |
| Cache-partial daily monitoring accuracy | `outputs/vn100_hybrid_accuracy_benchmark_cache_partial_usable/daily/benchmark_summary.json` | `overall_accuracy` | 0.5025831724764392 | extended monitoring |
| Cache-partial hourly monitoring accuracy | `outputs/vn100_hybrid_accuracy_benchmark_cache_partial_usable/hourly/benchmark_summary.json` | `overall_accuracy` | 0.4956980745977118 | extended monitoring |
| Focused tuning xgboost h=1 filtered accuracy | `outputs/vn100_hybrid_accuracy_benchmark_hourly_h1_tuned/hourly/confidence_filter_summary.csv` | `filtered_accuracy` | 0.5906432748538012 | extended monitoring |
| Focused tuning lightgbm h=1 filtered accuracy | same file | `filtered_accuracy` | 0.5691348195329087 | extended monitoring |
| Model errors in official run | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/model_error_summary.csv` | row count | 0 data rows | official 2025 |
| Official training rule | `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/run_config.json` | `training_label_cutoff_rule` | `target_timestamp <= train_cutoff` | official 2025 |
| Actual rows allowed after cutoff | same file | `actual_rows_allowed_after_train_cutoff` | true | official 2025 |
