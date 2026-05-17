# VN30 Daily 2015 Target60 Failure Postmortem

- Created at UTC: `2026-05-17T10:30:35+00:00`.
- Inputs: `outputs/vn30_daily_2015_benchmark`, `outputs/vn30_daily_2015_target60_optimization`, `outputs/vn30_daily_2015_target60_v2`, `data/market_cache/vnstock_data/vn30/daily_2015`, `configs/universes/vn30_constituents_frozen.csv`.
- Scope: daily-only; no hourly data; no daily-to-hourly resampling; no data fetch.
- Selection boundary: saved candidates and thresholds were selected from validation artifacts; final labels are used here only for postmortem scoring diagnostics.
- Reproduced candidates: existing h=40 v1 final-best and h=50 v2 final-best only; no new tuning sweep.

## Executive Answer

- Canonical best daily result: LightGBM `daily_cross` h=40, final accuracy 57.58%, final rows 8,880.
- Did v2 improve over v1: no (-0.13pp versus h=40).
- Target60 passed: no; h=40 gap to 60 is 2.42pp.
- h=40 remains the canonical best recorded daily candidate: yes.
- h=50 `volatility_normalized`: final accuracy 57.45%, validation mean 52.86%, final-minus-validation +4.59pp.
- Interpretation of h=50: it did not underperform its validation mean on final; it underperformed the h=40 record and remained below 60%.
- Daily v3 recommended: no. V2 did not improve over the h=40 record, the h=40 model is only 1.33pp above the final-period majority-class diagnostic baseline, and validation rankings do not provide a clear final-period fix.

## Why V2 Failed To Improve

- V2 broadened horizons, thresholds, and feature sets, but its best final result was 57.45%, below the 57.58% h=40 record.
- V2's validation-stability-selected candidate was not the final-best candidate, which indicates validation-final ranking mismatch rather than a clear new signal.
- The h=40 diagnostic model is only 1.33pp above the final-period majority-class baseline (56.25%), so the exploitable daily signal above a simple class baseline is small.
- Errors are concentrated by ticker and final-period regime; the allowed scope does not permit fixing this by ticker exclusion, universe change, hourly data, or final-label selection.

## Best Candidate Comparison

| source | selection_basis | model | horizon | feature_set | decision_threshold | validation_metric | stability_score | final_accuracy | gap_to_60_pp | delta_vs_canonical_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| benchmark | best_saved_final_accuracy | xgboost | 1 | daily_basic |  | 55.51% |  | 55.84% | 4.16pp | -1.74pp |
| v1 | best_saved_validation_accuracy | lightgbm | 1 | daily_extended |  | 55.21% |  | 55.13% | 4.87pp | -2.44pp |
| v1 | best_saved_final_accuracy | lightgbm | 40 | daily_cross |  | 52.95% |  | 57.58% | 2.42pp | +0.00pp |
| v2 | best_saved_stability_score | lightgbm | 40 | daily_cross | 0.5 | 52.16% | 52.16% | 53.92% | 6.08pp | -3.66pp |
| v2 | best_saved_rolling_validation_mean | lightgbm | 40 | volatility_normalized | 0.45 | 53.32% | 51.94% | 53.99% | 6.01pp | -3.59pp |
| v2 | best_saved_final_accuracy | lightgbm | 50 | volatility_normalized | 0.525 | 52.86% | 51.19% | 57.45% | 2.55pp | -0.13pp |

## Ticker Drag

Worst h=40 ticker diagnostics:

| ticker | n_rows | n_errors | accuracy | positive_rate | majority_baseline_accuracy | delta_to_candidate_accuracy_pp |
| --- | --- | --- | --- | --- | --- | --- |
| VIC | 296 | 190 | 35.81% | 90.20% | 90.20% | -21.77pp |
| VJC | 296 | 185 | 37.50% | 59.46% | 59.46% | -20.08pp |
| VIB | 296 | 177 | 40.20% | 38.85% | 61.15% | -17.38pp |
| ACB | 296 | 174 | 41.22% | 44.93% | 55.07% | -16.36pp |
| BVH | 296 | 174 | 41.22% | 63.85% | 63.85% | -16.36pp |
| VPB | 296 | 157 | 46.96% | 49.32% | 50.68% | -10.62pp |
| GAS | 296 | 156 | 47.30% | 53.38% | 53.38% | -10.28pp |
| SSB | 296 | 151 | 48.99% | 34.12% | 65.88% | -8.59pp |

## Time Drag

Worst h=40 monthly diagnostics:

| month_id | n_rows | n_errors | accuracy | positive_rate | majority_baseline_accuracy | delta_to_candidate_accuracy_pp |
| --- | --- | --- | --- | --- | --- | --- |
| 2025-09 | 600 | 349 | 41.83% | 26.83% | 73.17% | -15.75pp |
| 2025-10 | 690 | 401 | 41.88% | 38.99% | 61.01% | -15.69pp |
| 2025-11 | 600 | 322 | 46.33% | 75.17% | 75.17% | -11.25pp |
| 2025-03 | 630 | 337 | 46.51% | 25.87% | 74.13% | -11.07pp |
| 2026-01 | 600 | 291 | 51.50% | 14.17% | 85.83% | -6.08pp |
| 2025-02 | 600 | 283 | 52.83% | 27.17% | 72.83% | -4.75pp |

Worst h=40 quarterly diagnostics:

| quarter_id | n_rows | n_errors | accuracy | positive_rate | majority_baseline_accuracy | delta_to_candidate_accuracy_pp |
| --- | --- | --- | --- | --- | --- | --- |
| 2025-Q4 | 1980 | 1043 | 47.32% | 59.09% | 59.09% | -10.26pp |
| 2025-Q1 | 1740 | 793 | 54.43% | 42.53% | 57.47% | -3.15pp |
| 2026-Q1 | 1410 | 583 | 58.65% | 26.60% | 73.40% | +1.07pp |
| 2025-Q3 | 1920 | 792 | 58.75% | 54.32% | 54.32% | +1.17pp |

## Class Imbalance

- h=40 final positive rate: 56.25%.
- h=40 final majority-class diagnostic baseline: 56.25%.
- h=40 model lift over final majority baseline: +1.33pp.
- Class imbalance is not the only problem because weak months include both positive-light and positive-heavy regimes, but the majority baseline is close enough to constrain the daily claim.

## Baseline Comparison

| candidate_id | model_accuracy | naive_50_baseline_accuracy | final_majority_class_baseline_accuracy | model_minus_naive_50_pp | model_minus_final_majority_pp | gap_to_60_pp |
| --- | --- | --- | --- | --- | --- | --- |
| canonical_v1_lightgbm_h40_daily_cross_nl20_d3_lr0.02_n700_t500 | 57.58% | 50.00% | 56.25% | +7.58pp | +1.33pp | 2.42pp |
| v2_best_final_lightgbm_h50_volatility_normalized_nl20_d3_lr0.02_n700_t525 | 57.45% | 50.00% | 54.07% | +7.45pp | +3.38pp | 2.55pp |

- Against the fixed 50% baseline, h=40 is +7.58pp higher.
- Against the final majority-class diagnostic baseline, h=40 is only +1.33pp higher.

## Validation-Final Mismatch

| source | validation_metric_name | n_candidates | pearson_corr_validation_final | spearman_corr_validation_final | validation_best_final_accuracy | final_best_final_accuracy | selected_final_gap_to_final_best_pp | final_best_validation_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | validation_accuracy | 105 | 0.668 | 0.701 | 55.13% | 57.58% | 2.44pp | 22 |
| v2 | rolling_validation_mean_accuracy | 100 | 0.177 | 0.210 | 53.99% | 57.45% | 3.46pp | 12 |
| v2 | stability_score | 100 | -0.116 | -0.132 | 53.92% | 57.45% | 3.53pp | 44 |

Interpretation:

- V1 validation-best selection trailed the v1 final-best result by 2.44pp.
- V2 stability-best selection trailed the v2 final-best result by 3.53pp.
- This means final-period top results should be treated as postmortem evidence, not as proof that validation can reliably select a stronger daily candidate.

## Daily V3 Decision

- Daily v3 is not justified now.
- A broad v3 tuning sweep would be weakly motivated because v2 already widened the search without improving the h=40 record.
- A narrow v3 would require a pre-registered validation-only fix for class/regime calibration or validation-final ranking; the current diagnostics do not identify one with enough specificity.

## Current Daily Claim Boundary

- Daily target60 failed under the frozen 30/30 VN30 daily universe.
- The current daily benchmark boundary is best recorded final accuracy 57.58% for LightGBM `daily_cross` h=40.
- The daily result may be used only as robustness context for separate hourly available-window evidence.
- No trading-readiness claim.
- No profitability claim.
- No live-deployment claim.
- No hourly claim made from daily data.
- No paper or DOCX generated.

## Generated Files

- `reports/generated/vn30_daily_2015_target60_postmortem/daily_best_candidate_comparison.csv`
- `reports/generated/vn30_daily_2015_target60_postmortem/daily_ticker_drag.csv`
- `reports/generated/vn30_daily_2015_target60_postmortem/daily_time_drag.csv`
- `reports/generated/vn30_daily_2015_target60_postmortem/daily_class_balance.csv`
- `reports/generated/vn30_daily_2015_target60_postmortem/daily_baseline_comparison.csv`
- `reports/generated/vn30_daily_2015_target60_postmortem/daily_validation_final_mismatch.csv`
