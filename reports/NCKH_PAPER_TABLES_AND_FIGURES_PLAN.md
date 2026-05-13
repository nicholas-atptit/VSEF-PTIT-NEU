# NCKH Paper Tables and Figures Plan

## Tables

| Item | Source artifact | Columns or fields needed | Target chapter | Claim supported | Current status |
|---|---|---|---|---|---|
| Table 1: Dataset and evaluation scope | `run_config.json`, `manifest.json`, `usable_cache_summary.csv`, `daily/benchmark_summary.json`, `hourly/benchmark_summary.json` | `daily_start`, `daily_end`, `hourly_start`, `hourly_end`, `train_cutoff`, `eval_start`, `eval_end`, `effective_training_range`, `effective_evaluation_range`, `evaluated_tickers`, `benchmark_usable`, `actual_start`, `actual_end` | Chapter 3 | The official run is a 2025 held-out VN100 benchmark with limited usable-cache coverage. | ready |
| Table 2: Model and baseline list | `run_config.json`, `daily/baseline_summary.csv`, `hourly/baseline_summary.csv` | `models`, `target_mode`, baseline names, horizons, frequency | Chapter 3 | The study compares LightGBM, XGBoost, random forest, stacking, and simple directional baselines. | ready |
| Table 3: Global benchmark results | `daily/benchmark_summary.json`, `hourly/benchmark_summary.json`, `daily/classification_accuracy_summary.csv`, `hourly/classification_accuracy_summary.csv` | `overall_accuracy`, `n_predictions`, `best_model_accuracy`, `best_model_frequency`, `best_model_horizon`, `passed`, model/horizon accuracy rows | Chapter 4 | The official benchmark produced nonzero predictions but did not pass the global 60% threshold. | ready |
| Table 4: Baseline delta summary | `daily/baseline_delta_summary.csv`, `hourly/baseline_delta_summary.csv` | `frequency`, `model`, `horizon`, `baseline`, `model_accuracy`, `baseline_accuracy`, `accuracy_delta`, `model_better_than_baseline` | Chapter 4 | Some model/horizon slices outperform simple baselines, but that is not a global pass. | ready |
| Table 5: Confidence-filtered strategy diagnostics | `confidence_threshold_sweep_summary.csv`, `hourly/confidence_threshold_sweep_summary.csv`, `daily/confidence_filter_summary.csv`, `hourly/confidence_filter_summary.csv`, `reports/generated/vn100_confidence_coverage_review.md` | `threshold`, `total_rows`, `evaluated_rows`, `coverage_ratio`, `filtered_accuracy`, `passed_60pct`, `coverage_ok`, `selected_candidate` | Chapter 4 | The selected hourly stacking h=1 threshold 0.57 slice passes 60% only at about 31.30% coverage. | partial |
| Table 6: Regime-specific diagnostics | `daily/regime_accuracy_summary.csv`, `hourly/regime_accuracy_summary.csv`, `reports/generated/vn100_ticker_concentration_summary.csv` | `frequency`, `regime`, `model`, `horizon`, `n_obs`, `accuracy`, `passed_60pct`, `reliable`, ticker concentration fields | Chapter 4 | Bear-regime daily h=20 diagnostics are strong but remain regime-specific and coverage-limited. | ready |
| Table 7: Statistical significance summary | `daily/significance_summary.csv`, `hourly/significance_summary.csv`, `daily/mcnemar_summary.csv`, `hourly/mcnemar_summary.csv` | `n_obs`, `accuracy`, `null_accuracy`, `binomial_p_value`, `bootstrap_ci_low`, `bootstrap_ci_high`, `significant_at_5pct`, McNemar fields | Chapter 4 | Several slices are above a 50% null, but significance does not prove trading readiness. | partial |
| Table 8: Robustness and limitation matrix | `usable_cache_summary.csv`, `model_error_summary.csv`, `source_health_summary.csv`, `reports/generated/vn100_artifact_date_schema_verification.md`, `reports/generated/vn100_ticker_concentration_summary.md`, `reports/generated/vn100_confidence_coverage_review.md`, `reports/generated/vn100_cost_slippage_readiness_review.md` | coverage counts, skipped reasons, model errors, artifact date/schema caveats, concentration assessment, missing cost/slippage fields, readiness gaps | Chapter 4 and Chapter 5 | The main limitations are limited ticker coverage, selected-slice concentration, single-window evidence, and missing trading-cost validation. | ready |

## Figures

| Item | Source artifact | Columns or fields needed | Target chapter | Claim supported | Current status |
|---|---|---|---|---|---|
| Figure 1: Research pipeline | `scripts/run_vn100_hybrid_frequency_accuracy_benchmark.py`, `run_config.json`, `manifest.json` | cache-only setting, provider, train cutoff, model list, output files, evaluation stages | Chapter 3 | The benchmark pipeline separates data loading, training-label cutoff, walk-forward evaluation, diagnostics, and reporting. | partial |
| Figure 2: Walk-forward validation design | `run_config.json`, `manifest.json`, benchmark summaries | `train_cutoff`, `training_label_cutoff_rule`, `effective_training_range`, `effective_evaluation_range`, `eval_start`, `eval_end` | Chapter 3 | 2025 outcomes are evaluated out of sample after a 2024-12-31 training-label cutoff. | ready |
| Figure 3: Accuracy by model/horizon | `daily/accuracy_summary.csv`, `hourly/accuracy_summary.csv`, `daily/classification_accuracy_summary.csv`, `hourly/classification_accuracy_summary.csv` | `frequency`, `model`, `horizon`, `ticker`, `n_obs`, `accuracy`, `reliable` | Chapter 4 | Accuracy varies by model, horizon, frequency, and ticker. | ready |
| Figure 4: Confidence threshold vs coverage/accuracy | `confidence_threshold_sweep_summary.csv`, `hourly/confidence_threshold_sweep_summary.csv`, `reports/generated/vn100_confidence_coverage_review.md` | `threshold`, `coverage_ratio`, `filtered_accuracy`, `passed_60pct`, `selected_candidate` | Chapter 4 | Accuracy rises to the selected 60.03% slice only as coverage falls to about 31.30%. | partial |
| Figure 5: Regime-specific accuracy | `daily/regime_accuracy_summary.csv`, `hourly/regime_accuracy_summary.csv` | `frequency`, `regime`, `model`, `horizon`, `n_obs`, `accuracy`, `reliable`, `passed_60pct` | Chapter 4 | Regime diagnostics are conditional and should not be presented as global benchmark performance. | ready |

## Status Notes

- `ready` means the needed source artifact fields exist for the current official run.
- `partial` means the source evidence exists only for part of the desired comparison, or the table/figure still needs manual rendering from available rows.
- `missing` would mean the current artifacts cannot support the table or figure without a new experiment.

## Generated Artifact Pack

- Tables 1-8 are generated as CSV and Markdown under `reports/generated/paper_tables/`.
- Figures 1-2 are generated as Markdown schematics under `reports/generated/paper_figures/`.
- Figures 3-5 are generated as PNG charts under `reports/generated/paper_figures/`.
- Artifact-pack status notes are generated under `reports/generated/paper_notes/`.
- Generator: `scripts/research/build_vn100_paper_artifact_pack.py`.
