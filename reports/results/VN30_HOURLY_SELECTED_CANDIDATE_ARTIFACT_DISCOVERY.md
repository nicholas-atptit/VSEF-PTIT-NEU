# VN30 Hourly Selected Candidate Artifact Discovery

## Scope

This discovery covers the fixed Track A canonical-like VN30 stock-only hourly selected candidate:

- Model: L2 Logistic (`l2_logistic`).
- Horizon: h=40.
- Feature set: `feature_set_C_closest`.
- Threshold: 0.50.
- Existing final accuracy: 61.51%.
- Existing majority baseline: 50.44%.
- Existing majority baseline lift: +11.07 percentage points.

No market data was fetched, no broad benchmark sweep was run, no candidate was changed, and no DOCX or paper file was edited for this discovery.

## Artifact Files Found

Selected-candidate aggregate artifacts:

- `outputs/vn30_hourly_track_a_target62_validation_safe/selected_candidate_summary.csv`
- `outputs/vn30_hourly_track_a_target62_validation_safe/final_candidate_results.csv`
- `outputs/vn30_hourly_track_a_target62_validation_safe/validation_candidate_results.csv`
- `outputs/vn30_hourly_track_a_target62_validation_safe/run_config.json`
- `outputs/vn30_hourly_track_a_target62_validation_safe/target62_run_log.md`
- `reports/generated/vn30_hourly_track_a_target62_validation_safe/selected_candidate_summary.csv`
- `reports/generated/vn30_hourly_track_a_target62_validation_safe/final_candidate_results.csv`
- `reports/generated/vn30_hourly_track_a_target62_validation_safe/validation_candidate_results.csv`
- `reports/generated/vn30_hourly_track_a_target62_validation_safe/run_config.json`
- `reports/generated/vn30_hourly_track_a_target62_validation_safe/target62_run_log.md`

Existing selected-candidate audit artifacts:

- `reports/generated/vn30_hourly_track_a_target62_validation_safe/target62_audit.csv`
- `reports/generated/vn30_hourly_track_a_target62_validation_safe/target62_audit.md`
- `reports/generated/vn30_hourly_track_a_target62_validation_safe/target62_by_ticker.csv`
- `reports/generated/vn30_hourly_track_a_target62_validation_safe/target62_by_time.csv`
- `reports/generated/vn30_hourly_track_a_target62_validation_safe/target62_by_regime.csv`
- `reports/generated/vn30_hourly_track_a_target62_validation_safe/target62_validation_mismatch.csv`
- `reports/generated/vn30_hourly_target62_paper_ready_stability/global_summary.csv`
- `reports/generated/vn30_hourly_target62_paper_ready_stability/paper_ready_summary_table.csv`
- `reports/generated/vn30_hourly_target62_paper_ready_stability/paper_ready_stability_audit.md`

Related prediction artifacts found but not usable as selected-candidate row-level evidence:

- `outputs/vn30_hourly_track_a_all_algorithm_stacking/base_final_predictions.csv`
- `outputs/vn30_hourly_track_a_all_algorithm_stacking/base_oof_predictions.csv`
- `reports/generated/vn30_hourly_track_a_true_stacking_all_algorithms/base_final_predictions.csv`
- `reports/generated/vn30_hourly_track_a_true_stacking_all_algorithms/base_oof_predictions.csv`

These are stacking/all-algorithm artifacts, not the fixed selected L2 Logistic h=40 target62 validation-safe row-level prediction artifact requested here.

## Selected-Candidate Config Source

The primary selected-candidate configuration source is:

- `outputs/vn30_hourly_track_a_target62_validation_safe/run_config.json`

The config records:

- Track: Track A canonical-like.
- Horizons searched in the historical run: 40, 60, 80.
- Thresholds searched in the historical run: 0.45, 0.50, 0.55.
- Model order including `l2_logistic`.
- Feature-set order including `feature_set_C_closest`.
- Validation-only selection rule.
- `confidence_abstention`: false.
- `ticker_subset`: false.
- `topk`: false.
- `data_fetch`: false.
- `paper_docx`: false.

The exact selected candidate is recorded in:

- `outputs/vn30_hourly_track_a_target62_validation_safe/selected_candidate_summary.csv`
- `reports/generated/vn30_hourly_track_a_target62_validation_safe/selected_candidate_summary.csv`

The selected row is `l2_logistic`, h=40, `feature_set_C_closest`, threshold 0.50.

## Result Source

The primary result source is:

- `outputs/vn30_hourly_track_a_target62_validation_safe/selected_candidate_summary.csv`

The selected row records:

- Validation accuracy: 51.88145188145188%.
- Final accuracy: 61.51202749140894%.
- Final rows: 4,074.
- Final majority baseline: 50.44182621502209%.
- Final lift over majority baseline: +11.070201276386848 percentage points.
- Delta versus 60.43 Logistic h40 baseline: +1.080019636720675 percentage points.
- Delta versus 60.31 RF h60 historical reference: +1.2020274914089413 percentage points.

The validation-final mismatch source is:

- `reports/generated/vn30_hourly_track_a_target62_validation_safe/target62_validation_mismatch.csv`

It records validation-final gap +9.63057560995706 percentage points and `high_positive_final_gap`.

## Row-Level Predictions Status

Selected-candidate row-level predictions do not already exist in the target62 validation-safe output directories.

The existing paper-ready stability audit explicitly marks row-level rolling 250/500/1000 windows as unavailable because row-level predictions were not saved. Separate stacking prediction files exist, but they are not the selected fixed L2 Logistic h=40 target62 validation-safe row-level predictions.

Therefore, a fixed-candidate rerun is required only to save row-level final predictions.

## Feature Set Definition Status

`feature_set_C_closest` is found in code and config.

Definition sources:

- `scripts/research/vn30_hourly_dual_track_common.py`, `build_feature_set_c`.
- `scripts/research/run_vn30_hourly_track_a_target62_validation_safe.py`, `build_feature_sets`.
- `outputs/vn30_hourly_track_a_target62_validation_safe/run_config.json`, `feature_manifests.feature_set_C_closest`.

The run config records `feature_set_C_closest` with 99 features, leakage-safe flags, no future-return features, no future-regime features, no final-label-derived features, and no final-period manual filters.

## Exact Rerun Feasibility

Exact fixed-candidate rerun is possible from existing local code and cache:

- Stock cache: `data/market_cache/vnstock_data/vn30/hourly_2015`.
- Index cache: `data/market_cache/vnstock_data/indices/hourly_2015`.
- Ticker universe: `read_joint_panel_universe` via `active_stock_tickers`.
- Feature builder: `build_feature_set_c`.
- Label builder: `add_absolute_labels`.
- Split constants: train through 2023-12-31 23:59:59, validation 2024-01-01 through 2024-12-31 23:59:59, final from 2025-01-01.
- Model factory: `candidate_model("l2_logistic")`, L2 Logistic with `solver="liblinear"`, `C=0.3`, `class_weight="balanced"`, `random_state=42`.
- Threshold: 0.50.

The existing selected-candidate audit also contains a `replay_selected` path that reconstructs the selected model and final predictions for slice summaries, confirming that the data-side rerun can be performed without new model selection.

## Missing Dependency Or Missing Artifact

Missing artifact:

- Selected-candidate row-level final predictions were not saved before this task.

No missing local dependency was identified during artifact discovery. Dependency availability is still checked by compiling and running the fixed-candidate rerun script in the intended virtual environment.
