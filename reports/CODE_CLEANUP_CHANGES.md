# Code Cleanup Changes

## Summary

- Added a read-only code inventory script.
- Generated codebase structure inventory reports.
- Moved 37 superseded scripts from `scripts/research` into `scripts/legacy/research` subfolders.
- Rewrote the top-level README to the current VN Market Directional Benchmark Lab identity.
- Added cleanup scope, active code map, reorganization plan, and reorganization result reports.
- Did not change provider behavior, benchmark metrics, label logic, model logic, empirical results, data files, outputs, generated snapshots, or research claims.

## Files Added or Updated

- `README.md` - replaced old VSEF decision-diagnostic front page with a concise benchmark-lab README, provider/API boundary, active evidence tracks, LFS-backed data/artifact note, and validation commands.
- `scripts/research/audit_codebase_structure.py` - added read-only inventory script for Python file counts, duplicate module names, provider-import usage, local absolute paths, issue markers, large files, entrypoints, and likely active/legacy files.
- `reports/CODE_CLEANUP_SCOPE.md` - added cleanup scope and protected-path guardrails.
- `reports/ACTIVE_CODE_MAP.md` - added active provider, benchmark, research, legacy, and protected artifact map.
- `reports/SCRIPT_REORGANIZATION_PLAN.md` - added script classification and move plan.
- `reports/SCRIPT_REORGANIZATION_RESULT.md` - added move results and import-risk review.
- `reports/CODE_CLEANUP_CHANGES.md` - this change log.
- `reports/generated/code_cleanup/codebase_structure_inventory.csv` - generated inventory table.
- `reports/generated/code_cleanup/codebase_structure_inventory.md` - generated inventory summary.

## Files Moved to `scripts/legacy/research/failed_experiments/`

- `scripts/research/audit_vn30_hourly_2015_above60_optimization.py` -> `scripts/legacy/research/failed_experiments/audit_vn30_hourly_2015_above60_optimization.py`
- `scripts/research/audit_vn30_hourly_2015_all_60pct_candidates.py` -> `scripts/legacy/research/failed_experiments/audit_vn30_hourly_2015_all_60pct_candidates.py`
- `scripts/research/audit_vn30_hourly_2015_all_model_final65_router.py` -> `scripts/legacy/research/failed_experiments/audit_vn30_hourly_2015_all_model_final65_router.py`
- `scripts/research/audit_vn30_hourly_2015_final65_focus_v3.py` -> `scripts/legacy/research/failed_experiments/audit_vn30_hourly_2015_final65_focus_v3.py`
- `scripts/research/audit_vn30_hourly_2015_full_tuning_sweep.py` -> `scripts/legacy/research/failed_experiments/audit_vn30_hourly_2015_full_tuning_sweep.py`
- `scripts/research/audit_vn30_hourly_2015_hard_optimization_v2.py` -> `scripts/legacy/research/failed_experiments/audit_vn30_hourly_2015_hard_optimization_v2.py`
- `scripts/research/audit_vn30_hourly_2015_horizon_relative_target_results.py` -> `scripts/legacy/research/failed_experiments/audit_vn30_hourly_2015_horizon_relative_target_results.py`
- `scripts/research/audit_vn30_hourly_2015_jan2025_benchmark.py` -> `scripts/legacy/research/failed_experiments/audit_vn30_hourly_2015_jan2025_benchmark.py`
- `scripts/research/audit_vn30_hourly_2015_overall_directional_final65.py` -> `scripts/legacy/research/failed_experiments/audit_vn30_hourly_2015_overall_directional_final65.py`
- `scripts/research/audit_vn30_hourly_2015_overall_directional_final65_v2.py` -> `scripts/legacy/research/failed_experiments/audit_vn30_hourly_2015_overall_directional_final65_v2.py`
- `scripts/research/audit_vn30_hourly_2015_rf_h60_final65_focus.py` -> `scripts/legacy/research/failed_experiments/audit_vn30_hourly_2015_rf_h60_final65_focus.py`
- `scripts/research/audit_vn30_hourly_2015_rf_h60_final65_router_v2.py` -> `scripts/legacy/research/failed_experiments/audit_vn30_hourly_2015_rf_h60_final65_router_v2.py`
- `scripts/research/audit_vn30_hourly_2015_target60_65_results.py` -> `scripts/legacy/research/failed_experiments/audit_vn30_hourly_2015_target60_65_results.py`
- `scripts/research/audit_vn30_hourly_2015_target_redesign_results.py` -> `scripts/legacy/research/failed_experiments/audit_vn30_hourly_2015_target_redesign_results.py`
- `scripts/research/audit_vn30_hourly_2015_validation_final_mismatch.py` -> `scripts/legacy/research/failed_experiments/audit_vn30_hourly_2015_validation_final_mismatch.py`
- `scripts/research/audit_vn30_hourly_rf_h60_result_consistency.py` -> `scripts/legacy/research/failed_experiments/audit_vn30_hourly_rf_h60_result_consistency.py`
- `scripts/research/run_vn30_hourly_2015_above60_experiments.py` -> `scripts/legacy/research/failed_experiments/run_vn30_hourly_2015_above60_experiments.py`
- `scripts/research/run_vn30_hourly_2015_above60_optimization.py` -> `scripts/legacy/research/failed_experiments/run_vn30_hourly_2015_above60_optimization.py`
- `scripts/research/run_vn30_hourly_2015_all_model_final65_router.py` -> `scripts/legacy/research/failed_experiments/run_vn30_hourly_2015_all_model_final65_router.py`
- `scripts/research/run_vn30_hourly_2015_confidence_sweep.py` -> `scripts/legacy/research/failed_experiments/run_vn30_hourly_2015_confidence_sweep.py`
- `scripts/research/run_vn30_hourly_2015_cost_slippage_proxy.py` -> `scripts/legacy/research/failed_experiments/run_vn30_hourly_2015_cost_slippage_proxy.py`
- `scripts/research/run_vn30_hourly_2015_final65_focus_v3.py` -> `scripts/legacy/research/failed_experiments/run_vn30_hourly_2015_final65_focus_v3.py`
- `scripts/research/run_vn30_hourly_2015_full_tuning_sweep.py` -> `scripts/legacy/research/failed_experiments/run_vn30_hourly_2015_full_tuning_sweep.py`
- `scripts/research/run_vn30_hourly_2015_hard_optimization_v2.py` -> `scripts/legacy/research/failed_experiments/run_vn30_hourly_2015_hard_optimization_v2.py`
- `scripts/research/run_vn30_hourly_2015_horizon_relative_target_experiments.py` -> `scripts/legacy/research/failed_experiments/run_vn30_hourly_2015_horizon_relative_target_experiments.py`
- `scripts/research/run_vn30_hourly_2015_jan2025_benchmark.py` -> `scripts/legacy/research/failed_experiments/run_vn30_hourly_2015_jan2025_benchmark.py`
- `scripts/research/run_vn30_hourly_2015_overall_directional_final65.py` -> `scripts/legacy/research/failed_experiments/run_vn30_hourly_2015_overall_directional_final65.py`
- `scripts/research/run_vn30_hourly_2015_overall_directional_final65_v2.py` -> `scripts/legacy/research/failed_experiments/run_vn30_hourly_2015_overall_directional_final65_v2.py`
- `scripts/research/run_vn30_hourly_2015_regime_diagnostics.py` -> `scripts/legacy/research/failed_experiments/run_vn30_hourly_2015_regime_diagnostics.py`
- `scripts/research/run_vn30_hourly_2015_rf_h60_final65_focus.py` -> `scripts/legacy/research/failed_experiments/run_vn30_hourly_2015_rf_h60_final65_focus.py`
- `scripts/research/run_vn30_hourly_2015_rf_h60_final65_router_v2.py` -> `scripts/legacy/research/failed_experiments/run_vn30_hourly_2015_rf_h60_final65_router_v2.py`
- `scripts/research/run_vn30_hourly_2015_significance_diagnostics.py` -> `scripts/legacy/research/failed_experiments/run_vn30_hourly_2015_significance_diagnostics.py`
- `scripts/research/run_vn30_hourly_2015_target60_baseline_v2.py` -> `scripts/legacy/research/failed_experiments/run_vn30_hourly_2015_target60_baseline_v2.py`
- `scripts/research/run_vn30_hourly_2015_target65_final_v2.py` -> `scripts/legacy/research/failed_experiments/run_vn30_hourly_2015_target65_final_v2.py`
- `scripts/research/run_vn30_hourly_2015_target_redesign_experiments.py` -> `scripts/legacy/research/failed_experiments/run_vn30_hourly_2015_target_redesign_experiments.py`

## Other Files Moved

- `scripts/research/build_vn30_hourly_available_window_paper_artifact_pack.py` -> `scripts/legacy/research/paper_builders/build_vn30_hourly_available_window_paper_artifact_pack.py`
- `scripts/research/validate_external_vn30_hourly_dataset.py` -> `scripts/legacy/research/old_hourly_2005_2026/validate_external_vn30_hourly_dataset.py`

## Tracked Junk Review

- Tracked `__pycache__` or compiled Python junk found: none.

## Protected Paths

No cleanup edits were made under:

- `data/`
- `outputs/`
- `archive/generated_data_snapshots/`
- `reports/generated/` except the new `reports/generated/code_cleanup/` inventory folder
- `archive/reports_superseded/`
- `src/data/providers/`
- `src/data/adapters/`
- `tests/data/`
- `tests/ml/`
