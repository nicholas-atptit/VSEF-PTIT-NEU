# Code Cleanup Changes

## Summary

- Added a read-only code inventory script.
- Generated codebase structure inventory reports.
- Moved 37 superseded scripts from `scripts/research` into `scripts/legacy/research` subfolders.
- Rewrote the top-level README to the current VN Market Directional Benchmark Lab identity.
- Added cleanup scope, active code map, reorganization plan, and reorganization result reports.
- Added a follow-up validation hygiene fix for the full-data-backup repository policy and local path redaction.
- Did not change provider behavior, benchmark metrics, label logic, model logic, empirical results, data files, outputs, generated snapshots, or research claims.
- Standardized active VN30 hourly selected-candidate, paper-source, daily result, and index benchmark filenames with `git mv`.

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
- `scripts/check_repo_hygiene.py` - narrowed hygiene policy so approved full-data-backup paths are allowed only when covered by Git LFS attributes, while local path, bytecode/cache, dependency junk, malformed filename, and unapproved generated-root checks remain active.
- `reports/CODE_CLEANUP_VALIDATION_FIX_PLAN.md` - documented validation failures, classification, guardrails, and intended fixes.
- `reports/CODE_CLEANUP_VALIDATION_FIX_RESULT.md` - recorded final hygiene, preflight, provider policy, targeted test, and py_compile results.
- `reports/full_data_push_inventory.csv` - redacted machine-local repository prefixes to `<repo>` while preserving row count, sizes, timestamps, and relative filenames.
- `reports/full_data_push_largest_files.csv` - redacted machine-local repository prefixes to `<repo>` while preserving row count, sizes, timestamps, and relative filenames.
- `outputs/experiments/*` metadata files - redacted machine-local repository prefixes in logs, manifests, and one summary report without changing benchmark metrics.
- `outputs/walkforward_governance_audit*` internal model manifests - redacted machine-local repository prefixes without changing model metrics or result values.
- `docs/REPOSITORY_STRUCTURE.md` - updated current repository layout, protected paths, active evidence, generated artifacts, safe validation scripts, research/benchmark script boundaries, and naming convention.
- `reports/REPO_RENAME_CLEANUP_INVENTORY.md` - added rename inventory and keep/defer decisions.
- `reports/REPO_RENAME_CLEANUP_RESULT.md` - added final rename cleanup status and validation summary.
- `reports/VN30_RESEARCH_CLAIM_REGISTER.md` - added a claim-register index over existing VN30 claim registers and claim boundaries without upgrading claims.
- Protected generated folder rename decisions are documented in `reports/REPO_RENAME_CLEANUP_INVENTORY.md` and `reports/REPO_RENAME_CLEANUP_RESULT.md`; no new generated-folder index was added under `reports/generated/`.

## Files Renamed For Naming Standardization

- `reports/PAPER_FIGURE_DATA_SOURCE_INVENTORY.md` -> `reports/VN30_HOURLY_PAPER_FIGURE_DATA_SOURCE_INVENTORY.md`
- `reports/PAPER_TABLE_FIGURE_CAPTIONS_EN.md` -> `reports/VN30_HOURLY_PAPER_TABLE_FIGURE_CAPTIONS_EN.md`
- `reports/PAPER_TABLE_FIGURE_CAPTIONS_VI.md` -> `reports/VN30_HOURLY_PAPER_TABLE_FIGURE_CAPTIONS_VI.md`
- `reports/PAPER_MISSING_METRICS_TODO.md` -> `reports/VN30_HOURLY_PAPER_MISSING_METRICS_TODO.md`
- `reports/PAPER_ROW_LEVEL_FIGURE_TODO.md` -> `reports/VN30_HOURLY_PAPER_ROW_LEVEL_FIGURE_TODO.md`
- `reports/PAPER_LITERATURE_DATA_TODO.md` -> `reports/VN30_HOURLY_PAPER_LITERATURE_DATA_TODO.md`
- `reports/PAPER_DOCX_MISSING_FOR_FIGURE_INSERTION.md` -> `reports/VN30_HOURLY_PAPER_DOCX_MISSING_FOR_FIGURE_INSERTION.md`
- `reports/PAPER_WITH_FIGURES_LAYOUT_QA.md` -> `reports/VN30_HOURLY_PAPER_WITH_FIGURES_LAYOUT_QA.md`
- `reports/VN30_HOURLY_TARGET62_PAPER_READY_CLAIM_BOUNDARY.md` -> `reports/VN30_HOURLY_SELECTED_CANDIDATE_CLAIM_BOUNDARY.md`
- `reports/VN30_HOURLY_TARGET62_STABILITY_CLAIM_BOUNDARY.md` -> `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_CLAIM_BOUNDARY.md`
- `reports/VN30_HOURLY_TARGET62_STABILITY_ROBUSTNESS_PROTOCOL.md` -> `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_ROBUSTNESS_PROTOCOL.md`
- `reports/VN30_HOURLY_TARGET62_STABILITY_ROBUSTNESS_RESULT.md` -> `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_ROBUSTNESS_RESULT.md`
- `reports/VN30_HOURLY_TARGET62_PAPER_READY_STABILITY_AUDIT_PROTOCOL.md` -> `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_AUDIT_PROTOCOL.md`
- `reports/VN30_HOURLY_TARGET62_PAPER_READY_STABILITY_AUDIT_RESULT.md` -> `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_AUDIT_RESULT.md`
- `reports/VN30_DAILY_2015_BENCHMARK_RESULT_SUMMARY.md` -> `reports/VN30_DAILY_2015_RESULT_SUMMARY.md`
- `reports/INDEX_DIRECTIONAL_BENCHMARK_RESULT_SUMMARY.md` -> `reports/VN30_INDEX_BENCHMARK_RESULT_SUMMARY.md`
- `reports/INDEX_DIRECTIONAL_BENCHMARK_CLAIM_REGISTER.md` -> `reports/VN30_INDEX_BENCHMARK_CLAIM_REGISTER.md`
- `scripts/research/rerun_vn30_hourly_selected_l2_logistic_h40_row_predictions.py` -> `scripts/research/rerun_vn30_hourly_selected_candidate_row_predictions.py`
- `scripts/research/build_paper_empirical_tables.py` -> `scripts/research/build_vn30_hourly_paper_empirical_tables.py`
- `scripts/research/build_paper_empirical_figures.py` -> `scripts/research/build_vn30_hourly_paper_empirical_figures.py`
- `scripts/research/audit_vn30_hourly_target62_paper_ready_stability.py` -> `scripts/research/audit_vn30_hourly_selected_candidate_stability_summary.py`
- `scripts/research/audit_vn30_hourly_target62_stability_robustness.py` -> `scripts/research/audit_vn30_hourly_selected_candidate_stability_robustness.py`

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
