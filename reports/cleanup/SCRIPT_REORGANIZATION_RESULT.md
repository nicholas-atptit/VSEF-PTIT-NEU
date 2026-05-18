# Script Reorganization Result

## Summary

- Scripts moved count: 37.
- Scripts kept active count: 51.
- Diagnostics kept count: 14.
- Manual review count: 13.
- Import risks found after move: none from `src`, `tests`, or remaining `scripts/research` references to moved module names.
- Benchmarks run: no.
- Data fetch run: no.
- Model training run: no.
- Paper/DOCX generated: no.

## Moved Scripts

Moved to `scripts/legacy/research/failed_experiments/`:

- `audit_vn30_hourly_2015_above60_optimization.py`
- `audit_vn30_hourly_2015_all_60pct_candidates.py`
- `audit_vn30_hourly_2015_all_model_final65_router.py`
- `audit_vn30_hourly_2015_final65_focus_v3.py`
- `audit_vn30_hourly_2015_full_tuning_sweep.py`
- `audit_vn30_hourly_2015_hard_optimization_v2.py`
- `audit_vn30_hourly_2015_horizon_relative_target_results.py`
- `audit_vn30_hourly_2015_jan2025_benchmark.py`
- `audit_vn30_hourly_2015_overall_directional_final65.py`
- `audit_vn30_hourly_2015_overall_directional_final65_v2.py`
- `audit_vn30_hourly_2015_rf_h60_final65_focus.py`
- `audit_vn30_hourly_2015_rf_h60_final65_router_v2.py`
- `audit_vn30_hourly_2015_target60_65_results.py`
- `audit_vn30_hourly_2015_target_redesign_results.py`
- `audit_vn30_hourly_2015_validation_final_mismatch.py`
- `audit_vn30_hourly_rf_h60_result_consistency.py`
- `run_vn30_hourly_2015_above60_experiments.py`
- `run_vn30_hourly_2015_above60_optimization.py`
- `run_vn30_hourly_2015_all_model_final65_router.py`
- `run_vn30_hourly_2015_confidence_sweep.py`
- `run_vn30_hourly_2015_cost_slippage_proxy.py`
- `run_vn30_hourly_2015_final65_focus_v3.py`
- `run_vn30_hourly_2015_full_tuning_sweep.py`
- `run_vn30_hourly_2015_hard_optimization_v2.py`
- `run_vn30_hourly_2015_horizon_relative_target_experiments.py`
- `run_vn30_hourly_2015_jan2025_benchmark.py`
- `run_vn30_hourly_2015_overall_directional_final65.py`
- `run_vn30_hourly_2015_overall_directional_final65_v2.py`
- `run_vn30_hourly_2015_regime_diagnostics.py`
- `run_vn30_hourly_2015_rf_h60_final65_focus.py`
- `run_vn30_hourly_2015_rf_h60_final65_router_v2.py`
- `run_vn30_hourly_2015_significance_diagnostics.py`
- `run_vn30_hourly_2015_target60_baseline_v2.py`
- `run_vn30_hourly_2015_target65_final_v2.py`
- `run_vn30_hourly_2015_target_redesign_experiments.py`

Moved to `scripts/legacy/research/paper_builders/`:

- `build_vn30_hourly_available_window_paper_artifact_pack.py`

Moved to `scripts/legacy/research/old_hourly_2005_2026/`:

- `validate_external_vn30_hourly_dataset.py`

## Kept in Place

- Provider gateway and adapter files were not moved.
- Active supported-index, daily VN30 2015, hourly available-window, top-k, audit, hygiene, and preflight scripts were not moved.
- Imported 2005/2026 compatibility modules were not moved.
- Ambiguous listing-aware and vnstock wrapper scripts were left for manual review.

## Import Risk Review

Reference checks were run before moving with `rg` over module names and filenames. After moving, a second check searched moved module names in `scripts/research`, `tests`, and `src`; it returned no references.

The remaining risk is documentation-only: some historical reports and generated environment reports may mention old paths. Those reports are evidence records and were not rewritten.
