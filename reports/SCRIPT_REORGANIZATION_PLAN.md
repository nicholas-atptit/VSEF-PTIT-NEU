# Script Reorganization Plan

## Rules

- Keep active provider scripts, canonical evaluators, active daily scripts, active hourly available-window scripts, active index scripts, active audit scripts, hygiene checks, and preflight checks in place.
- Before moving a script, check both module-name references and filename references with `rg`.
- Move only scripts that are unimported by active code or referenced only from reports/archive evidence.
- Preserve all files; do not delete scripts as part of this cleanup.
- Do not run benchmarks, fetch data, train models, generate paper/DOCX, create tags, or push tags.

## A. Active - Keep in `scripts/research`

- `analyze_vn100_ticker_concentration.py`
- `audit_codebase_structure.py`
- `audit_supported_indices_data_scope.py`
- `audit_supported_indices_directional_benchmark.py`
- `audit_vn100_cache_coverage.py`
- `audit_vn30_daily_2015_accuracy_drag.py`
- `audit_vn30_daily_2015_benchmark.py`
- `audit_vn30_daily_2015_target60_failure_postmortem.py`
- `audit_vn30_daily_2015_target60_optimization.py`
- `audit_vn30_daily_2015_target60_v2.py`
- `audit_vn30_hourly_2015_topk_ranking_results.py`
- `audit_vn30_hourly_available_window.py`
- `audit_vn30_hourly_data_locations.py`
- `build_vn30_daily_2015_readiness_manifest.py`
- `build_vn30_gateway_benchmark_readiness_manifest.py`
- `fetch_supported_indices_daily_gateway_2015.py`
- `fetch_supported_indices_hourly_gateway_2015.py`
- `fetch_vn30_daily_gateway_2015.py`
- `fetch_vn30_stocks_hourly_gateway_2015.py`
- `fetch_vnstock_supported_indices_hourly.py`
- `index_benchmark_common.py`
- `prepare_clean_vn30_hourly_data_workspace.py`
- `refetch_supported_indices_hourly_gateway.py`
- `refetch_vn30_stocks_hourly_gateway.py`
- `run_supported_indices_directional_benchmark.py`
- `run_vn100_cost_slippage_validation.py`
- `run_vn100_exante_regime_validation.py`
- `run_vn100_full_confidence_sweep.py`
- `run_vn100_multiwindow_validation.py`
- `run_vn30_daily_2015_benchmark.py`
- `run_vn30_daily_2015_target60_optimization.py`
- `run_vn30_daily_2015_target60_v2.py`
- `run_vn30_hourly_2015_topk_ranking_experiments.py`
- `run_vn30_hourly_available_window_benchmark.py`
- `run_vn30_hourly_available_window_confidence_sweep.py`
- `run_vn30_hourly_available_window_cost_slippage_validation.py`
- `run_vn30_hourly_available_window_exante_regime_validation.py`
- `validate_supported_indices_benchmark_readiness.py`
- `validate_supported_indices_hourly_gateway.py`
- `validate_supported_indices_hourly_gateway_2015.py`
- `validate_vn30_daily_2015.py`
- `validate_vn30_stocks_hourly_gateway.py`
- `validate_vn30_stocks_hourly_gateway_2015.py`
- `verify_vn30_hourly_2015_topk_75_null_test.py`
- `verify_vn30_hourly_2015_topk_75_result.py`
- `vn30_hourly_2015_canonical_eval.py`
- `vn30_hourly_2015_effective_start.py`
- `vn30_hourly_2015_fetch_plan.py`
- `vn30_hourly_available_window_common.py`
- `vn30_hourly_common.py`
- `vn30_hourly_vnstock_common.py`

## B. Diagnostic - Keep in `scripts/research`

- `audit_vn30_hourly_coverage_2005_2026.py`
- `diagnose_vn30_hourly_2015_accuracy_drag.py`
- `diagnose_vn30_vnstock_hourly_fetch_failures.py`
- `fetch_vn30_hourly_from_vnstock_2005_2026.py`
- `fetch_vn30_hourly_listing_aware_from_vnstock.py`
- `probe_vnstock_hourly_provider_capability.py`
- `probe_vnstock_supported_indices.py`
- `reset_vn30_hourly_2015_workspace.py`
- `validate_fetched_vn30_hourly_2005_2026.py`
- `validate_vn30_hourly_listing_aware_dataset.py`
- `validate_vnstock_supported_indices_hourly.py`
- `verify_repo_vnstock_provider_paths.py`
- `verify_vnstock_data_environment.py`
- `vn30_hourly_listing_aware_common.py`

## C. Legacy - Safe to Move

Move to `scripts/legacy/research/failed_experiments/`:

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

Move to `scripts/legacy/research/paper_builders/`:

- `build_vn30_hourly_available_window_paper_artifact_pack.py`

Move to `scripts/legacy/research/old_hourly_2005_2026/`:

- `validate_external_vn30_hourly_dataset.py`

## D. Manual Review - Keep in `scripts/research`

- `build_vn30_2015_benchmark_readiness_manifest.py` - referenced by `reports/VN30_HOURLY_2015_DATA_READINESS_PLAN.md`; keep until the readiness-doc dependency is clarified.
- `run_vn30_hourly_benchmark_2005_2026.py` - referenced by `scripts/research/build_vn30_gateway_benchmark_readiness_manifest.py`; keep.
- `run_vn30_hourly_benchmark_2005_2026_from_fetched.py` - referenced by `scripts/research/build_vn30_gateway_benchmark_readiness_manifest.py`; keep.
- `run_vn30_hourly_confidence_sweep_2005_2026.py` - imported by available-window/listing-aware/vnstock wrapper scripts; keep.
- `run_vn30_hourly_cost_slippage_validation_2005_2026.py` - imported by available-window/listing-aware/vnstock wrapper scripts; keep.
- `run_vn30_hourly_exante_regime_validation_2005_2026.py` - imported by available-window/listing-aware/vnstock wrapper scripts; keep.
- `run_vn30_hourly_listing_aware_benchmark.py` - older listing-aware track but related wrappers remain; keep.
- `run_vn30_hourly_listing_aware_confidence_sweep.py` - wrapper import dependencies remain; keep.
- `run_vn30_hourly_listing_aware_cost_slippage_validation.py` - wrapper import dependencies remain; keep.
- `run_vn30_hourly_listing_aware_exante_regime_validation.py` - wrapper import dependencies remain; keep.
- `run_vn30_hourly_vnstock_confidence_sweep.py` - imports shared 2005/2026 validation module; keep.
- `run_vn30_hourly_vnstock_cost_slippage_validation.py` - imports shared 2005/2026 validation module; keep.
- `run_vn30_hourly_vnstock_exante_regime_validation.py` - imports shared 2005/2026 validation module; keep.

## Reference Check Result

- Failed-experiment candidates had no active code references after excluding generated cleanup inventory and archived reports; two were mentioned only in `reports/smoke_cart.csv`.
- The paper-builder candidate was referenced only by generated environment reports and manual-review documentation.
- The external dataset validator was referenced only by generated environment reports.
- Imported 2005/2026 wrapper modules were not moved.
