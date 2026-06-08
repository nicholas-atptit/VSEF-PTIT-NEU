# Research Script Status

- Created at UTC: 2026-05-17.
- Branch: `research/vn100-evidence-hardening-v1`.
- Scripts moved in this cleanup: 4.
- Destination for moved scripts: `scripts/legacy/research/`.

## A. Active Canonical Scripts

Provider gateway-facing fetch/validate scripts:

- `scripts/research/fetch_supported_indices_daily_gateway_2015.py`
- `scripts/research/fetch_supported_indices_hourly_gateway_2015.py`
- `scripts/research/fetch_vn30_daily_gateway_2015.py`
- `scripts/research/fetch_vn30_stocks_hourly_gateway_2015.py`
- `scripts/research/refetch_supported_indices_hourly_gateway.py`
- `scripts/research/refetch_vn30_stocks_hourly_gateway.py`
- `scripts/research/validate_supported_indices_benchmark_readiness.py`
- `scripts/research/validate_supported_indices_hourly_gateway.py`
- `scripts/research/validate_supported_indices_hourly_gateway_2015.py`
- `scripts/research/validate_vn30_daily_2015.py`
- `scripts/research/validate_vn30_stocks_hourly_gateway.py`
- `scripts/research/validate_vn30_stocks_hourly_gateway_2015.py`

Canonical evaluator and current benchmark scripts:

- `scripts/research/vn30_hourly_2015_canonical_eval.py`
- `scripts/research/vn30_hourly_available_window_common.py`
- `scripts/research/index_benchmark_common.py`
- `scripts/research/run_supported_indices_directional_benchmark.py`
- `scripts/research/run_vn30_daily_2015_benchmark.py`
- `scripts/research/run_vn30_daily_2015_target60_optimization.py`
- `scripts/research/run_vn30_daily_2015_target60_v2.py`
- `scripts/research/run_vn30_hourly_available_window_benchmark.py`
- `scripts/research/run_vn30_hourly_available_window_confidence_sweep.py`
- `scripts/research/run_vn30_hourly_available_window_cost_slippage_validation.py`
- `scripts/research/run_vn30_hourly_available_window_exante_regime_validation.py`

Current audit scripts:

- `scripts/research/audit_supported_indices_data_scope.py`
- `scripts/research/audit_supported_indices_directional_benchmark.py`
- `scripts/research/audit_vn30_daily_2015_accuracy_drag.py`
- `scripts/research/audit_vn30_daily_2015_benchmark.py`
- `scripts/research/audit_vn30_daily_2015_target60_failure_postmortem.py`
- `scripts/research/audit_vn30_daily_2015_target60_optimization.py`
- `scripts/research/audit_vn30_daily_2015_target60_v2.py`
- `scripts/research/audit_vn30_hourly_available_window.py`
- `scripts/research/audit_vn30_hourly_data_locations.py`

Top-k scripts kept active as a separate metric family:

- `scripts/research/run_vn30_hourly_2015_topk_ranking_experiments.py`
- `scripts/research/audit_vn30_hourly_2015_topk_ranking_results.py`
- `scripts/research/verify_vn30_hourly_2015_topk_75_result.py`
- `scripts/research/verify_vn30_hourly_2015_topk_75_null_test.py`

## B. Legacy or Superseded Scripts

Moved because `rg <module name>` showed no active imports and the scripts are old paper/artifact-pack builders:

- `scripts/legacy/research/build_vn100_paper_artifact_pack.py`
- `scripts/legacy/research/build_vn30_hourly_listing_aware_paper_artifact_pack.py`
- `scripts/legacy/research/build_vn30_hourly_paper_artifact_pack_2005_2026.py`
- `scripts/legacy/research/build_vn30_hourly_vnstock_paper_artifact_pack.py`

Kept in place despite legacy naming because they are imported or referenced:

- `scripts/research/run_vn30_hourly_confidence_sweep_2005_2026.py` - imported by available-window, listing-aware, and vnstock wrappers.
- `scripts/research/run_vn30_hourly_cost_slippage_validation_2005_2026.py` - imported by available-window, listing-aware, and vnstock wrappers.
- `scripts/research/run_vn30_hourly_exante_regime_validation_2005_2026.py` - imported by available-window, listing-aware, and vnstock wrappers.
- `scripts/research/run_vn30_hourly_benchmark_2005_2026_from_fetched.py` - referenced by the gateway readiness manifest builder.
- `scripts/research/fetch_vn30_hourly_from_vnstock_2005_2026.py` - referenced by provider-standardization evidence.
- `scripts/research/validate_fetched_vn30_hourly_2005_2026.py` - referenced by provider-standardization evidence.

## C. Provider Probes and Diagnostics

Keep, diagnostics only:

- `scripts/research/probe_vnstock_hourly_provider_capability.py`
- `scripts/research/probe_vnstock_supported_indices.py`
- `scripts/research/validate_vnstock_supported_indices_hourly.py`
- `scripts/research/verify_repo_vnstock_provider_paths.py`
- `scripts/research/verify_vnstock_data_environment.py`
- `scripts/research/diagnose_vn30_vnstock_hourly_fetch_failures.py`
- `scripts/research/diagnose_vn30_hourly_2015_accuracy_drag.py`
- `scripts/research/audit_vn100_cache_coverage.py`
- `scripts/research/audit_vn30_hourly_coverage_2005_2026.py`

## Move Safety Check

- Before moving scripts, module-name searches were run with `rg`.
- Scripts with active imports were not moved.
- No benchmark scripts were executed.
- No data-fetch scripts were executed.
