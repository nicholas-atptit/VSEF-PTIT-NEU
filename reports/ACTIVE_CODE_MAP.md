# Active Code Map

This map describes the repository structure after the new-remote backup and before code cleanup. It is descriptive only; it does not change research claims or benchmark results.

## 1. Provider/API Adapter Layer

Canonical provider contract and gateway:

- `src/data/providers/vn_price_gateway.py` - guarded VN OHLCV gateway for stock/index history requests.
- `src/data/providers/vn_provider_contract.py` - typed request/response contract for provider usage.
- `src/data/adapters/vnstock_adapter.py` - repository adapter around provider behavior and provenance.

Provider enforcement and tests:

- `scripts/check_provider_usage_policy.py` - static policy guard for raw `vnstock`/`vnstock_data` imports.
- `tests/data/test_provider_usage_policy.py` - policy-unit tests.
- `tests/data/test_vn_price_gateway_contract.py` - gateway contract tests.

Provider-facing active fetch/validation scripts:

- `scripts/research/fetch_supported_indices_daily_gateway_2015.py`
- `scripts/research/fetch_supported_indices_hourly_gateway_2015.py`
- `scripts/research/fetch_vn30_daily_gateway_2015.py`
- `scripts/research/fetch_vn30_stocks_hourly_gateway_2015.py`
- `scripts/research/fetch_vnstock_supported_indices_hourly.py`
- `scripts/research/refetch_supported_indices_hourly_gateway.py`
- `scripts/research/refetch_vn30_stocks_hourly_gateway.py`
- `scripts/research/validate_supported_indices_hourly_gateway.py`
- `scripts/research/validate_supported_indices_hourly_gateway_2015.py`
- `scripts/research/validate_vn30_stocks_hourly_gateway.py`
- `scripts/research/validate_vn30_stocks_hourly_gateway_2015.py`

## 2. Benchmark/Evaluation Layer

Canonical evaluator and helpers:

- `scripts/research/vn30_hourly_2015_canonical_eval.py` - canonical hourly 2015 evaluation utility.
- `scripts/research/vn30_hourly_available_window_common.py` - active available-window dataset/evaluation helpers.
- `scripts/research/index_benchmark_common.py` - supported-index benchmark data/evaluation helpers.
- `src/ml/metrics.py` - core directional accuracy and prediction metric helpers.
- `tests/ml/test_directional_accuracy_metrics.py` - targeted metric tests.

Benchmark scripts:

- `scripts/research/run_supported_indices_directional_benchmark.py`
- `scripts/research/run_vn30_daily_2015_benchmark.py`
- `scripts/research/run_vn30_hourly_available_window_benchmark.py`
- `scripts/run_vn100_hybrid_frequency_accuracy_benchmark.py`

Audit and readiness scripts:

- `scripts/research/audit_codebase_structure.py`
- `scripts/research/audit_supported_indices_data_scope.py`
- `scripts/research/audit_supported_indices_directional_benchmark.py`
- `scripts/research/audit_vn30_daily_2015_benchmark.py`
- `scripts/research/audit_vn30_daily_2015_accuracy_drag.py`
- `scripts/research/audit_vn30_hourly_available_window.py`
- `scripts/research/audit_vn30_hourly_data_locations.py`
- `scripts/research/build_vn30_daily_2015_readiness_manifest.py`
- `scripts/research/build_vn30_gateway_benchmark_readiness_manifest.py`
- `scripts/research/validate_supported_indices_benchmark_readiness.py`
- `scripts/research/validate_vn30_daily_2015.py`

## 3. Research Scripts

Active daily:

- `scripts/research/run_vn30_daily_2015_benchmark.py`
- `scripts/research/run_vn30_daily_2015_target60_optimization.py`
- `scripts/research/run_vn30_daily_2015_target60_v2.py`
- `scripts/research/audit_vn30_daily_2015_benchmark.py`
- `scripts/research/audit_vn30_daily_2015_accuracy_drag.py`
- `scripts/research/audit_vn30_daily_2015_target60_failure_postmortem.py`
- `scripts/research/audit_vn30_daily_2015_target60_optimization.py`
- `scripts/research/audit_vn30_daily_2015_target60_v2.py`

Active hourly available-window:

- `scripts/research/run_vn30_hourly_available_window_benchmark.py`
- `scripts/research/run_vn30_hourly_available_window_confidence_sweep.py`
- `scripts/research/run_vn30_hourly_available_window_cost_slippage_validation.py`
- `scripts/research/run_vn30_hourly_available_window_exante_regime_validation.py`
- `scripts/research/audit_vn30_hourly_available_window.py`
- `scripts/research/vn30_hourly_available_window_common.py`

Active index:

- `scripts/research/run_supported_indices_directional_benchmark.py`
- `scripts/research/audit_supported_indices_data_scope.py`
- `scripts/research/audit_supported_indices_directional_benchmark.py`
- `scripts/research/index_benchmark_common.py`
- `scripts/research/validate_supported_indices_benchmark_readiness.py`

Active data forensics and provider diagnostics:

- `scripts/research/audit_vn30_hourly_data_locations.py`
- `scripts/research/diagnose_vn30_hourly_2015_accuracy_drag.py`
- `scripts/research/diagnose_vn30_vnstock_hourly_fetch_failures.py`
- `scripts/research/probe_vnstock_hourly_provider_capability.py`
- `scripts/research/probe_vnstock_supported_indices.py`
- `scripts/research/verify_repo_vnstock_provider_paths.py`
- `scripts/research/verify_vnstock_data_environment.py`

Top-k separate metric family:

- `scripts/research/run_vn30_hourly_2015_topk_ranking_experiments.py`
- `scripts/research/audit_vn30_hourly_2015_topk_ranking_results.py`
- `scripts/research/verify_vn30_hourly_2015_topk_75_result.py`
- `scripts/research/verify_vn30_hourly_2015_topk_75_null_test.py`

## 4. Legacy Scripts

Legacy code is preserved rather than deleted. Existing legacy scripts are under:

- `scripts/legacy/`
- `scripts/legacy/research/`

Likely legacy families:

- Old 2005/2026 hourly scripts, especially full-period VN30 hourly runners and external dataset validators.
- Old paper/artifact builders.
- Old failed-gate and optimization sweeps for target60/target65/final65 experiments.
- Diagnostic probes that intentionally inspect provider behavior directly.

Imported compatibility wrappers are kept in `scripts/research` until a later refactor can update import paths safely.

## 5. Protected Data/Artifact Paths

The following data and artifact paths are protected for this cleanup:

- `data/`
- `outputs/`
- `archive/generated_data_snapshots/`
- `reports/generated/`
- `archive/reports_superseded/`

These paths remain preserved. This cleanup does not delete data, rerun benchmarks, fetch data, train models, or regenerate paper/DOCX artifacts.
