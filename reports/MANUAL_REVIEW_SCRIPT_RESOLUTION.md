# Manual Review Script Resolution

## Summary

- Scripts reviewed count: 13.
- Scripts moved count: 0.
- Scripts kept count: 13.
- Scripts still ambiguous count: 4.
- Benchmark run: no.
- Data fetch run: no.
- Model training run: no.
- Paper/DOCX generated: no.

This pass was intentionally conservative. The previous cleanup already moved 37 superseded scripts. Remaining candidates either serve as active wrapper dependencies, readiness-plan references, or provider/data-forensics evidence-adjacent entrypoints.

## Reviewed Scripts

| script | status | decision | reference evidence | reason |
|---|---|---|---|---|
| `scripts/research/build_vn30_2015_benchmark_readiness_manifest.py` | keep_manual_review | kept | Filename and module referenced by `reports/VN30_HOURLY_2015_DATA_READINESS_PLAN.md`. | Readiness-plan dependency; keep until the 2015 hourly readiness docs are superseded. |
| `scripts/research/run_vn30_hourly_benchmark_2005_2026.py` | keep_manual_review | kept | Referenced by archived rerun plan, generated missing-evidence report, provider-path evidence, cleanup inventory, and cleanup plan. | Old full-period runner, but still evidence-adjacent and paired with readiness gate documentation. |
| `scripts/research/run_vn30_hourly_benchmark_2005_2026_from_fetched.py` | active_wrapper_dependency | kept | Referenced by `scripts/research/build_vn30_gateway_benchmark_readiness_manifest.py` as the gated later benchmark command. | Readiness manifest builder points to this command; moving would break documented gate output. |
| `scripts/research/run_vn30_hourly_confidence_sweep_2005_2026.py` | active_imported | kept | Imported by `run_vn30_hourly_available_window_confidence_sweep.py`, `run_vn30_hourly_listing_aware_confidence_sweep.py`, and `run_vn30_hourly_vnstock_confidence_sweep.py`. | Shared implementation used by active available-window wrapper family. |
| `scripts/research/run_vn30_hourly_cost_slippage_validation_2005_2026.py` | active_imported | kept | Imported by `run_vn30_hourly_available_window_cost_slippage_validation.py`, `run_vn30_hourly_listing_aware_cost_slippage_validation.py`, and `run_vn30_hourly_vnstock_cost_slippage_validation.py`. | Shared implementation used by active available-window wrapper family. |
| `scripts/research/run_vn30_hourly_exante_regime_validation_2005_2026.py` | active_imported | kept | Imported by `run_vn30_hourly_available_window_exante_regime_validation.py`, `run_vn30_hourly_listing_aware_exante_regime_validation.py`, and `run_vn30_hourly_vnstock_exante_regime_validation.py`. | Shared implementation used by active available-window wrapper family. |
| `scripts/research/run_vn30_hourly_listing_aware_benchmark.py` | keep_manual_review | kept | Referenced by generated listing-aware missing-evidence report and provider-path evidence. | Listing-aware track is older but still data-forensics/evidence-adjacent. Keep until generated listing-aware evidence is reconciled. |
| `scripts/research/run_vn30_hourly_listing_aware_confidence_sweep.py` | keep_manual_review | kept | Imports shared confidence-sweep implementation; referenced by provider-path evidence and cleanup inventory. | Wrapper is not imported elsewhere, but it documents the listing-aware diagnostic family. |
| `scripts/research/run_vn30_hourly_listing_aware_cost_slippage_validation.py` | keep_manual_review | kept | Imports shared cost/slippage implementation; referenced by provider-path evidence and cleanup inventory. | Wrapper is not imported elsewhere, but it documents the listing-aware diagnostic family. |
| `scripts/research/run_vn30_hourly_listing_aware_exante_regime_validation.py` | keep_manual_review | kept | Imports shared ex-ante regime implementation; referenced by provider-path evidence and cleanup inventory. | Wrapper is not imported elsewhere, but it documents the listing-aware diagnostic family. |
| `scripts/research/run_vn30_hourly_vnstock_confidence_sweep.py` | diagnostic_only | kept | Imports shared confidence-sweep implementation; referenced by provider-path evidence and cleanup inventory. | Vnstock-specific diagnostic wrapper remains useful for provider/data-forensics traceability. |
| `scripts/research/run_vn30_hourly_vnstock_cost_slippage_validation.py` | diagnostic_only | kept | Imports shared cost/slippage implementation; referenced by provider-path evidence and cleanup inventory. | Vnstock-specific diagnostic wrapper remains useful for provider/data-forensics traceability. |
| `scripts/research/run_vn30_hourly_vnstock_exante_regime_validation.py` | diagnostic_only | kept | Imports shared ex-ante regime implementation; referenced by provider-path evidence and cleanup inventory. | Vnstock-specific diagnostic wrapper remains useful for provider/data-forensics traceability. |

## Import And Reference Method

For each reviewed script, both filename and module-name searches were run:

- `rg "<filename>"`
- `rg "<module_name_without_py>"`

The resulting references were checked against active code, generated inventories, active evidence reports, archived reports, and provider-path diagnostics.

## Decision

No scripts were moved to `scripts/legacy/research/manual_review_resolved/` in this pass. The four listing-aware wrappers remain manual-review because they are not imported by active scripts but still map to generated listing-aware evidence and diagnostic scope. Moving them should wait for a specific listing-aware evidence reconciliation task.
