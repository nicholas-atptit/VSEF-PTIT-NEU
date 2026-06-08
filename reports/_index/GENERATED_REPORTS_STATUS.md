# Generated Reports Status

- Created at UTC: 2026-05-17.
- Branch: `research/vn100-evidence-hardening-v1`.
- Generated folders archived: 21.
- Generated files archived: 104.
- Generated evidence deleted: no.
- Tracked `outputs/index_directional_benchmark/*` files were removed from Git tracking only; local files were preserved.

## Active Evidence

Kept in `reports/generated/` because these support current evidence or claim boundaries:

- `environment/`
- `index_benchmark/`
- `index_hourly_fetch/`
- `index_hourly_gateway/`
- `vn30_daily_2015/`
- `vn30_daily_2015_target60/`
- `vn30_daily_2015_target60_postmortem/`
- `vn30_daily_2015_target60_v2/`
- `vn30_gateway_benchmark_readiness/`
- `vn30_hourly_2015/`
- `vn30_hourly_2015_audit/`
- `vn30_hourly_2015_benchmark/`
- `vn30_hourly_2015_benchmark_readiness/`
- `vn30_hourly_2015_consistency/`
- `vn30_hourly_2015_topk_verification/`
- `vn30_hourly_2015_validation_final_mismatch/`
- `vn30_hourly_available_window/`
- `vn30_hourly_data_forensics/`
- `vn30_hourly_gateway/`

## Superseded Evidence Archived

Moved to `archive/reports_superseded/generated/`:

- `paper_figures/`
- `paper_notes/`
- `paper_tables/`
- `index_hourly_2015/`
- `vn30_hourly_2005_2026/`
- `vn30_hourly_2015_above60_optimization/`
- `vn30_hourly_2015_all_model_final65_router/`
- `vn30_hourly_2015_final65_focus_v3/`
- `vn30_hourly_2015_full_tuning/`
- `vn30_hourly_2015_hard_optimization_v2/`
- `vn30_hourly_2015_horizon_relative_target/`
- `vn30_hourly_2015_overall_directional_final65/`
- `vn30_hourly_2015_overall_directional_final65_v2/`
- `vn30_hourly_2015_reset/`
- `vn30_hourly_2015_reverse_fetch_prep/`
- `vn30_hourly_2015_rf_h60_final65_focus/`
- `vn30_hourly_2015_rf_h60_final65_router_v2/`
- `vn30_hourly_2015_target60_65/`
- `vn30_hourly_2015_target_redesign/`
- `vn30_hourly_clean_workspace/`
- `working_tree_cleanup/`

## Failed-Gate Evidence Kept

Kept rather than archived where it still supports data-forensics or claim-boundary reasoning:

- `vn30_hourly_2015/`
- `vn30_hourly_2015_benchmark_readiness/`
- `vn30_hourly_2015_validation_final_mismatch/`
- `vn30_hourly_listing_aware/`
- `vn30_hourly_vnstock_fetch/`

## Temporary Junk

- No generated report evidence was deleted.
- Deleted junk was limited to ignored Python bytecode under `scripts/legacy/`.

## Manual Review

Review before moving or deleting:

- `evidence_gap_closure/`
- `vn30_hourly_listing_aware/`
- `vn30_hourly_vnstock_fetch/`
- loose VN100 generated files under `reports/generated/`

## Boundary

- Raw data, market cache, ignored outputs, and archive snapshots were not touched.
- Existing dirty generated fetch files under `reports/generated/vn30_hourly_2015/fetch/` were not staged by this cleanup.
