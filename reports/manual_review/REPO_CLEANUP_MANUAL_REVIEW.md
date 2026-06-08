# Repository Cleanup Manual Review

- Created at UTC: 2026-05-17.
- Branch: `research/vn100-evidence-hardening-v1`.
- Purpose: list ambiguous files or folders that were intentionally kept in place.

Manual review item count: 16.

## Kept Reports Needing Review

1. `reports/superseded/VN100_HYBRID_BENCHMARK_CLOSEOUT.md` - final-looking VN100 evidence, but not part of the current active five-track evidence index.
2. `reports/superseded/NCKH_EVIDENCE_GAP_CLOSURE_REGISTER.md` - may contain evidence traceability, so it was kept.
3. `reports/superseded/NCKH_EVIDENCE_UPGRADE_ADDENDUM_V2.md` - may contain current claim-boundary notes, so it was kept.
4. `reports/superseded/NCKH_EXPERIMENT_INVENTORY.md` - references old paper artifact builders but may still serve as historical experiment index.
5. `reports/superseded/NCKH_RESULTS_CLAIM_REGISTER.md` - superseded status unclear because a V2 also exists.
6. `reports/superseded/NCKH_RESULTS_CLAIM_REGISTER_V2.md` - may be evidence-adjacent and should be reviewed against the active claim registers.
7. `reports/superseded/NCKH_RESEARCH_DESIGN_VN100.md` - old VN100 design scope but could still explain historical experiments.
8. `reports/results/VN30_HOURLY_2015_JAN2025_BENCHMARK_RESULT_SUMMARY.md` - universe-specific evidence; kept until reviewed against current data-forensics scope.
9. `reports/protocols/VN30_HOURLY_2015_JAN2025_UNIVERSE_DECISION.md` - may still explain universe membership.
10. `reports/results/VN30_HOURLY_2015_POST_BENCHMARK_DIAGNOSTICS_SUMMARY.md` - diagnostic evidence; kept rather than archived.

## Kept Generated Folders Needing Review

11. `reports/generated/evidence_gap_closure/` - VN100 generated artifacts; kept because evidence status is ambiguous.
12. `reports/generated/vn30_hourly_listing_aware/` - provider/fetch diagnostics may still support data-forensics claims.
13. `reports/generated/vn30_hourly_vnstock_fetch/` - provider/fetch diagnostics may still support data-forensics claims.
14. Loose generated VN100 files under `reports/generated/` - kept until reconciled with `VN100_HYBRID_BENCHMARK_CLOSEOUT.md`.

## Kept Scripts Needing Review

15. `scripts/research/*2005_2026*.py` - several are old, but some are imported by active available-window wrappers or referenced by provider-standardization evidence.
16. `scripts/research/build_vn30_hourly_available_window_paper_artifact_pack.py` - paper artifact builder kept because it targets the current available-window scope; review before moving.

## Review Rule

If a manual-review item is not needed for current evidence, move it to an archive path in a separate cleanup commit. If it is needed, keep it in place and add a short note to the relevant active evidence report.
