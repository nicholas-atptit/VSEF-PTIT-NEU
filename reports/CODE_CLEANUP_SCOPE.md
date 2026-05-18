# Code Cleanup Scope

## Goal

Clean and reorganize the repository so the active VN market directional benchmark code is easier to understand, audit, and maintain. This is a code hygiene and structure cleanup task only.

## Protected Paths

The following paths are protected and must not be deleted or reorganized as part of this cleanup:

- `data/`
- `outputs/`
- `archive/generated_data_snapshots/`
- `reports/generated/`
- `archive/reports_superseded/`
- `src/data/providers/`
- `src/data/adapters/`
- `scripts/check_provider_usage_policy.py`
- `scripts/check_repo_hygiene.py`
- `scripts/check_runtime_preflight.py`
- `tests/data/`
- `tests/ml/`

The following active evidence reports must remain preserved:

- `reports/ACTIVE_EVIDENCE_INDEX.md`
- `reports/REPO_CLEANUP_INVENTORY.md`
- `reports/VN30_INDEX_BENCHMARK_RESULT_SUMMARY.md`
- `reports/VN30_DAILY_2015_RESULT_SUMMARY.md`
- `reports/VN30_HOURLY_DATA_FORENSICS_DIAGNOSIS.md`
- `reports/VN30_HOURLY_2015_BASELINE60_RESULT_LOCK.md`

## Out of Scope

This cleanup will not:

- Delete data, market cache, raw fetch data, outputs, generated data snapshots, generated reports, or archived superseded reports.
- Alter empirical benchmark results.
- Change research claims.
- Run benchmarks.
- Fetch market data.
- Train models.
- Generate paper or DOCX artifacts.
- Change provider behavior or break the provider/API adapter contract.
- Break current active research scripts.
- Stage unrelated dirty files.
- Touch or merge `main`.

## Git Scope

- Tags will not be created.
- Tags will not be pushed.
- `git push --mirror` will not be used.
- The cleanup will be committed on `research/vn100-evidence-hardening-v1`.
- The branch will be pushed with a branch-only push.
