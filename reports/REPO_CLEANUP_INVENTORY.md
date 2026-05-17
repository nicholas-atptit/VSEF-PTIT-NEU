# Repository Cleanup Inventory

- Created at UTC: 2026-05-17.
- Branch: `research/vn100-evidence-hardening-v1`.
- Cleanup type: safe archive-and-document cleanup.
- Tags created: no.
- Tags pushed: no.
- Main touched: no.
- Benchmark run: no.
- Data fetch: no.
- Model training: no.
- Paper or DOCX generated: no.

## Starting State

- `git status --short` showed pre-existing unrelated dirty files in `README.md`, archived VN100 scripts, `reports/audits/backtest_risk_audit.md`, several `reports/generated/vn30_hourly_2015/fetch/*` files, and production source files.
- Those pre-existing dirty files were not used as cleanup scope and must not be staged as part of this cleanup.
- `git ls-files` showed tracked reports, source, scripts, docs, configs, tests, some tracked generated summaries, and one tracked output directory for `outputs/index_directional_benchmark/`.
- `outputs/`, raw data, market cache, and archive snapshots were treated as do-not-touch locations.

## A. Keep Active

Reason: these files define current evidence, current claim boundaries, current source behavior, or reproducibility checks.

- Current source code under `src/`, `config/`, `configs/`, `alembic/`, and tests under `tests/`.
- Provider gateway and provider-policy tests:
  - `src/data/providers/vn_price_gateway.py`
  - `src/data/providers/vn_provider_contract.py`
  - `tests/data/test_provider_usage_policy.py`
  - `tests/data/test_vn_price_gateway_contract.py`
- Current index benchmark evidence:
  - `reports/INDEX_DIRECTIONAL_BENCHMARK_RESULT_SUMMARY.md`
  - `reports/INDEX_DIRECTIONAL_BENCHMARK_CLAIM_REGISTER.md`
  - `reports/INDEX_HOURLY_FETCH_README.md`
  - `outputs/index_directional_benchmark/`
  - `reports/generated/index_benchmark/`
- Current VN30 daily 2015 evidence:
  - `reports/VN30_DAILY_2015_BCM_VIB_RECOVERY_REPORT.md`
  - `reports/VN30_DAILY_2015_BENCHMARK_RESULT_SUMMARY.md`
  - `reports/VN30_DAILY_2015_CLAIM_REGISTER.md`
  - `reports/VN30_DAILY_2015_TARGET60_FAILURE_POSTMORTEM_PROTOCOL.md`
  - `reports/VN30_DAILY_2015_TARGET60_NEXT_STEP_DECISION.md`
  - `reports/generated/vn30_daily_2015/`
  - `reports/generated/vn30_daily_2015_target60_postmortem/`
- Current hourly available-window and data-forensics evidence:
  - `reports/VN30_HOURLY_DATA_FORENSICS_DIAGNOSIS.md`
  - `reports/VN30_HOURLY_2015_BASELINE60_RESULT_LOCK.md`
  - `reports/VN30_HOURLY_2015_BENCHMARK_CLAIM_REGISTER.md`
  - `reports/VN30_HOURLY_2015_CANONICAL_EVALUATOR_DECISION.md`
  - `reports/generated/vn30_hourly_available_window/`
  - `reports/generated/vn30_hourly_data_forensics/`
- Top-k ranking evidence, because it is a separate metric family:
  - `reports/VN30_HOURLY_2015_TOPK_RANKING_RESULT_SUMMARY.md`
  - `reports/VN30_HOURLY_2015_TOPK_RANKING_CLAIM_REGISTER.md`
  - `reports/VN30_HOURLY_2015_TOPK_75_VERIFICATION_DECISION.md`
  - `reports/generated/vn30_hourly_2015_topk_verification/`
- Current provider standardization docs:
  - `reports/VNSTOCK_PROVIDER_STANDARDIZATION_GUIDE.md`
  - `reports/VNSTOCK_PROVIDER_STANDARDIZATION_CHANGELOG.md`
  - `reports/PROVIDER_STANDARDIZATION_AUDIT.md`
- Hygiene and runtime checks:
  - `scripts/check_repo_hygiene.py`
  - `scripts/check_runtime_preflight.py`

## B. Archive

Reason: these are superseded paper drafts, old build notes, old failed-gate reports, or duplicated generated summaries. They are preserved under `archive/reports_superseded/` rather than deleted.

- Archived root reports: 65 files.
- Archived generated files: 104 files.
- Archived report destination: `archive/reports_superseded/`.
- Archived generated destination: `archive/reports_superseded/generated/`.
- Tracked generated output files under `outputs/index_directional_benchmark/` were removed from Git tracking only. The local files remain in place under the ignored `outputs/` directory.
- Main archived groups:
  - NCKH paper drafts, submission notes, defense notes, and DOCX build notes.
  - VN30 hourly 2005/2026 data requirements and rerun plans superseded by current data forensics.
  - VN30 hourly 2015 failed target-60/65, final65, full tuning, hard optimization, router, and target-redesign report bundles.
  - Generated paper tables/figures/notes.
  - Generated VN30 hourly 2005/2026 and failed VN30 hourly 2015 tuning folders.
  - Generated cleanup/reset/reverse-fetch prep folders.

## C. Delete

Reason: these were local generated Python bytecode files and not evidence.

- Deleted 20 ignored `__pycache__` files under `scripts/legacy/`.
- No raw data was deleted.
- No market cache was deleted.
- No benchmark output directory was deleted.
- No archive snapshot was deleted.

## D. Do Not Touch

Reason: these paths are data, cache, ignored output, snapshots, or unrelated dirty work.

- `data/raw/`
- `data/market_cache/`
- `data/hourly_data_split_data/`
- `outputs/`
- `archive/generated_data_snapshots/`
- Existing archive snapshots under `archive/`
- Pre-existing dirty source and report files shown by `git status --short`
- Large untracked or ignored data/output locations

## Cleanup Boundary

- This cleanup uses commits and reports only.
- It does not create or push tags.
- It does not merge or switch to `main`.
- It does not alter empirical metrics or benchmark results.
- It does not make trading, profitability, live-deployment, or paper-completion claims.
