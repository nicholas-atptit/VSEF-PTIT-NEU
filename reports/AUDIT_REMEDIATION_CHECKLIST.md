# Audit Remediation Checklist

Repository: governed workspace root
Branch: `audit-remediation-governed-runtime`
Baseline source commit: `56e53bc1d24db9a5d296399702f8419a43793ed9`
Checklist status: VERIFIED

Allowed statuses: NOT STARTED, IN PROGRESS, BLOCKED, PARTIALLY VERIFIED, VERIFIED, CLOSED

| ID | Item | Severity | Owner | Status | Date | Verification | Rollback | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CA-001 | Align active API with diagnostic-only authority boundary. | CRITICAL | TBD | NOT STARTED | 2026-05-10 | Pending API contract tests. | Pending rollback plan. | Active routes and schemas expose BUY/SELL/execution language. |
| CA-002 | Prevent mock/synthetic fallback from entering governed runtime outputs silently. | CRITICAL | Codex | VERIFIED | 2026-05-10 | `tests/test_api.py`, `tests/test_backtest.py`, `tests/test_data_loader_vn100.py`, and full `python -m pytest tests -q`: `807 passed, 5 skipped, 33 warnings in 305.62s`. | File-scoped restore of Phase 2 runtime/provenance files; rerun targeted tests. | Phase 2 added runtime modes, explicit mock provenance, audit mock blocking, and research silent-fallback blocking. |
| CA-003 | Enforce risk/regime feature exclusion when risk is disabled. | CRITICAL | Codex | VERIFIED | 2026-05-10 | Phase 1 commit `f4272a46676f2e4ed6bb2b973f1e58cd9a40119b`; `tests/test_risk_engine.py`; Phase 1 full suite `796 passed, 5 skipped, 33 warnings`; Phase 2 full suite `807 passed, 5 skipped, 33 warnings`. | Restore Phase 1 changes in `src/ml/trainer.py` and `tests/test_risk_engine.py`; rerun risk-engine tests and full suite. | Runtime exclusion and manifest-level test coverage were stabilized. Broader risk/regime validation-hardening remains deferred; not marked CLOSED. |
| CA-004 | Reconcile `arch` dependency and GARCH runtime preflight. | CRITICAL | Codex | VERIFIED | 2026-05-10 | Current repo declares `arch>=8.0` in `requirements.txt` and `pyproject.toml`; `tests/phase2/test_garch_risk.py`; Phase 1 evidence recorded `arch_version=8.0.0`; Phase 2 full suite `807 passed, 5 skipped, 33 warnings`. | Restore dependency declarations only if intentionally reverting GARCH support; rerun `tests/phase2/test_garch_risk.py` and full suite. | No Phase 1/2 code commit specifically changed dependency files in reviewed history; verification is dependency-file plus active-environment evidence. |
| CA-005 | Restore VN100 universe count and historical/as-of contract. | HIGH | Codex | PARTIALLY VERIFIED | 2026-05-10 | Phase 1 commit `f4272a46676f2e4ed6bb2b973f1e58cd9a40119b`; `tests/test_universe.py`; Phase 1 full suite `796 passed, 5 skipped, 33 warnings`; Phase 2 full suite `807 passed, 5 skipped, 33 warnings`. | Restore Phase 1 changes in `src/data/universe.py` and `tests/test_universe.py`; rerun universe tests. | Static fallback count/provenance and undersized-live fallback behavior were stabilized. Full historical/as-of constituent governance is not implemented; keep open for later validation governance. |
| CA-006 | Stabilize news crawler import/provider injection contract. | HIGH | Codex | VERIFIED | 2026-05-10 | Phase 1 commit `f4272a46676f2e4ed6bb2b973f1e58cd9a40119b`; `tests/test_context.py`; Phase 1 full suite `796 passed, 5 skipped, 33 warnings`; Phase 2 full suite `807 passed, 5 skipped, 33 warnings`. | Restore Phase 1 changes in `src/data/context/news_crawler.py` and `tests/test_context.py`; rerun context tests. | Provider injection/public patch target was stabilized without hardcoding provider behavior. Not a broader context-source governance closure. |
| CA-007 | Version and update VN100 feature catalogue contract. | HIGH | Codex | PARTIALLY VERIFIED | 2026-05-10 | Phase 1 commit `f4272a46676f2e4ed6bb2b973f1e58cd9a40119b`; `tests/test_feature_engineering_vn100.py`; Phase 1 full suite `796 passed, 5 skipped, 33 warnings`; Phase 2 full suite `807 passed, 5 skipped, 33 warnings`. | Restore Phase 1 test updates in `tests/test_feature_engineering_vn100.py`; rerun feature-engineering VN100 tests. | Stale fixed-count assertion was replaced with uniqueness, canonical presence, deterministic ordering, and transform compatibility checks. Formal runtime catalogue versioning/governance metadata remains deferred. |
| CA-008 | Stop tests from mutating tracked report artifacts. | HIGH | Codex | VERIFIED | 2026-05-10 | Full suite passed; Phase 0 report/plan diff is empty after test execution. | File-scoped restore of report checklist/evidence only if required. | Phase 2 generated no benchmark/report artifacts; only governed checklist/evidence docs are intentionally changed. |
| CA-009 | Define reproducible environment/service/data preflight. | HIGH | TBD | NOT STARTED | 2026-05-10 | Pending preflight command output. | Pending rollback plan. | Local Postgres/Redis/Kafka/Ollama/provider assumptions are implicit. |
| CA-010 | Remove or isolate API v2 hardcoded/mock predictions and sizing. | HIGH | Codex | PARTIALLY VERIFIED | 2026-05-10 | Phase 2 commit `4b67e38048afbb1283c6f02fc6818e68dc4f84f4`; `tests/test_api.py::TestRuntimeModeGovernance`; Phase 2 full suite `807 passed, 5 skipped, 33 warnings`. | Restore `src/api/routes_v2.py` and `src/api/schemas_v2.py`; rerun API tests. | v2 static prediction outputs are isolated/provenanced and audit-blocked. Hardcoded prediction semantics were not redesigned, and mock portfolio sizing in debate remains deferred by Phase 2 scope. |
| CA-011 | Govern cleanup of tracked `node_modules` artifacts. | MEDIUM | Codex | VERIFIED | 2026-05-10 | `reports/PHASE3_REPOSITORY_HYGIENE_EVIDENCE.md`; `python scripts/check_repo_hygiene.py`; `npm.cmd ci`; `npm.cmd run build`; full suite `807 passed, 5 skipped, 33 warnings in 262.15s`. | Revert the Phase 3 hygiene commit if dependency artifacts must be reintroduced; rerun `npm.cmd ci`, `npm.cmd run build`, and hygiene check. | 3333 tracked `node_modules` files were removed from git tracking; `package-lock.json` remains tracked. |
| CA-011A | Remove repository web UI surfaces from governed runtime. | MEDIUM | Codex | VERIFIED | 2026-05-10 | `reports/PHASE4A_FRONTEND_DESCOPE_EVIDENCE.md`; `tests/test_api.py`: `19 passed`; full suite: `808 passed, 5 skipped`; hygiene check passed. | Restore the dedicated frontend removal commit if rollback is required; rerun API/full tests plus hygiene check. | Removed `src/api/ui/web/`, root-level `web/`, FastAPI `/web` static mount, and dashboard redirect. |
| CA-012 | Create canonical command and runner registry. | MEDIUM | TBD | NOT STARTED | 2026-05-10 | Pending docs review and smoke command mapping. | Pending rollback plan. | Many overlapping research, legacy, and runtime scripts. |
| CA-013 | Align FastAPI metadata/debug output with governance language. | MEDIUM | TBD | NOT STARTED | 2026-05-10 | Pending API smoke and log review. | Pending rollback plan. | `src/api/main.py` retains trading/execution framing and debug prints. |
| CA-014 | Label or isolate experimental training branches. | MEDIUM | TBD | NOT STARTED | 2026-05-10 | Pending module inventory and tests/docs. | Pending rollback plan. | TFT/RL/allocator placeholders mixed with active code. |
| CA-015 | Resolve documentation encoding and filename hygiene issues. | LOW | Codex | PARTIALLY VERIFIED | 2026-05-10 | `reports/PHASE3_REPOSITORY_HYGIENE_EVIDENCE.md`; `python scripts/check_repo_hygiene.py` passes after malformed filename removal and active local-path cleanup. | Revert Phase 3 hygiene commit; rerun hygiene check. | Malformed filename and active machine-local path pollution were removed or normalized. Broader historical encoding cleanup remains deferred and is not marked closed. |
| CA-016 | Clarify canonical status of Phase 0 reports under `reports/`. | LOW | TBD | NOT STARTED | 2026-05-10 | Pending docs note. | Pending rollback plan. | README says `reports/` are historical snapshots. |

## Phase Closure Checklist

| Phase | Goal | Status | Verification Evidence | Rollback Verification | Closure Date | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Phase 0 | Audit baseline and tracking framework created. | VERIFIED | Report, plan, checklist, branch result, pytest result, and git status are recorded. Initial commit hash is recorded in the final Phase 0 handoff because a commit cannot contain its own hash. | No runtime rollback required. | 2026-05-10 | Runtime remediation remains not started. |
| Phase 1 | Runtime Stabilization. | CLOSED | Accepted complete by user; Phase 1 commit `f4272a46676f2e4ed6bb2b973f1e58cd9a40119b`; full suite was `796 passed, 5 skipped, 33 warnings`. | File-scoped Phase 1 rollback only if regression evidence appears. | 2026-05-10 | Do not reopen Phase 1 absent regression, reproducibility break, or governance conflict. |
| Phase 2 | Mock Isolation and Provenance Enforcement. | VERIFIED | `reports/PHASE2_MOCK_PROVENANCE_EVIDENCE.md`; full suite `807 passed, 5 skipped, 33 warnings in 305.62s`. | File-scoped restore of Phase 2 files; rerun targeted API/backtest/data-loader/trainer tests and full suite. | 2026-05-10 | VN100, feature catalogue, risk/regime manifest hardening, allocator logic, and API authority semantics are deferred. |
| Phase 3 | Repository Hygiene Cleanup. | VERIFIED | `reports/PHASE3_REPOSITORY_HYGIENE_EVIDENCE.md`; hygiene check, npm rebuild, full pytest, and git status recorded. | Revert Phase 3 hygiene commit; rerun hygiene check, npm rebuild, and full pytest. | 2026-05-10 | API Governance Enforcement is deferred to Phase 4 unless separately approved. |
| Phase 4A | Frontend Web UI De-Scope. | VERIFIED | `reports/PHASE4A_FRONTEND_DESCOPE_EVIDENCE.md`; `tests/test_api.py`: `19 passed`; full suite: `808 passed, 5 skipped`; hygiene check passed. | Restore dedicated frontend removal commit if required; rerun required checks. | 2026-05-10 | Removed repository web UI surfaces while retaining backend API runtime. |
| Phase 4 | API Governance Enforcement. | NOT STARTED | Pending. | Pending. | TBD | Must remove active BUY/SELL/execution authority from governed API under separate approval. |
| Phase 5 | Architecture Cleanup. | NOT STARTED | Pending. | Pending. | TBD | Cleanup requires explicit approval for removals. |
| Phase 6 | Final Verification and Signoff. | NOT STARTED | Pending. | Pending. | TBD | Full suite and git status signoff required. |

## Baseline Verification Record

| Command | Exit | Result | Status |
| --- | --- | --- | --- |
| `git checkout -b audit-remediation-governed-runtime` | 1 | Branch already existed; current branch already matched target. | VERIFIED |
| `python -m pytest tests -q` | 1 | `6 failed, 784 passed, 6 skipped, 30 warnings in 261.90s`. | VERIFIED |
| `git status` | 0 | Three existing reports modified by tests; audit report already untracked before normalization. | VERIFIED |

## Commit Verification Record

| Command | Exit | Result | Status |
| --- | --- | --- | --- |
| `git add reports/` | See final handoff | Executed after document authoring. Test-generated report diffs are intentionally excluded from the Phase 0 docs commit and left dirty. | VERIFIED |
| `git commit -m "docs: add audit remediation baseline"` | See final handoff | Commit hash is recorded in the final handoff. | VERIFIED |
