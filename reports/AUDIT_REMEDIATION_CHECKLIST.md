# Audit Remediation Checklist

Repository: `K:/Repos/VSEF-PTIT-NEU`
Branch: `audit-remediation-governed-runtime`
Baseline source commit: `56e53bc1d24db9a5d296399702f8419a43793ed9`
Checklist status: VERIFIED

Allowed statuses: NOT STARTED, IN PROGRESS, BLOCKED, VERIFIED, CLOSED

| ID | Item | Severity | Owner | Status | Date | Verification | Rollback | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CA-001 | Align active API with diagnostic-only authority boundary. | CRITICAL | TBD | NOT STARTED | 2026-05-10 | Pending API contract tests. | Pending rollback plan. | Active routes and schemas expose BUY/SELL/execution language. |
| CA-002 | Prevent mock/synthetic fallback from entering governed runtime outputs silently. | CRITICAL | TBD | NOT STARTED | 2026-05-10 | Pending fallback/provenance tests. | Pending rollback plan. | Add fail-closed audit mode and explicit provenance. |
| CA-003 | Enforce risk/regime feature exclusion when risk is disabled. | CRITICAL | TBD | NOT STARTED | 2026-05-10 | Pending `tests/test_risk_engine.py`. | Pending rollback plan. | Baseline test found `var_q` in CART features with `risk_enabled=False`. |
| CA-004 | Reconcile `arch` dependency and GARCH runtime preflight. | CRITICAL | TBD | NOT STARTED | 2026-05-10 | Pending `tests/phase2/test_garch_risk.py`. | Pending rollback plan. | Active Python 3.13.5 cannot import `arch`. |
| CA-005 | Restore VN100 universe count and historical/as-of contract. | HIGH | TBD | NOT STARTED | 2026-05-10 | Pending `tests/test_universe.py`. | Pending rollback plan. | Baseline fallback returned 82 tickers. |
| CA-006 | Stabilize news crawler import/provider injection contract. | HIGH | TBD | NOT STARTED | 2026-05-10 | Pending `tests/test_context.py`. | Pending rollback plan. | `src.context.news_crawler.Vnstock` patch target missing. |
| CA-007 | Version and update VN100 feature catalogue contract. | HIGH | TBD | NOT STARTED | 2026-05-10 | Pending `tests/test_feature_engineering_vn100.py`. | Pending rollback plan. | Test expects 19 features; current catalogue has 41. |
| CA-008 | Stop tests from mutating tracked report artifacts. | HIGH | TBD | NOT STARTED | 2026-05-10 | Pending full pytest plus clean `git status`. | Pending rollback plan. | Baseline pytest modified three existing reports. |
| CA-009 | Define reproducible environment/service/data preflight. | HIGH | TBD | NOT STARTED | 2026-05-10 | Pending preflight command output. | Pending rollback plan. | Local Postgres/Redis/Kafka/Ollama/provider assumptions are implicit. |
| CA-010 | Remove or isolate API v2 hardcoded/mock predictions and sizing. | HIGH | TBD | NOT STARTED | 2026-05-10 | Pending API v2 tests. | Pending rollback plan. | `routes_v2.py` returns hardcoded BUY/fusion and mocked portfolio value. |
| CA-011 | Govern cleanup of tracked `node_modules` artifacts. | MEDIUM | TBD | NOT STARTED | 2026-05-10 | Pending git tracked-file audit and front-end rebuild proof. | Pending rollback plan. | Baseline found 3333 tracked files under `src/api/ui/web/node_modules`. |
| CA-012 | Create canonical command and runner registry. | MEDIUM | TBD | NOT STARTED | 2026-05-10 | Pending docs review and smoke command mapping. | Pending rollback plan. | Many overlapping research, legacy, and runtime scripts. |
| CA-013 | Align FastAPI metadata/debug output with governance language. | MEDIUM | TBD | NOT STARTED | 2026-05-10 | Pending API smoke and log review. | Pending rollback plan. | `src/api/main.py` retains trading/execution framing and debug prints. |
| CA-014 | Label or isolate experimental training branches. | MEDIUM | TBD | NOT STARTED | 2026-05-10 | Pending module inventory and tests/docs. | Pending rollback plan. | TFT/RL/allocator placeholders mixed with active code. |
| CA-015 | Resolve documentation encoding and filename hygiene issues. | LOW | TBD | NOT STARTED | 2026-05-10 | Pending filename/encoding audit. | Pending rollback plan. | Mojibake and local-shell artifact signs observed. |
| CA-016 | Clarify canonical status of Phase 0 reports under `reports/`. | LOW | TBD | NOT STARTED | 2026-05-10 | Pending docs note. | Pending rollback plan. | README says `reports/` are historical snapshots. |

## Phase Closure Checklist

| Phase | Goal | Status | Verification Evidence | Rollback Verification | Closure Date | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Phase 0 | Audit baseline and tracking framework created. | VERIFIED | Report, plan, checklist, branch result, pytest result, and git status are recorded. Initial commit hash is recorded in the final Phase 0 handoff because a commit cannot contain its own hash. | No runtime rollback required. | 2026-05-10 | Runtime remediation remains not started. |
| Phase 1 | Runtime Stabilization. | NOT STARTED | Pending. | Pending. | TBD | Must address failing tests and test-generated dirty reports. |
| Phase 2 | Validation Hardening. | NOT STARTED | Pending. | Pending. | TBD | Must address VN100, feature catalogue, and risk-feature gating contracts. |
| Phase 3 | Governance Enforcement. | NOT STARTED | Pending. | Pending. | TBD | Must remove active BUY/SELL/execution authority from governed API. |
| Phase 4 | Architecture Cleanup. | NOT STARTED | Pending. | Pending. | TBD | Cleanup requires explicit approval for removals. |
| Phase 5 | Observability and Monitoring. | NOT STARTED | Pending. | Pending. | TBD | Add provenance and fallback visibility. |
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
