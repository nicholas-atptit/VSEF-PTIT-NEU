# Code Audit Remediation Plan

Repository: `K:/Repos/VSEF-PTIT-NEU`
Branch: `audit-remediation-governed-runtime`
Baseline source commit: `56e53bc1d24db9a5d296399702f8419a43793ed9`
Plan status: NOT STARTED

## Governance Rules

- Phase 0 is documentation and tracking only.
- Runtime code changes must start in Phase 1 or later.
- Every remediation item must map to a finding ID in `reports/cleanup/CODE_AUDIT_REPORT.md`.
- Every phase must include verification evidence before closure.
- Generated artifacts must not be silently committed as remediation evidence.
- Rollback must be identified before editing production/runtime files.

## Phase 1 - Runtime Stabilization

| Field | Plan |
| --- | --- |
| Objectives | Make the baseline suite runnable without crashes and stop tests from mutating tracked reports. |
| Scope boundaries | Dependency/environment reconciliation, failing test triage, test-output isolation. No API authority redesign yet. |
| Files/modules affected | `pyproject.toml`, `requirements.txt`, `src/risk/garch.py`, `tests/phase2/test_garch_risk.py`, `tests/ml/*benchmark*`, `reports/superseded/risk_tuning_report.md`, `reports/superseded/stress_test_report.md`, `reports/superseded/system_benchmark.md`, `.pytest_cache` handling. |
| Expected risks | Python 3.13 compatibility for `arch`; accidental update of benchmark expected outputs; masking test failures by skipping instead of fixing. |
| Required tests | `python -m pytest tests/phase2/test_garch_risk.py -q`; targeted benchmark/report tests that currently dirty tracked reports; full `python -m pytest tests -q` after targeted fixes. |
| Completion evidence | Full test summary recorded; `git status` proves tests no longer dirty tracked reports; dependency preflight result recorded. |
| Rollback considerations | Revert only Phase 1 dependency/test-output changes if GARCH or report isolation breaks supported environments. Do not revert Phase 0 baseline docs. |
| Dependency ordering | Must complete before broad validation or architecture cleanup because later phases require stable tests. |

## Phase 2 - Validation Hardening

| Field | Plan |
| --- | --- |
| Objectives | Restore explicit contracts for VN100 universe size, feature catalogue versioning, and risk/regime feature gating. |
| Scope boundaries | Validation and contract fixes only. Avoid large rewrites of data loading or model architecture. |
| Files/modules affected | `src/data/universe.py`, `src/data/adapters/vnstock_adapter.py`, `src/ml/feature_engineering.py`, `src/ml/trainer.py`, `tests/test_universe.py`, `tests/test_feature_engineering_vn100.py`, `tests/test_risk_engine.py`. |
| Expected risks | Changing feature names or universe membership can invalidate historical reports; disabling features incorrectly can break trained model manifests. |
| Required tests | `python -m pytest tests/test_universe.py tests/test_feature_engineering_vn100.py tests/test_risk_engine.py -q`; targeted trainer manifest tests; full suite after passing targeted tests. |
| Completion evidence | VN100 count/as-of contract documented; feature catalogue version recorded; risk-disabled training manifest excludes risk/regime columns. |
| Rollback considerations | Keep data artifacts versioned so a bad universe or feature-catalogue change can be rolled back without touching unrelated runtime code. |
| Dependency ordering | Depends on Phase 1 stable test execution. Must complete before governance enforcement claims can be trusted. |

## Phase 3 - Governance Enforcement

| Field | Plan |
| --- | --- |
| Objectives | Align active API and public schemas with diagnostic-only authority boundaries. |
| Scope boundaries | No new recommendation logic and no live trading/execution semantics. Existing legacy behavior must be isolated or explicitly deprecated. |
| Files/modules affected | `src/api/main.py`, `src/api/routes.py`, `src/api/routes_v2.py`, `src/api/schemas.py`, `src/api/schemas_v2.py`, `src/api/ui/*`, `docs/AUTHORITY_BOUNDARY.md`, `docs/governance/*`, API tests. |
| Expected risks | Breaking existing callers that rely on BUY/SELL/execution-shaped payloads; incomplete migration of UI labels. |
| Required tests | API contract tests asserting no active public route emits BUY/SELL/execution authority; `python -m pytest tests/test_api.py -q`; targeted UI/schema tests where available. |
| Completion evidence | API docs/root metadata use diagnostic labels; legacy/demo endpoints are isolated; tests enforce authority boundary. |
| Rollback considerations | Keep a compatibility layer only if clearly namespaced as demo/legacy and excluded from governed runtime. |
| Dependency ordering | Depends on Phase 1 and Phase 2 so governance changes are tested on stable runtime contracts. |

## Phase 4 - Architecture Cleanup

| Field | Plan |
| --- | --- |
| Objectives | Reduce confusion from duplicate entry points, tracked generated dependencies, and experimental modules mixed into active runtime. |
| Scope boundaries | Cleanup must be governed and separately reviewed. No removal of tracked artifacts without explicit file-level justification. |
| Files/modules affected | `scripts/`, `src/api/ui/web/node_modules/**`, `.gitignore`, `src/ml/training_pipeline/*`, `docs/usage/`, `docs/REPOSITORY_STRUCTURE.md`. |
| Expected risks | Removing tracked dependencies can break local front-end workflows; moving scripts can break user runbooks. |
| Required tests | Python full suite; front-end install/build verification if node dependency cleanup is approved; command registry validation. |
| Completion evidence | Canonical command registry exists; generated dependencies reconstructed from lockfiles; experimental modules labeled or isolated. |
| Rollback considerations | Use a dedicated cleanup commit; preserve lockfiles and migration notes; avoid deleting data or reports without owner approval. |
| Dependency ordering | After runtime and governance contracts are stable. |

## Phase 5 - Observability and Monitoring

| Field | Plan |
| --- | --- |
| Objectives | Make runtime provenance, fallback usage, environment assumptions, and artifact generation visible in logs/manifests. |
| Scope boundaries | Add telemetry and manifest fields without changing model behavior unless required for fail-closed audit mode. |
| Files/modules affected | `src/utils/logging.py`, `src/api/tracing.py`, `src/ml/data_loader.py`, `src/ml/trainer.py`, `src/evaluation/*`, `src/reporting/*`, `src/engine/experiment_orchestrator.py`. |
| Expected risks | Excessive logging noise; inconsistent provenance fields across modules; leaking local paths in public API responses. |
| Required tests | Manifest schema tests; fallback/provenance tests; API tracing tests; full suite. |
| Completion evidence | Every governed output records source data type, fallback use, dependency/runtime metadata, and command provenance. |
| Rollback considerations | Observability additions should be additive and easy to disable through config if they break consumers. |
| Dependency ordering | After core contracts are stable enough to define durable provenance fields. |

## Phase 6 - Final Verification and Signoff

| Field | Plan |
| --- | --- |
| Objectives | Prove the repository is stable, auditable, and governed after remediation. |
| Scope boundaries | Verification, documentation, and signoff only; no new feature work. |
| Files/modules affected | `reports/`, `docs/`, test reports, final manifests. |
| Expected risks | Late-stage test flakiness; hidden dirty files; unrecorded environment assumptions. |
| Required tests | `python -m pytest tests -q`; targeted smoke commands from README/runbooks; `git status`; optional dependency preflight and API smoke where services are available. |
| Completion evidence | Clean test summary, clean or explicitly documented dirty tree, updated checklist with verification evidence, signoff note. |
| Rollback considerations | If verification fails, reopen the relevant phase rather than applying untracked fixes in signoff. |
| Dependency ordering | Final phase only; depends on Phases 1 through 5. |

## Cross-Phase Verification Requirements

| Requirement | Evidence required |
| --- | --- |
| Test state | Command, exit code, pass/fail/skip/warning counts, and failure list if nonzero. |
| Git state | `git status` before and after each phase; any generated dirty files listed. |
| Runtime provenance | Python version, dependency preflight, local service assumptions, data source availability. |
| Governance proof | Tests or static checks preventing active API BUY/SELL/execution authority. |
| Rollback proof | Each phase identifies files changed and rollback command strategy before closure. |

## Dependency Ordering Summary

1. Phase 1 must run first because the suite currently fails and mutates reports.
2. Phase 2 depends on stable tests and locks down data/feature contracts.
3. Phase 3 depends on stable contracts and enforces authority boundaries.
4. Phase 4 depends on settled runtime/governance decisions to avoid cleanup churn.
5. Phase 5 adds durable observability after contracts are known.
6. Phase 6 verifies the final governed state and closes remediation.
