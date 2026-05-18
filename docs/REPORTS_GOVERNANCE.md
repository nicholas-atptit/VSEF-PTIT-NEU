# Reports Governance

Status: canonical reports-status clarification
Last updated: 2026-05-11

The repository uses `reports/` for two different kinds of files:

- governed remediation records created by the audit-remediation workflow
- historical benchmark, research, smoke, audit, chart, and report-pack snapshots

This document resolves that ambiguity. It does not rewrite historical report
content and does not promote historical benchmark reports to current runtime
authority.

## Canonical Remediation Records

The following `reports/` files are controlled remediation records for the current
audit-remediation branch:

- `reports/cleanup/AUDIT_REMEDIATION_CHECKLIST.md`
- `reports/cleanup/PHASE2_MOCK_PROVENANCE_EVIDENCE.md`
- `reports/cleanup/PHASE3_REPOSITORY_HYGIENE_EVIDENCE.md`
- `reports/cleanup/PHASE4A_FRONTEND_DESCOPE_EVIDENCE.md`
- `reports/cleanup/PHASE4_API_GOVERNANCE_EVIDENCE.md`
- `reports/cleanup/PHASE5_REPRODUCIBILITY_COMMAND_REGISTRY_EVIDENCE.md`

These files are canonical only for remediation status, verification evidence,
rollback notes, and phase closure. They are not schema authority and do not
override active docs under `docs/governance/`.

Phase 0 records in `reports/`, including the checklist and remediation plan,
are controlled because the remediation workflow needs stable evidence that can
be reviewed and committed. Phase 2, Phase 3, Phase 4A, Phase 4, and Phase 5
evidence files are controlled for the same reason: they record scoped changes,
verification commands, and unresolved risks.

## Historical Evidence And Benchmark Reports

The following report groups are historical evidence snapshots unless a future
task explicitly promotes or curates a specific file:

- `reports/forecasting_core/`
- `reports/robustness/`
- `reports/risk_aware/`
- `reports/regime_analysis/`
- `reports/feature_analysis/`
- `reports/audits/`
- `reports/vsef_v1_executive_pack/`
- `reports/final_supervisor_package/`
- `reports/repeated_seed_1000_smoke_report_pack/`
- top-level benchmark/smoke reports such as `system_benchmark.md`,
  `stress_test_report.md`, `full_system_report.md`, `ml_benchmark.csv`,
  `pilot_benchmark.csv`, `smoke_cart.csv`, `smoke_lstm.csv`,
  `backtest_portfolio_result.json`, and comparison reports

These files may be useful for traceability, empirical interpretation, or prior
experiment review. They are not current command authority, API contract
authority, model-governance authority, or runtime-readiness proof by themselves.

## Generated Outputs

Generated outputs should remain untracked unless explicitly curated:

- new report CSVs, charts, JSONL files, and report packs
- new decision cards under `reports/decision_cards/`
- benchmark artifacts emitted by report-generation scripts
- local model, artifact, cache, and service-state outputs

If a generated output is promoted to a controlled record, the promoting task
must state why the file is stable, what command produced it, and what rollback
or regeneration rule applies.

## Relationship To `docs/`

Active canonical behavior, schemas, and operator workflows live in `docs/`,
especially:

- `docs/COMMAND_REGISTRY.md`
- `docs/REPORTS_GOVERNANCE.md`
- `docs/AUTHORITY_BOUNDARY.md`
- `docs/DECISION_DIAGNOSTIC_CHAIN.md`
- `docs/governance/PIPELINE_CONTRACTS.md`
- active output schema documents under `docs/governance/`

When `reports/` content conflicts with active `docs/` content, active `docs/`
win unless the report is one of the controlled remediation records listed above
and the question is specifically about remediation status or evidence.

## Untracked Audit Report

`reports/results/RESEARCH_TOPIC_AND_CODE_AUDIT_REPORT.md` is intentionally left
untracked in the current workspace unless a future task explicitly requests that
file. Phase 5 does not edit, stage, or commit it.
