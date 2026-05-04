# VSEF Documentation Map
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Architecture note |
| Created / authored | Sunday, 2026-04-26 15:48:16 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-05-05 00:00:00 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `dc76edd78a8b3ba8541a5b3691a345c49ceec67c` |
| Timestamp source | Local deterministic decision-chain documentation refactor |
| Status | Active |

This folder contains project documentation for the private VSEF research repository. The documentation is grouped to make architecture notes, governance notes, audit reports, usage guides, and historical records easier to find.

## Canonical Decision-Chain Docs

Start here for the active deterministic decision-diagnostic chain:

- [System Overview](SYSTEM_OVERVIEW.md)
- [Decision Diagnostic Chain](DECISION_DIAGNOSTIC_CHAIN.md)
- [Authority Boundary](AUTHORITY_BOUNDARY.md)
- [Run Full Decision Chain Smoke](runbooks/RUN_FULL_DECISION_CHAIN_SMOKE.md)
- [Troubleshooting](runbooks/TROUBLESHOOTING.md)

Governance schemas:

- [Pipeline Contracts](governance/PIPELINE_CONTRACTS.md)
- [Quant Core Governance](governance/QUANT_CORE_GOVERNANCE.md)
- [Quant Core Output Schema](governance/QUANT_CORE_OUTPUT_SCHEMA.md)
- [Scenario Output Schema](governance/SCENARIO_OUTPUT_SCHEMA.md)
- [Risk Governance Output Schema](governance/RISK_GOVERNANCE_OUTPUT_SCHEMA.md)
- [Decision Lane Output Schema](governance/DECISION_LANE_OUTPUT_SCHEMA.md)
- [Portfolio Allocator Output Schema](governance/PORTFOLIO_ALLOCATOR_OUTPUT_SCHEMA.md)
- [Phase 3 Router Output Schema](governance/PHASE3_ROUTER_OUTPUT_SCHEMA.md)

Roadmap and audit context:

- [Social Listening Integration](roadmap/SOCIAL_LISTENING_INTEGRATION.md)
- [Decision Physics Next Steps](roadmap/DECISION_PHYSICS_NEXT_STEPS.md)
- [Decision Chain Audit 2026-05-05](audits/DECISION_CHAIN_AUDIT_2026_05_05.md)

## Documentation Timestamp Standard

Active VSEF documentation should include a `Document Metadata` table near the top of the file. The table records document type, created/authored timestamp, last updated timestamp, timezone, branch, commit, timestamp source, and status. The preferred timezone is `Asia/Ho_Chi_Minh / ICT (UTC+07:00)`.

Historical and archived documents should use:

`YYYY-MM-DD_Day__Original_Slug.md`

Active canonical docs may keep stable names to avoid broken references.

## Layout

| Path | Purpose |
| --- | --- |
| `architecture/` | Active architecture maps and codebase-structure notes. |
| `governance/` | Feature governance, context timing, coverage, and interpretability diagnostics. |
| `audits/` | Model gap audits, real-data or cached-data governance audits, and test hardening notes. |
| `reports/` | Polished technical reports that synthesize multiple audits for review or presentation. |
| `usage/` | Focused command references for reproducible local workflows. |
| `roadmap/` | Conservative development roadmap and future research sequencing. |
| `archive/cleanup/` | Historical cleanup, refactor, changelog, and supervisor-summary notes. |
| `archive/phases/` | Historical phase maps, decisions, audits, and phase review notes. |
| `archive/retrieval/` | Superseded retrieval/RAG notes when a retrieval document is no longer active. |
| `archive/vn100/` | Superseded VN100 pipeline notes when usage docs no longer match the active code path. |
| `archive/root/` | Root-level historical or loose documents retained for traceability. |
| `runbooks/` | Reproducible local runbooks and troubleshooting notes. |
| root `docs/*.md` | Stable canonical docs only: this map, system overview, decision-chain map, authority boundary, repository structure, and evaluation workflows. |

## Placement Rules

The root docs folder should contain only active canonical navigation/workflow documents. Historical phase notes, cleanup notes, and loose root notes belong under `docs/archive/`.

- Put new governance policy or diagnostic documentation in `docs/governance/`.
- Put one-off audit reports in `docs/audits/`.
- Put synthesized technical reports in `docs/reports/`.
- Put focused command references in `docs/usage/`.
- Put active operational runbooks in `docs/runbooks/`.
- Put roadmap and future sequencing documents in `docs/roadmap/`.
- Put active architecture and implementation wiring notes in `docs/architecture/`.
- Put superseded, historical, phase, retrieval, VN100, or root-cleanup documents in the matching `docs/archive/` subfolder.
- Prefix archived, historical, audit, report, and summary filenames with `YYYY-MM-DD_Day__` when they are moved into archive or categorized folders.
- Keep generated CSVs, reports, charts, and model artifacts out of `docs/`; use `artifacts/`, `outputs/`, or `tmp/` depending on retention needs.

Recent audit examples include `docs/audits/DECISION_CHAIN_AUDIT_2026_05_05.md`, `docs/audits/VSEF_BROADER_FEATURE_GOVERNANCE_AUDIT.md`, `docs/audits/VSEF_FOREIGN_FLOW_COVERAGE_INVESTIGATION.md`, `docs/audits/VSEF_FOREIGN_FLOW_CURATED_SAMPLE.md`, `docs/audits/VSEF_FOREIGN_FLOW_PROVIDER_CURATION_ATTEMPT.md`, `docs/audits/VSEF_FOREIGN_FLOW_REAL_COVERAGE_AUDIT.md`, `docs/audits/VSEF_10Y_WALKFORWARD_AUDIT.md`, `docs/audits/VSEF_OHLCV_CACHE_10Y_AVAILABILITY_AUDIT.md`, `docs/audits/VSEF_15Y_DAILY_WALKFORWARD_AUDIT.md`, `docs/audits/VSEF_15Y_DAILY_WALKFORWARD_NO_FOREIGN_FLOW_AUDIT.md`, `docs/audits/VSEF_15Y_MULTIHORIZON_WALKFORWARD_AUDIT.md`, `docs/audits/VSEF_15Y_BROADER_TICKER_MULTIHORIZON_AUDIT.md`, `docs/audits/VSEF_SIGNAL_EFFECTIVENESS_BACKTEST.md`, `docs/audits/VSEF_HELDOUT_THRESHOLD_SELECTION.md`, `docs/audits/VSEF_ROLLING_REGIME_THRESHOLD_SELECTION.md`, `docs/audits/VSEF_SIGNAL_REGIME_JOIN.md`, `docs/audits/VSEF_QUANT_CORE_SMOKE_AUDIT.md`, and `docs/audits/VSEF_QUANT_CORE_REPEATED_SEED_STABILITY_AUDIT.md`. The synthesized 15-year technical report is `docs/reports/VSEF_15Y_DAILY_MULTIHORIZON_TECHNICAL_REPORT.md`, and the empirical findings artifact-verification summary is `docs/reports/VSEF_EMPIRICAL_FINDINGS_AND_LIMITATIONS_SUMMARY.md`. Portfolio Allocator v1 governance lives at `docs/governance/PORTFOLIO_ALLOCATOR_V1.md`, deterministic Phase 3 Router v1 governance lives at `docs/governance/PHASE3_ROUTER_V1.md`, and canonical schemas for the active chain live under `docs/governance/`. Foreign-flow artifact curation policy lives at `docs/governance/VSEF_FOREIGN_FLOW_ARTIFACT_POLICY.md`, intentional exclusion is documented in `docs/governance/VSEF_FOREIGN_FLOW_DISABLE_MODE.md`, and foreign-flow audit command examples live at `docs/usage/VSEF_FOREIGN_FLOW_AUDIT_COMMANDS.md`.

This repository is private and proprietary. Documentation placement does not grant permission to copy, redistribute, deploy, or reuse the repository.
