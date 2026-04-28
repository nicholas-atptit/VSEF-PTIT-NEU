# VSEF Documentation Map

## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Documentation map |
| Created / authored | Sunday, 2026-04-26 15:48:16 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-04-28 21:37:10 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | not specified |
| Commit | `1bbb75a793c5ea30a26ffef7860aadd06306a640` |
| Timestamp source | Git history |

This folder contains project documentation for the private VSEF research repository. The documentation is grouped to make governance notes, audit reports, usage guides, and historical records easier to find.

## Documentation Timestamp Standard

Active VSEF documentation should include a `Document Metadata` table near the top of the file. The table records document type, created/authored timestamp, last updated timestamp, timezone, branch, commit, and timestamp source. The preferred timezone is `Asia/Ho_Chi_Minh / ICT (UTC+07:00)`.

## Layout

| Path | Purpose |
| --- | --- |
| `governance/` | Feature governance, context timing, coverage, and interpretability diagnostics. |
| `audits/` | Model gap audits, real-data or cached-data governance audits, and test hardening notes. |
| `reports/` | Polished technical reports that synthesize multiple audits for review or presentation. |
| `usage/` | Focused command references for reproducible local workflows. |
| `roadmap/` | Conservative development roadmap and future research sequencing. |
| `archive/` | Historical notes, root-level legacy documents, and prompt-run records retained for traceability. |
| root `docs/*.md` | Active architecture, usage, workflow, and research-limitations guides that still have broad references. |

## Placement Rules

- Put new governance policy or diagnostic documentation in `docs/governance/`.
- Put one-off audit reports in `docs/audits/`.
- Put synthesized technical reports in `docs/reports/`.
- Put focused command references in `docs/usage/`.
- Put roadmap and future sequencing documents in `docs/roadmap/`.
- Put superseded, historical, or root-cleanup documents in `docs/archive/`.
- Keep generated CSVs, reports, charts, and model artifacts out of `docs/`; use `artifacts/`, `outputs/`, or `tmp/` depending on retention needs.

Recent audit examples include `docs/audits/VSEF_BROADER_FEATURE_GOVERNANCE_AUDIT.md`, `docs/audits/VSEF_FOREIGN_FLOW_COVERAGE_INVESTIGATION.md`, `docs/audits/VSEF_FOREIGN_FLOW_CURATED_SAMPLE.md`, `docs/audits/VSEF_FOREIGN_FLOW_PROVIDER_CURATION_ATTEMPT.md`, `docs/audits/VSEF_FOREIGN_FLOW_REAL_COVERAGE_AUDIT.md`, `docs/audits/VSEF_10Y_WALKFORWARD_AUDIT.md`, `docs/audits/VSEF_OHLCV_CACHE_10Y_AVAILABILITY_AUDIT.md`, `docs/audits/VSEF_15Y_DAILY_WALKFORWARD_AUDIT.md`, `docs/audits/VSEF_15Y_DAILY_WALKFORWARD_NO_FOREIGN_FLOW_AUDIT.md`, `docs/audits/VSEF_15Y_MULTIHORIZON_WALKFORWARD_AUDIT.md`, `docs/audits/VSEF_15Y_BROADER_TICKER_MULTIHORIZON_AUDIT.md`, `docs/audits/VSEF_SIGNAL_EFFECTIVENESS_BACKTEST.md`, `docs/audits/VSEF_HELDOUT_THRESHOLD_SELECTION.md`, `docs/audits/VSEF_ROLLING_REGIME_THRESHOLD_SELECTION.md`, and `docs/audits/VSEF_SIGNAL_REGIME_JOIN.md`. The synthesized 15-year technical report is `docs/reports/VSEF_15Y_DAILY_MULTIHORIZON_TECHNICAL_REPORT.md`. Foreign-flow artifact curation policy lives at `docs/governance/VSEF_FOREIGN_FLOW_ARTIFACT_POLICY.md`, intentional exclusion is documented in `docs/governance/VSEF_FOREIGN_FLOW_DISABLE_MODE.md`, and foreign-flow audit command examples live at `docs/usage/VSEF_FOREIGN_FLOW_AUDIT_COMMANDS.md`.

This repository is private and proprietary. Documentation placement does not grant permission to copy, redistribute, deploy, or reuse the repository.
