# VSEF Documentation Map

This folder contains project documentation for the private VSEF research repository. The documentation is grouped to make governance notes, audit reports, usage guides, and historical records easier to find.

## Layout

| Path | Purpose |
| --- | --- |
| `governance/` | Feature governance, context timing, coverage, and interpretability diagnostics. |
| `audits/` | Model gap audits, real-data or cached-data governance audits, and test hardening notes. |
| `roadmap/` | Conservative development roadmap and future research sequencing. |
| `archive/` | Historical notes, root-level legacy documents, and prompt-run records retained for traceability. |
| root `docs/*.md` | Active architecture, usage, workflow, and research-limitations guides that still have broad references. |

## Placement Rules

- Put new governance policy or diagnostic documentation in `docs/governance/`.
- Put one-off audit reports in `docs/audits/`.
- Put roadmap and future sequencing documents in `docs/roadmap/`.
- Put superseded, historical, or root-cleanup documents in `docs/archive/`.
- Keep generated CSVs, reports, charts, and model artifacts out of `docs/`; use `artifacts/`, `outputs/`, or `tmp/` depending on retention needs.

Recent audit examples include `docs/audits/VSEF_BROADER_FEATURE_GOVERNANCE_AUDIT.md`, `docs/audits/VSEF_FOREIGN_FLOW_COVERAGE_INVESTIGATION.md`, and `docs/audits/VSEF_FOREIGN_FLOW_CURATED_SAMPLE.md`. Foreign-flow artifact curation policy lives at `docs/governance/VSEF_FOREIGN_FLOW_ARTIFACT_POLICY.md`.

This repository is private and proprietary. Documentation placement does not grant permission to copy, redistribute, deploy, or reuse the repository.
