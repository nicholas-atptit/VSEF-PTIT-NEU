# VSEF Repository Structure
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Architecture note |
| Created / authored | Sunday, 2026-04-26 15:48:16 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-04-28 22:51:14 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `ef20ce73b466d75a61ca4768d4f4129405df7fb0` |
| Timestamp source | Git history |
| Status | Active |

## Purpose

This document records the intended repository layout for VSEF. The goal is to keep source code, tests, documentation, generated artifacts, and local data separated without disrupting existing research workflows.

VSEF is private and proprietary. This structure document does not grant any right to copy, distribute, deploy, or reuse the repository.

## Root Files

Root files should be limited to project entry points and core metadata:

- `README.md`: project positioning and primary navigation
- `LICENSE`: proprietary all-rights-reserved license notice
- `SECURITY.md`: private security reporting policy
- `pyproject.toml`: Python package metadata and dependencies
- `requirements.txt`: environment dependency snapshot
- `.gitignore`: generated-file and local-cache exclusion rules
- `alembic.ini`, `conftest.py`, and core config files needed by tooling

Historical root markdown files should live under `docs/archive/root/`.

## Source Code

`src/` contains importable project code. Keep model, feature, risk, regime, reporting, API, and evaluation modules under their existing package boundaries unless a separate refactor explicitly updates imports and tests.

Do not place generated outputs, notebooks, local caches, or ad hoc scratch scripts inside `src/`.

## Scripts

`scripts/` contains runnable CLI entry points. The current script surface is broad, and this cleanup does not move scripts to avoid breaking documented commands.

Preferred future grouping:

- `scripts/audit/`: audit and verification runners
- `scripts/backtest/`: backtest, walk-forward, benchmark, and evaluation runners
- `scripts/data/`: ingestion, sync, extraction, and cache preparation scripts
- `scripts/tools/`: one-off utility scripts
- `scripts/legacy/`: superseded scripts retained for traceability

Before moving scripts, update README commands, docs commands, tests, and any import assumptions.

## Tests

`tests/` contains automated tests. Active tests should stay under domain folders such as:

- `tests/ml/`
- `tests/quant_core/`
- `tests/phase1/`
- `tests/tools/`

Broken or excluded test files should not remain mixed with active tests. If a test is not runnable but should be retained for context, move it to `tests/archive/` and document why it is excluded.

## Documentation

`docs/` is grouped as follows:

- `docs/architecture/`: active architecture maps and codebase-structure notes
- `docs/governance/`: governance and diagnostic policy notes
- `docs/audits/`: audit reports and validation notes
- `docs/reports/`: polished technical reports and empirical summaries
- `docs/usage/`: command and workflow reference guides
- `docs/roadmap/`: conservative development sequencing
- `docs/archive/cleanup/`: historical cleanup, refactor, changelog, and summary notes
- `docs/archive/phases/`: historical phase maps, decisions, audits, and phase review notes
- `docs/archive/retrieval/`: superseded retrieval/RAG notes
- `docs/archive/vn100/`: superseded VN100 pipeline notes
- `docs/archive/root/`: root-level historical or loose documents retained for traceability
- root `docs/*.md`: stable canonical docs only

See `docs/README.md` for the documentation map.

Historical and archived documents should use `YYYY-MM-DD_Day__Original_Slug.md`. Active canonical docs may keep stable names to avoid broken references.

The root docs folder should contain only active canonical navigation/workflow documents. Historical phase notes, cleanup notes, and loose root notes belong under `docs/archive/`.

## Data, Artifacts, Outputs, And Tmp

`data/` contains local/cache market and context data. A large amount of curated CSV cache data is currently tracked, so this cleanup does not globally untrack `data/`.

`artifacts/`, `outputs/`, `models/`, and `tmp/` are generated or local runtime locations and should not receive new tracked files by default. Existing tracked generated files should be removed only when they are clearly reproducible or temporary and no tests depend on them.

Daily OHLCV cache coverage can be audited with `scripts/audit_ohlcv_cache_coverage.py`. New or refreshed large OHLCV cache files should be staged in ignored scratch space and validated before replacing tracked files under `data/daily_market_split_data/`. The all-model walk-forward runner can read staged per-ticker OHLCV files directly with `--ohlcv-data-dir`, which avoids overwriting tracked cache files for empirical audits such as `docs/audits/VSEF_15Y_DAILY_WALKFORWARD_AUDIT.md`.

Foreign-flow artifacts require separate governance before interpretation. See `docs/governance/VSEF_FOREIGN_FLOW_ARTIFACT_POLICY.md` for schema, provenance, and fixture-vs-curated artifact rules. For audits that intentionally exclude foreign-flow because no governed long-window artifact exists, use the documented `--foreign-flow-mode disabled` workflow in `docs/governance/VSEF_FOREIGN_FLOW_DISABLE_MODE.md`.

## Tracking Rules

Track:

- source code
- tests
- curated documentation
- small configuration and schema files
- intentionally curated sample data only when necessary for tests or reproducible examples

Do not track:

- `__pycache__/`
- `.pytest_cache/`
- ad hoc `tmp/` output
- generated walk-forward outputs
- model training artifacts
- local logs
- backup files such as `*.bak`, `*.tmp`, or `*.broken`

## Cleanup Notes

This cleanup intentionally avoids large script moves and broad `data/` changes. Those areas need separate review because many commands and tests reference established paths.
