# VSEF Repository Structure

Date: 2026-04-26

Branch: `vsef-repository-structure-cleanup`

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

- `docs/governance/`: governance and diagnostic policy notes
- `docs/audits/`: audit reports and validation notes
- `docs/roadmap/`: conservative development sequencing
- `docs/archive/`: historical documents and superseded notes
- root `docs/*.md`: active architecture, workflow, usage, and research-limitations guides

See `docs/README.md` for the documentation map.

## Data, Artifacts, Outputs, And Tmp

`data/` contains local/cache market and context data. A large amount of curated CSV cache data is currently tracked, so this cleanup does not globally untrack `data/`.

`artifacts/`, `outputs/`, `models/`, and `tmp/` are generated or local runtime locations and should not receive new tracked files by default. Existing tracked generated files should be removed only when they are clearly reproducible or temporary and no tests depend on them.

Daily OHLCV cache coverage can be audited with `scripts/audit_ohlcv_cache_coverage.py`. New or refreshed large OHLCV cache files should be staged in ignored scratch space and validated before replacing tracked files under `data/daily_market_split_data/`.

Foreign-flow artifacts require separate governance before interpretation. See `docs/governance/VSEF_FOREIGN_FLOW_ARTIFACT_POLICY.md` for schema, provenance, and fixture-vs-curated artifact rules.

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
