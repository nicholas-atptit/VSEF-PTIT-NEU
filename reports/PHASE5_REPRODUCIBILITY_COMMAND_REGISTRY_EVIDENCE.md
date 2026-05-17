# Phase 5 Reproducibility And Command Registry Evidence

Date: 2026-05-11
Scope: environment/service/data preflight, canonical command registry, and reports governance clarification

## Summary

Phase 5 defines explicit local reproducibility and operator contracts without changing runtime forecasting, API governance semantics, model behavior, allocator logic, VN100 logic, feature catalogue logic, repository hygiene rules, or frontend removal work.

## Pre-Edit Inventory Summary

Major runner scripts were classified into:

- canonical runtime: `run_quant_core.py`, `run_portfolio_allocator.py`, `run_phase3_router.py`, `run_analysis_feed.py`, retrieval prep/ingest/query, and `run_experiment.py`
- smoke/test: preflight, hygiene, pytest, DB validation, ML baseline, and stability checks
- research: backtests, walk-forward/robustness, forecast rehabilitation, candidate research, report generation, and meta-selector analysis
- legacy/deprecated: `scripts/legacy/**` plus older root runners such as `run_agent_decision.py`, `per_session_predict.py`, and ad hoc historical extractors
- maintenance/data ops: sync, backdate, streaming/consumer, service bootstrap, provider curation, local cache auditing, and CSV/data preparation scripts

Required local assumptions identified:

- services: database, Redis, Kafka, Chroma/vector store, local LLM/Ollama, and provider availability
- artifacts: local market data caches, market/context proxy files, model bundles, generated `artifacts/**`, and report output packs
- reports: controlled remediation records vs historical/generated snapshots

## Preflight Output Summary

Command:

```powershell
python scripts/check_runtime_preflight.py
```

Result:

```text
SUMMARY: ok=40 warn=20 fail=0
```

Important warnings recorded:

- `asyncpg`, `psycopg2`, and `vnstock_data` were not importable in this local environment.
- Optional imports `dnse`, `sentence-transformers`, `gymnasium`, `stable-baselines3`, and `google-generativeai` were not importable.
- Database, Redis, Kafka, Chroma, and Ollama endpoints were configured at localhost defaults but not reachable.
- `data/market_proxy.csv` was absent.
- `TIMESCALE_URL`, `REDIS_URL`, `KAFKA_BROKER_URL`, `OPENAI_API_KEY`, `GROQ_API_KEY`, and `GEMINI_API_KEY` were unset.

The script exits successfully when checks complete and no hard Python-version failure occurs. Missing services and optional/local-only dependencies are explicit warnings rather than silent assumptions.

## Command Registry Summary

Created `docs/COMMAND_REGISTRY.md`.

The registry defines:

- canonical verification commands
- canonical diagnostic-chain runtime commands
- service bootstrap/runtime operations
- data operations
- research-only commands
- legacy/deprecated commands
- output locations
- whether commands may generate ignored artifacts
- whether commands require external services

Canonical operator entrypoints now include:

- `python scripts/check_runtime_preflight.py`
- `python scripts/check_repo_hygiene.py`
- `python -m pytest tests -q`
- the governed `run_quant_core.py` smoke command
- allocator/router/analysis-feed/retrieval commands that operate on saved artifacts

## Reports Governance Summary

Created `docs/REPORTS_GOVERNANCE.md`.

The document clarifies:

- `reports/AUDIT_REMEDIATION_CHECKLIST.md` and Phase 2/3/4A/4/5 evidence files are controlled remediation records.
- historical benchmark, audit, report-pack, chart, CSV, and decision-card outputs remain historical/generated snapshots unless explicitly promoted.
- `reports/` content does not override active canonical docs under `docs/` except for remediation status/evidence questions involving controlled remediation records.
- `reports/RESEARCH_TOPIC_AND_CODE_AUDIT_REPORT.md` remains intentionally untracked and untouched.

README was updated only to link the command registry and reports governance docs and to clarify the controlled remediation-record exception.

## Verification

| Command | Result |
| --- | --- |
| `python scripts/check_runtime_preflight.py` | `ok=40 warn=20 fail=0` |
| `python scripts/check_repo_hygiene.py` | `Repository hygiene check passed.` |
| `python -m pytest tests -q` | `815 passed, 5 skipped, 33 warnings in 448.17s` |

## Git Status

Pre-commit status after implementation and verification:

```text
 M README.md
 M reports/AUDIT_REMEDIATION_CHECKLIST.md
?? docs/COMMAND_REGISTRY.md
?? docs/REPORTS_GOVERNANCE.md
?? reports/PHASE5_REPRODUCIBILITY_COMMAND_REGISTRY_EVIDENCE.md
?? reports/RESEARCH_TOPIC_AND_CODE_AUDIT_REPORT.md
?? scripts/check_runtime_preflight.py
```

Final post-commit status observed after the Phase 5 commit:

```text
?? reports/RESEARCH_TOPIC_AND_CODE_AUDIT_REPORT.md
```

## Unresolved Service/Data Limitations

- Local service endpoints are documented but not reachable in this environment: database, Redis, Kafka, Chroma, and Ollama.
- `vnstock_data` is the canonical provider but is not importable in this environment.
- DB drivers `asyncpg` and `psycopg2` are not importable in this environment.
- `data/market_proxy.csv` is absent and should be generated with `python scripts/compute_market_proxy.py` when market-proxy workflows require it.
- Optional RL/cloud/embedding packages are absent and remain warnings unless their associated workflows are invoked.
