# Code Audit Report

Repository: `K:/Repos/VSEF-PTIT-NEU`
Audit phase: Phase 0 - governed audit baseline and tracking
Audit timestamp: 2026-05-10 15:00:12 +07:00
Audit branch: `audit-remediation-governed-runtime`
Source commit before Phase 0 documentation commit: `56e53bc1d24db9a5d296399702f8419a43793ed9`

## 1. Repository Overview

### Purpose

VSEF is a Vietnamese equity research and decision-diagnostic framework. The active documentation describes a deterministic, diagnostic-only chain that produces forecast, scenario, risk, decision-lane, allocation, and routing diagnostics. The documented authority boundary prohibits BUY/SELL recommendations, live execution instructions, and production trading authority.

### Major Architectural Domains

| Domain | Primary paths | Baseline role |
| --- | --- | --- |
| Governance and contracts | `docs/governance/`, `src/core/` | Output schemas, authority boundaries, analysis packet contracts, and governance metadata. |
| Data and context | `src/data/`, `src/context/`, `src/retrieval/`, `data/` | VN market data adapters, OHLCV loading, context enrichment, retrieval preparation, news and macro context. |
| Forecasting and ML | `src/ml/`, `src/forecast/`, `src/evaluation/` | Feature engineering, model training, walk-forward evaluation, statistical forecasting, benchmark reports, inference utilities. |
| Risk and regime | `src/risk/`, `src/regime/`, `src/risk_governance/`, `src/ml/risk/`, `src/ml/regime/` | VaR/CVaR/GARCH risk, regime detection, risk scoring, risk action diagnostics. |
| Scenario, allocation, routing | `src/scenario/`, `src/allocation/`, `src/portfolio_allocator/`, `src/phase3_router/` | Scenario evaluation, portfolio allocation candidates, Phase 3 route decisions. |
| API and interfaces | `src/api/`, `web/`, `src/api/ui/` | FastAPI app, v1/v2 routes, schemas, streaming consumers, dashboard/UI entry points. |
| Orchestration and research scripts | `scripts/`, `benchmarks/`, `tools/` | CLI runners for quant core, reports, backtests, data sync, audits, and experiments. |
| Infrastructure and config | `pyproject.toml`, `requirements.txt`, `config/`, `configs/`, `alembic.ini`, `infra/docker/` | Python dependencies, experiment configs, local service defaults, database migrations, Docker scaffolding. |

### Runtime Entry Points

Observed runtime surfaces include:

- `python scripts/run_quant_core.py ...` for the governed quant-core chain.
- `python scripts/run_portfolio_allocator.py ...` for allocator diagnostics.
- `python scripts/run_phase3_router.py ...` for route decision artifacts.
- `python scripts/run_experiment.py ...` for config-driven experiment orchestration.
- `python scripts/train_ml_tickers.py ...` and `src/ml/trainer.py` for model training.
- `python scripts/run_backtest_*.py ...`, `scripts/run_walkforward_all_models_stacking_eval.py`, and `src/evaluation/walkforward.py` for backtest and walk-forward research.
- `uvicorn src.api.main:app` or equivalent ASGI launch for the FastAPI service.
- `python -m pytest tests -q` for the full test suite baseline.

### ML, Training, Backtest, and Inference Components

Core ML components are concentrated in:

- `src/ml/data_loader.py`: OHLCV, context, market proxy, fallback, and mock-data loading.
- `src/ml/feature_engineering.py`: technical, VN100, context, risk, regime, delta, and compatibility features.
- `src/ml/trainer.py`: dual-task training, manifests, model feature columns, risk/regime feature handling.
- `src/ml/inference/engine.py`: batch inference and prediction payloads.
- `src/evaluation/walkforward.py`, `src/ml/backtest/`, `src/ml/benchmark/`: walk-forward splits, strategy metrics, stress tests, robustness, tuning.
- `src/ml/statistics/`: bootstrap and Diebold-Mariano statistical helpers.

### API, Services, and Interfaces

The API surface is active and externally visible through:

- `src/api/main.py`: FastAPI app construction, routers, CORS, tracing middleware, root service metadata.
- `src/api/routes.py`: v1 endpoints for prediction, analysis, execution-style decisioning, paper trade, ingestion, and chat.
- `src/api/routes_v2.py`: v2 endpoints with hardcoded/mock prediction and fusion behavior.
- `src/api/schemas.py` and `src/api/schemas_v2.py`: public request/response contracts.
- `src/api/streaming/`: Kafka, session, fallback, and consumer modules.
- `web/` and `src/api/ui/`: dashboard and terminal interfaces.

### Infrastructure Dependencies

Declared dependencies include Python `>=3.11`, pandas/numpy, statsmodels, `arch`, scikit-learn, xgboost, lightgbm, torch, FastAPI, uvicorn, SQLAlchemy, asyncpg, psycopg2, Alembic, Redis, Kafka, ChromaDB, sentence-transformers, OpenAI, and `vnstock_data`.

Observed environment assumptions include local Postgres, Redis, Kafka, ChromaDB/Qdrant-style retrieval, Ollama-compatible local LLM endpoints, and locally available VN market data artifacts. The baseline test run used Python `3.13.5`.

## 2. Current Runtime Status

### Required Branch Action

Command executed:

```powershell
git checkout -b audit-remediation-governed-runtime
```

Result:

- Exit code: `1`
- Outcome: branch creation did not create a new branch because the branch already existed.
- Git message: `fatal: a branch named 'audit-remediation-governed-runtime' already exists`
- Current branch before and after command: `audit-remediation-governed-runtime`
- Governance interpretation: branch requirement is satisfied by pre-existing branch presence, but branch creation itself returned the expected already-exists failure.

### Baseline Test Execution

Command executed:

```powershell
python -m pytest tests -q
```

Result:

- Exit code: `1`
- Passed: `784`
- Failed: `6`
- Skipped: `6`
- Warnings: `30`
- Duration: `261.90s`

Failed tests:

| Test | Failure class | Baseline observation |
| --- | --- | --- |
| `tests/phase2/test_garch_risk.py::test_garch_forecast_interface_returns_vol_and_tail_metrics` | Missing dependency/runtime crash | `src/risk/garch.py` raised `ImportError: GARCHRiskModel requires arch`; active interpreter does not have `arch` importable. |
| `tests/test_context.py::TestNewsCrawler::test_crawl_ticker_mock` | Interface/test drift | Test patches `src.context.news_crawler.Vnstock`, but the wrapper module resolves to `src.data.context.news_crawler` and has no `Vnstock` attribute. |
| `tests/test_feature_engineering_vn100.py::TestVN100Catalogue::test_catalogue_count` | Contract drift | Test expects `len(VN100_DAILY_FEATURES) == 19`; current constant has `41` entries. |
| `tests/test_risk_engine.py::test_trainer_boosters_receive_optional_risk_and_regime_features` | Feature-gating defect | With `risk_enabled=False`, trained CART feature columns still included risk/regime fields such as `var_q`. |
| `tests/test_universe.py::test_get_vn100_universe_default` | Universe contract failure | `get_vn100_universe()` returned `82` tickers; test requires at least `100`. |
| `tests/test_universe.py::test_get_vn100_universe_historical_fallback` | Historical universe fallback failure | Historical request for `2020-01-01` fell back to current universe and returned `82` tickers; test requires at least `100`. |

Warning and environment notes:

- Pytest cache writes failed with access denied under `.pytest_cache/v/cache`.
- Test logs reported `vnstock_data_not_installed` during fallback paths.
- Test logs reported missing `data/market_proxy.csv`.
- The suite modified existing tracked report artifacts during execution.

### Required Git Status After Baseline Tests

Command executed:

```powershell
git status
```

Result after pytest and before Phase 0 document edits:

```text
On branch audit-remediation-governed-runtime
Changes not staged for commit:
  modified:   reports/risk_tuning_report.md
  modified:   reports/stress_test_report.md
  modified:   reports/system_benchmark.md

Untracked files:
  reports/CODE_AUDIT_REPORT.md

no changes added to commit
```

Unstable artifacts:

- `reports/risk_tuning_report.md` was modified by test execution.
- `reports/stress_test_report.md` was modified by test execution.
- `reports/system_benchmark.md` was modified by test execution.
- These test-generated changes are baseline evidence of report mutability and should not be treated as intentional remediation.

### Missing Dependencies and Broken Execution Paths

- `arch` is declared in dependency files but absent from the active Python 3.13.5 interpreter.
- `vnstock_data` appears unavailable in tested fallback paths despite being declared as a canonical dependency.
- VN100 universe fallback currently returns 82 tickers, not a VN100-sized universe.
- The news crawler compatibility wrapper and tests disagree on the public patch target.
- The active API surface still exposes execution and BUY/SELL language despite the documented diagnostic-only boundary.
- Generated reports under `reports/` can be overwritten during tests, weakening audit immutability.

## 3. Audit Findings

### CRITICAL

#### CA-001 - Active API violates the documented diagnostic-only authority boundary

- Severity: CRITICAL
- Affected files: `src/api/routes.py`, `src/api/routes_v2.py`, `src/api/schemas.py`, `src/api/schemas_v2.py`, `src/api/main.py`, `docs/AUTHORITY_BOUNDARY.md`, `README.md`
- Root cause: legacy service contracts and API copy still encode BUY/SELL, execution, and broker-style behavior while governance docs now prohibit recommendations and live execution authority.
- Operational impact: external callers can receive or infer trade/execution semantics from active runtime endpoints, creating a direct governance violation.
- Recommended remediation: isolate legacy execution routes behind an explicit deprecated/demo namespace or convert all active public API contracts to diagnostic-only labels that match governance schemas.
- Complexity estimate: Medium to High

#### CA-002 - Mock and fallback paths can masquerade as real runtime outputs

- Severity: CRITICAL
- Affected files: `src/api/routes.py`, `src/api/routes_v2.py`, `src/ml/data_loader.py`, `src/ml/backtest/paper.py`, `src/engine/agents/technical_agent.py`
- Root cause: resilience/demo fallbacks generate synthetic data or hardcoded decisions without a consistently enforced audit-mode failure policy.
- Operational impact: downstream diagnostics can be produced from synthetic or placeholder data while appearing operationally valid, creating silent data provenance corruption.
- Recommended remediation: add an audit/research runtime mode that fails closed on missing real data, requires explicit provenance fields, and prevents synthetic fallback from entering governed outputs.
- Complexity estimate: High

#### CA-003 - Risk/regime feature disable path leaks disabled columns into training features

- Severity: CRITICAL
- Affected files: `src/ml/trainer.py`, `src/ml/feature_engineering.py`, `tests/test_risk_engine.py`
- Root cause: `risk_enabled=False` does not fully prevent risk/regime columns from entering booster feature manifests when other risk flags are present.
- Operational impact: model manifests can claim a disabled risk configuration while trained feature columns still include risk/regime inputs, invalidating experiment comparison and audit claims.
- Recommended remediation: define a single authoritative feature gate for risk/regime columns and add manifest assertions for disabled-feature exclusion.
- Complexity estimate: Medium

#### CA-004 - GARCH runtime path crashes in the active interpreter

- Severity: CRITICAL
- Affected files: `src/risk/garch.py`, `tests/phase2/test_garch_risk.py`, `pyproject.toml`, `requirements.txt`
- Root cause: the active Python environment does not provide the declared `arch` dependency.
- Operational impact: GARCH-based risk forecasts crash at runtime, blocking risk metrics and any downstream chain relying on volatility/tail-risk outputs.
- Recommended remediation: reconcile dependency installation with supported Python versions, then add environment verification before risk model execution.
- Complexity estimate: Low to Medium

### HIGH

#### CA-005 - VN100 universe fallback returns an undersized universe

- Severity: HIGH
- Affected files: `src/data/universe.py`, `src/data/adapters/vnstock_adapter.py`, `tests/test_universe.py`
- Root cause: live provider fallback resolves to a static/listing fallback with 82 tickers and historical universe requests fall back to the same current set.
- Operational impact: backtests, feature generation, and universe-level diagnostics can run on an incomplete universe while still labeled as VN100.
- Recommended remediation: establish a versioned VN100 universe artifact, validate minimum count and as-of-date semantics, and fail closed when the universe contract cannot be met.
- Complexity estimate: Medium

#### CA-006 - News crawler compatibility surface has drifted from tests

- Severity: HIGH
- Affected files: `src/context/news_crawler.py`, `src/data/context/news_crawler.py`, `tests/test_context.py`
- Root cause: public wrapper module re-exports the implementation but no longer exposes the legacy `Vnstock` patch target expected by tests.
- Operational impact: news/context ingestion tests fail, and caller assumptions about the public import surface are unstable.
- Recommended remediation: choose one canonical news crawler import contract, update tests and wrappers together, and document provider injection points.
- Complexity estimate: Low to Medium

#### CA-007 - VN100 feature catalogue contract has drifted from tests

- Severity: HIGH
- Affected files: `src/ml/feature_engineering.py`, `tests/test_feature_engineering_vn100.py`
- Root cause: `VN100_DAILY_FEATURES` expanded to 41 entries while the test contract still asserts 19.
- Operational impact: feature-set comparability, historical reports, and downstream training manifests can become ambiguous across phases.
- Recommended remediation: define a versioned feature catalogue with migration notes and update tests to assert named groups rather than stale total count.
- Complexity estimate: Low to Medium

#### CA-008 - Baseline tests mutate tracked report artifacts

- Severity: HIGH
- Affected files: `reports/risk_tuning_report.md`, `reports/stress_test_report.md`, `reports/system_benchmark.md`, related tests under `tests/ml/`
- Root cause: report-generation tests write to fixed tracked report paths or rewrite committed report outputs as side effects.
- Operational impact: the repository cannot maintain immutable report baselines if validation commands rewrite historical evidence.
- Recommended remediation: redirect test report writes to isolated temporary directories and add checks preventing tests from dirtying tracked artifacts.
- Complexity estimate: Medium

#### CA-009 - Reproducibility depends on undeclared local services and local files

- Severity: HIGH
- Affected files: `config/settings.py`, `alembic.ini`, `scripts/*`, `src/ml/data_loader.py`, `src/api/streaming/*`
- Root cause: defaults assume local Postgres, Redis, Kafka, ChromaDB/Qdrant-like retrieval, Ollama, market proxy artifacts, and provider availability.
- Operational impact: another environment cannot reproduce results from repository metadata alone.
- Recommended remediation: create a deterministic environment contract, preflight command, and manifest fields for service availability, data source provenance, and package versions.
- Complexity estimate: Medium

#### CA-010 - API v2 exposes hardcoded/mock predictions and portfolio sizing

- Severity: HIGH
- Affected files: `src/api/routes_v2.py`, `src/api/schemas_v2.py`
- Root cause: demo scaffolding remained in active route handlers.
- Operational impact: API consumers can mistake hardcoded forecast/fusion payloads and mocked portfolio assumptions for real governed inference.
- Recommended remediation: move mock endpoints behind an explicit demo flag/namespace or replace them with governed inference calls that emit diagnostic-only payloads.
- Complexity estimate: Medium

### MEDIUM

#### CA-011 - Front-end dependency artifacts are tracked in git

- Severity: MEDIUM
- Affected files: `src/api/ui/web/node_modules/**`, `.gitignore`
- Root cause: `node_modules` artifacts are tracked despite being generated dependency material.
- Operational impact: repository size, diffs, supply-chain review, and audit noise are materially increased.
- Recommended remediation: plan a governed cleanup phase that removes tracked dependency artifacts and enforces dependency reconstruction through lockfiles.
- Complexity estimate: Medium

#### CA-012 - Overlapping orchestration scripts obscure the canonical runtime path

- Severity: MEDIUM
- Affected files: `scripts/`, `src/ml/benchmark/`, `src/evaluation/`, `docs/usage/`
- Root cause: many legacy, research, benchmark, report, and production-like scripts coexist without a single audited command matrix.
- Operational impact: operators can run noncanonical paths and produce outputs that look comparable but have different assumptions.
- Recommended remediation: publish a command registry that marks each runner as canonical, legacy, demo, smoke, or research-only.
- Complexity estimate: Medium

#### CA-013 - FastAPI service metadata and debug output conflict with governance language

- Severity: MEDIUM
- Affected files: `src/api/main.py`
- Root cause: service title, description, root endpoints, and debug `print` statements retain algorithmic-trading/execution framing.
- Operational impact: service startup and docs reinforce the wrong authority model and leak environment details to stdout.
- Recommended remediation: align API metadata with diagnostic-only governance and replace debug prints with governed logging or remove them.
- Complexity estimate: Low

#### CA-014 - Experimental training branches are mixed into active source

- Severity: MEDIUM
- Affected files: `src/ml/training_pipeline/train_tft.py`, `src/ml/training_pipeline/train_rl_allocator.py`, `scripts/train_ppo_allocator.py`
- Root cause: placeholder and experimental model workflows are committed beside mature runtime components.
- Operational impact: unsupported capabilities may be mistaken for governed runtime support.
- Recommended remediation: label experimental modules explicitly, move them to a research namespace, or add tests and governance manifests before promotion.
- Complexity estimate: Low to Medium

### LOW

#### CA-015 - Documentation and encoding hygiene issues reduce readability

- Severity: LOW
- Affected files: `.gitignore`, historical docs/reports, root-level malformed filename observed in file listing
- Root cause: mixed encodings and accidental local-shell artifact naming entered repository history.
- Operational impact: lower audit readability and increased chance of misinterpreting repository structure.
- Recommended remediation: perform a governed documentation and filename hygiene pass after runtime stabilization.
- Complexity estimate: Low

#### CA-016 - Reports directory has mixed canonical and historical meanings

- Severity: LOW
- Affected files: `README.md`, `reports/`, `docs/`
- Root cause: current README states `docs/` are canonical and `reports/` are historical snapshots, while this Phase 0 workflow requires governance baselines under `reports/`.
- Operational impact: future maintainers may be unsure whether `reports/CODE_AUDIT_REPORT.md` is canonical or historical.
- Recommended remediation: add a short governance note clarifying that these three Phase 0 audit files are controlled remediation records.
- Complexity estimate: Low

## 4. Governance Risks

| Risk class | Baseline risk | Evidence |
| --- | --- | --- |
| Undocumented runtime behavior | Active routes can emit execution-style responses despite diagnostic-only documentation. | `src/api/routes.py`, `src/api/routes_v2.py`, `src/api/schemas*.py` |
| Hidden coupling | API, data loader, paper backtest, and model training rely on implicit local services and fallback files. | `config/settings.py`, `alembic.ini`, `src/ml/data_loader.py` |
| Mutable state risks | Tests modify tracked report artifacts and cannot be treated as read-only validation. | `git status` after pytest |
| Unsafe fallback behavior | Synthetic/mock data and hardcoded outputs can flow through operationally named endpoints. | `src/api/routes.py`, `src/api/routes_v2.py`, `src/ml/backtest/paper.py` |
| Non-deterministic execution | Provider availability, missing local artifacts, cache write failures, and local services change behavior. | pytest logs, settings defaults, `.pytest_cache` warning |
| Auditability weaknesses | Many scripts produce reports/artifacts without a single canonical command registry. | `scripts/`, `src/ml/benchmark/`, `reports/` |
| Environment drift | Declared dependencies and active interpreter differ; Python 3.13.5 lacks `arch`. | pytest failure, `python --version` |
| Data scope ambiguity | VN100 label can resolve to 82 tickers and historical universe requests use current fallback. | `tests/test_universe.py` failures |

## 5. Baseline Snapshot

| Item | Baseline value |
| --- | --- |
| Branch | `audit-remediation-governed-runtime` |
| Branch creation command | `git checkout -b audit-remediation-governed-runtime` |
| Branch creation result | Failed because branch already existed; current branch already matched target. |
| Source commit before Phase 0 docs | `56e53bc1d24db9a5d296399702f8419a43793ed9` |
| Python runtime | `Python 3.13.5` |
| Test command | `python -m pytest tests -q` |
| Test result | Exit `1`; `6 failed, 784 passed, 6 skipped, 30 warnings in 261.90s` |
| Git status after tests | Modified `reports/risk_tuning_report.md`, `reports/stress_test_report.md`, `reports/system_benchmark.md`; untracked `reports/CODE_AUDIT_REPORT.md`. |
| Missing dependencies observed | `arch`; `vnstock_data` unavailable in tested fallback logs. |
| Environment assumptions | Windows PowerShell, local repo path `K:/Repos/VSEF-PTIT-NEU`, local service defaults for Postgres/Redis/Kafka/Ollama/retrieval, missing `data/market_proxy.csv`. |
| Unresolved blockers | Six failing tests, test-generated report mutations, active API governance boundary violation, incomplete VN100 universe fallback, unsafe mock/fallback behavior. |

## Phase 0 Scope Boundary

No runtime fixes, dependency installation, refactors, or cleanup were performed in this phase. This report establishes the audit baseline for governed remediation only.
