# Phase 8 Governed Integration Testing Evidence

Date: 2026-05-11
Branch: `audit-remediation-governed-runtime`

## Scope

Phase 8 adds integration coverage for the governed diagnostic chain:

`data -> feature generation -> forecast summary -> risk summary -> allocation candidate -> router diagnostic`

The phase is integration-test only. It does not modify production model logic,
forecast algorithm behavior, allocator logic, router logic, API governance
semantics, benchmark acceptance policy, command registry, runtime path registry,
repository hygiene rules, or frontend removal work.

## Pre-Edit Inventory

Reviewed entrypoints and existing tests:

- `scripts/run_quant_core.py`: best governed chain orchestrator. It already
  coordinates quant core, scenario engine, risk governance, Decision Lane,
  Portfolio Allocator v1, Phase 3 Router v1, and run-manifest writing.
- `src/evaluation/quant_core.py`: stable in-process quant-core scenario helper
  used by the runner; it builds forecasts, risk, regime, policy, and summaries.
- `scripts/run_portfolio_allocator.py`: standalone allocator CLI, but it points
  at the legacy `src.allocation.portfolio_allocator` path rather than the
  canonical `src.portfolio_allocator` path used by `scripts/run_quant_core.py`.
- `scripts/run_phase3_router.py`: standalone router CLI, useful for from-file
  router checks, but not needed because `scripts/run_quant_core.py` already
  writes canonical router artifacts.
- Existing smoke/integration tests:
  - `tests/risk_governance/test_quant_core_risk_governance_integration.py`
  - `tests/portfolio_allocator/test_quant_core_allocator_integration.py`
  - `tests/phase3_router/test_quant_core_router_integration.py`
  - `tests/quant_core/test_portfolio_allocator.py`
  - `tests/quant_core/test_phase3_router.py`

## Invocation Classification

The integration test calls Python functions directly, with `scripts.run_quant_core.main()`
as the orchestrator and a patched `sys.argv` command vector. This preserves the
command/artifact contract while allowing a temporary deterministic fixture data
directory to be injected without subprocess-only environment plumbing or
production code edits.

Subprocess CLI was not selected because the governed chain needed deterministic
temporary fixture data and the current quant-core CLI has no data-dir argument.
The test still exercises the same runner flags and artifact-writing path used by
the CLI.

## Test Added

Added:

- `tests/integration/test_governed_diagnostic_chain.py`

The test uses:

- deterministic OHLCV fixture data for ticker `P8A`
- `FeatureEngineer().transform(..., build_mode="fast_core_mode")`
- temporary prepared-feature and output directories
- `research_core` run mode
- `random_forest` as the governed forecast model
- scenario engine enabled
- risk governance enabled
- Portfolio Allocator v1 enabled
- Phase 3 Router v1 enabled

## Fixture And Provenance

Fixture source:

- `phase8_deterministic_feature_fixture`

Fixture provenance asserted:

- `source == "phase8_deterministic_feature_fixture"`
- `uses_mock_data is False`
- `fallback_triggered is False`
- `runtime_mode == "research"`
- selected governed forecast feature count is nonzero

Mock fallback policy asserted:

- audit mode rejects explicit mock data through `ensure_mock_allowed(RuntimeMode.AUDIT, explicit_mock=True)`
- prediction output source is the explicit fixture source
- no silent mock fallback is accepted

## Generated Artifacts

The integration test writes artifacts only under pytest temporary directories.
The governed output directory has the form:

`<tmp>/governed_chain_output/`

Required artifacts asserted:

- `forecast_summary.csv`
- `risk_summary.csv`
- `portfolio_allocation.csv`
- `portfolio_summary.csv`
- `portfolio_risk_summary.csv`
- `portfolio_decision_cards.jsonl`
- `router_decisions.csv`
- `router_summary.csv`
- `run_manifest.json`
- `risk_manifest.json`
- `allocator_manifest.json`
- `router_manifest.json`

Manifest files asserted:

- `run_manifest.json`
- `risk_manifest.json`
- `allocator_manifest.json`
- `router_manifest.json`

Manifest types asserted:

- `risk_governance_layer_v1_manifest`
- `portfolio_allocator_v1_manifest`
- `phase3_router_v1_manifest`

## Integration Assertions

The integration test asserts:

- forecast summary is produced
- risk summary is produced
- allocation output is produced as allocation candidates or all-cash/no-candidate state
- router diagnostic output is produced
- required manifests exist
- provenance fields exist through runtime manifest, run-mode columns, source fields, and fixture provenance
- runtime mode is explicit in `run_manifest["runtime"]["runtime_mode"]`
- mock fallback is not silent and audit mock use is rejected
- governed artifacts do not contain exact `BUY`, `SELL`, `STRONG_BUY`, or `STRONG_SELL` tokens
- governed artifacts do not expose broker/order/final-trade authority terms
- allocator and router outputs remain diagnostic-only
- router decisions are limited to:
  - `route_allocation_candidate`
  - `hold`
  - `reject`
  - `no_candidate`

## Router Diagnostic Assertions

The test verifies:

- `router_decisions.csv` is non-empty
- `diagnostic_only_authority` is true
- `no_buy_sell_recommendation_authority` is true
- `route_decision` values are a subset of canonical `ROUTE_DECISIONS`
- `router_manifest.json` lists the canonical router diagnostic labels
- `router_manifest.json` includes route decision counts for every canonical label

## Verification Results

Targeted Phase 8 integration:

- Command: `python -m pytest tests/integration/test_governed_diagnostic_chain.py -q`
- Result: `1 passed, 1 warning in 1.70s`

Full pytest:

- Command: `python -m pytest tests -q`
- Result: `825 passed, 5 skipped, 33 warnings in 304.11s (0:05:04)`

Repository hygiene:

- Command: `python scripts/check_repo_hygiene.py`
- Result: `Repository hygiene check passed.`

## Git Status

Status collected after full pytest, hygiene, evidence, and checklist edits,
before the Phase 8 commit:

```text
 M reports/cleanup/AUDIT_REMEDIATION_CHECKLIST.md
A  tests/integration/test_governed_diagnostic_chain.py
?? reports/cleanup/PHASE8_GOVERNED_INTEGRATION_TESTING_EVIDENCE.md
```

`reports/results/RESEARCH_TOPIC_AND_CODE_AUDIT_REPORT.md` was not edited during Phase 8.

## Unresolved Or Deferred Risks

- The standalone `scripts/run_portfolio_allocator.py` CLI still references the
  legacy allocator path, so Phase 8 used the canonical allocator through
  `scripts/run_quant_core.py`.
- The integration test covers the candidate-producing chain. Existing allocator
  and router tests cover no-allocation and missing-candidate diagnostics.
- Phase 8 does not change or broaden production governance semantics; final
  repository signoff is moved to Phase 9.
