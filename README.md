# VSEF - Vietnam Stock Evaluation and Forecasting Framework

VSEF is a deterministic decision-diagnostic framework for Vietnamese stock market research. It produces forecast, scenario, risk, decision-lane, allocation, and routing diagnostics. It does not produce BUY or SELL recommendations.

## Current Diagnostic Chain

```text
Quant Core
-> Scenario Evaluation Engine v1
-> Risk Governance Layer v1
-> Decision Lane v2
-> Portfolio Allocator v1
-> Phase 3 Router v1
```

- Quant Core: forecasts, consensus, model health, risk/regime/strategy diagnostics, and analysis packets.
- Scenario Evaluation Engine v1: probability, dominance, uncertainty, and calibration diagnostics.
- Risk Governance Layer v1: `risk_score`, `risk_level`, `risk_action`, confidence adjustment, and block/force-hold flags.
- Decision Lane v2: enriched diagnostic candidates.
- Portfolio Allocator v1: `allocation_candidate` or `no_allocation` with exposure and cash rules.
- Phase 3 Router v1: `route_allocation_candidate`, `hold`, `reject`, or `no_candidate`.

## Authority Boundary

Allowed outputs:

- forecast diagnostics
- scenario diagnostics
- risk diagnostics
- diagnostic candidates
- `allocation_candidate` / `no_allocation`
- `route_allocation_candidate` / `hold` / `reject` / `no_candidate`

Not allowed:

- BUY recommendation
- SELL recommendation
- live execution
- production trading authority
- learned meta-model authority

## Quick Start

Run from the repository root in PowerShell:

```powershell
python scripts/check_runtime_preflight.py
```

```powershell
python scripts/run_quant_core.py --preset smoke --run-mode research_core --enable-scenario-engine --enable-risk-governance --enable-portfolio-allocator --enable-phase3-router --output-dir artifacts/quant_core_router_smoke
```

Generated artifacts under `artifacts/` must not be committed.

## Validation Commands

```powershell
python -m compileall src/phase3_router
pytest tests/phase3_router -q
pytest tests/portfolio_allocator -q
pytest tests/decision_lane -q
pytest tests/risk_governance -q
pytest tests/scenario -q
pytest tests/quant_core -q
```

## Main Artifact Groups

Base Quant Core:

- `run_manifest.json`
- `full_model_predictions.csv`
- `forecast_summary.csv`
- `model_consensus_summary.csv`
- `model_health_summary.csv`
- `analysis_packets.jsonl`
- `decision_lane_candidates.csv`

Scenario:

- `scenario_probability.csv`
- `scenario_dominance_summary.csv`
- `scenario_uncertainty_summary.csv`
- `scenario_calibration_summary.csv`

Risk Governance:

- `risk_governance_summary.csv`
- `risk_adjusted_candidates.csv`
- `risk_override_log.csv`

Decision Lane:

- `decision_lane_enriched_candidates.csv`

Portfolio Allocator:

- `portfolio_allocation.csv`
- `portfolio_summary.csv`
- `portfolio_risk_summary.csv`

Router:

- `router_decisions.csv`
- `router_summary.csv`
- `router_manifest.json`

## Documentation Map

- [docs/SYSTEM_OVERVIEW.md](docs/SYSTEM_OVERVIEW.md)
- [docs/DECISION_DIAGNOSTIC_CHAIN.md](docs/DECISION_DIAGNOSTIC_CHAIN.md)
- [docs/AUTHORITY_BOUNDARY.md](docs/AUTHORITY_BOUNDARY.md)
- [docs/governance/PIPELINE_CONTRACTS.md](docs/governance/PIPELINE_CONTRACTS.md)
- [docs/governance/QUANT_CORE_OUTPUT_SCHEMA.md](docs/governance/QUANT_CORE_OUTPUT_SCHEMA.md)
- [docs/governance/RISK_GOVERNANCE_OUTPUT_SCHEMA.md](docs/governance/RISK_GOVERNANCE_OUTPUT_SCHEMA.md)
- [docs/governance/PORTFOLIO_ALLOCATOR_OUTPUT_SCHEMA.md](docs/governance/PORTFOLIO_ALLOCATOR_OUTPUT_SCHEMA.md)
- [docs/governance/PHASE3_ROUTER_OUTPUT_SCHEMA.md](docs/governance/PHASE3_ROUTER_OUTPUT_SCHEMA.md)
- [docs/runbooks/RUN_FULL_DECISION_CHAIN_SMOKE.md](docs/runbooks/RUN_FULL_DECISION_CHAIN_SMOKE.md)
- [docs/runbooks/TROUBLESHOOTING.md](docs/runbooks/TROUBLESHOOTING.md)
- [docs/DOCS_INVENTORY.md](docs/DOCS_INVENTORY.md)
- [docs/COMMAND_REGISTRY.md](docs/COMMAND_REGISTRY.md)
- [docs/REPORTS_GOVERNANCE.md](docs/REPORTS_GOVERNANCE.md)
- [docs/README.md](docs/README.md)

## Repository Structure

```text
src/        Core framework packages and diagnostic layers
scripts/    CLI entry points and local workflow runners
tests/      Unit and smoke-style validation suites
docs/       Active source of truth for architecture, schemas, runbooks, and roadmap
reports/    Historical snapshots plus controlled remediation evidence
artifacts/  Generated workflow outputs; do not commit
```

`docs/` contains the active source of truth. Most `reports/` content is historical snapshot material; controlled audit-remediation evidence under `reports/` is canonical only for remediation status and verification. See `docs/REPORTS_GOVERNANCE.md`.

## Current Status

The deterministic decision-diagnostic chain is implemented. Documentation inventory and legacy governance archive are normalized. Full smoke validation should be run locally before claiming end-to-end runtime readiness.

## Development Rules

- Keep diagnostics separate from recommendations.
- Keep canonical artifacts stable.
- Update docs when schemas change.
- Do not commit generated artifacts.
- Archive stale docs instead of deleting them blindly.
