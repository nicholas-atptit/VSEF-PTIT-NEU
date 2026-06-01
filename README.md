# VSEF — Vietnam Stock Evaluation and Forecasting Framework

**VSEF** is a deterministic decision-diagnostic framework designed for Vietnamese stock market research. The framework produces structured diagnostics across forecasting, scenario evaluation, risk governance, decision-lane generation, portfolio allocation, and routing.

VSEF is built strictly for **research and diagnostic purposes**. It does **not** issue BUY or SELL recommendations, execute trades, or provide production trading authority.

---

## Research Attribution

This research project is conducted by:

* **Luong Minh Quan** — Posts and Telecommunications Institute of Technology

  Contact: [luongminhquan.working.research@gmail.com](mailto:luongminhquan.working.research@gmail.com)

* **Nguyen Nguyet Ha** — National Economics University

  Contact: [nghnguyetha.workspace@gmail.com](mailto:nghnguyetha.workspace@gmail.com)

The project is developed in collaboration with, and financially supported by, the **Risk Management Department — Viettel Global**.

Market data is connected and retrieved through **Vnstock**, a Python package for Vietnamese stock market analysis.

> Vnstock by thinh-vu on GitHub. Copyright © 2022–2026.

---

## Proprietary Notice and Access Restrictions

VSEF is a proprietary research framework. It is **not an open-source project** and is not released under any open-source license.

All source code, documentation, schemas, diagnostic logic, research workflows, generated structures, naming conventions, and related project materials are protected intellectual property unless explicitly stated otherwise in writing.

Access to this repository, documentation, or any part of the project does **not** grant any license, ownership right, reuse right, redistribution right, publication right, or commercial usage right.

The following actions are strictly prohibited without prior written permission from the project owners:

* copying, cloning, or redistributing the framework;
* reusing the source code, architecture, diagnostic chain, schemas, or governance logic in another project;
* modifying and republishing the framework as a derivative system;
* using the framework for commercial, production, advisory, or trading services;
* extracting project materials for external publication, benchmarking, or model training;
* reverse engineering, reproducing, or imitating the protected design of the framework.

All rights are reserved by the project authors and authorized institutional collaborators.

Any unauthorized use, reproduction, redistribution, or derivative implementation of VSEF may constitute a violation of intellectual property rights and may be subject to legal action.

For permission requests, research inquiries, or institutional correspondence, contact:

* Luong Minh Quan: [luongminhquan.working.research@gmail.com](mailto:luongminhquan.working.research@gmail.com)
* Nguyen Nguyet Ha: [nghnguyetha.workspace@gmail.com](mailto:nghnguyetha.workspace@gmail.com)

---

## 1. System Overview

VSEF implements a deterministic diagnostic chain that transforms market inputs into governed research outputs. The current diagnostic pipeline is:

```text
Quant Core
-> Scenario Evaluation Engine v1
-> Risk Governance Layer v1
-> Decision Lane v2
-> Portfolio Allocator v1
-> Phase 3 Router v1
```

Each layer has a clearly defined role and output contract.

### Quant Core

The Quant Core produces the primary forecasting and analytical outputs, including:

* model forecasts
* model consensus
* model health diagnostics
* risk, regime, and strategy diagnostics
* analysis packets
* initial decision-lane candidates

### Scenario Evaluation Engine v1

The Scenario Evaluation Engine generates scenario-level diagnostics, including:

* scenario probability
* dominance analysis
* uncertainty diagnostics
* calibration diagnostics

### Risk Governance Layer v1

The Risk Governance Layer applies rule-based risk controls and governance diagnostics, including:

* `risk_score`
* `risk_level`
* `risk_action`
* confidence adjustment
* block flags
* force-hold flags

### Decision Lane v2

Decision Lane v2 enriches diagnostic candidates with additional structured information required for downstream allocation and routing.

### Portfolio Allocator v1

The Portfolio Allocator converts governed diagnostic candidates into allocation-level outputs, including:

* `allocation_candidate`
* `no_allocation`
* exposure constraints
* cash management rules

### Phase 3 Router v1

The Phase 3 Router produces the final routing diagnostic, including:

* `route_allocation_candidate`
* `hold`
* `reject`
* `no_candidate`

---

## 2. Authority Boundary

VSEF is not a trading system. It is a research framework with strict authority limits.

### Allowed Outputs

The framework may produce:

* forecast diagnostics
* scenario diagnostics
* risk diagnostics
* diagnostic candidates
* `allocation_candidate`
* `no_allocation`
* `route_allocation_candidate`
* `hold`
* `reject`
* `no_candidate`

### Prohibited Outputs

The framework must not produce:

* BUY recommendations
* SELL recommendations
* live execution instructions
* production trading authority
* learned meta-model authority
* investment advice presented as actionable trading instruction

The system is designed to support research interpretation, validation, and governance — not direct trading execution.

---

## 3. Quick Start

Run the following commands from the repository root using PowerShell.

### Runtime Preflight Check

```powershell
python scripts/check_runtime_preflight.py
```

### Smoke Run for the Full Diagnostic Chain

```powershell
python scripts/run_quant_core.py --preset smoke --run-mode research_core --enable-scenario-engine --enable-risk-governance --enable-portfolio-allocator --enable-phase3-router --output-dir artifacts/quant_core_router_smoke
```

Generated files under `artifacts/` are runtime outputs and must not be committed to the repository.

---

## 4. Validation Commands

Use the following commands to validate the main runtime and diagnostic modules:

```powershell
python -m compileall src/phase3_router
pytest tests/phase3_router -q
pytest tests/portfolio_allocator -q
pytest tests/decision_lane -q
pytest tests/risk_governance -q
pytest tests/scenario -q
pytest tests/quant_core -q
```

Before claiming end-to-end runtime readiness, run the full relevant validation suite locally.

---

## 5. Main Artifact Groups

VSEF generates structured artifacts by pipeline layer.

### Base Quant Core

* `run_manifest.json`
* `full_model_predictions.csv`
* `forecast_summary.csv`
* `model_consensus_summary.csv`
* `model_health_summary.csv`
* `analysis_packets.jsonl`
* `decision_lane_candidates.csv`

### Scenario Evaluation

* `scenario_probability.csv`
* `scenario_dominance_summary.csv`
* `scenario_uncertainty_summary.csv`
* `scenario_calibration_summary.csv`

### Risk Governance

* `risk_governance_summary.csv`
* `risk_adjusted_candidates.csv`
* `risk_override_log.csv`

### Decision Lane

* `decision_lane_enriched_candidates.csv`

### Portfolio Allocator

* `portfolio_allocation.csv`
* `portfolio_summary.csv`
* `portfolio_risk_summary.csv`

### Router

* `router_decisions.csv`
* `router_summary.csv`
* `router_manifest.json`

---

## 6. Documentation Map

The active documentation source is located under `docs/`.

* [System Overview](docs/SYSTEM_OVERVIEW.md)
* [Decision Diagnostic Chain](docs/DECISION_DIAGNOSTIC_CHAIN.md)
* [Authority Boundary](docs/AUTHORITY_BOUNDARY.md)
* [Pipeline Contracts](docs/governance/PIPELINE_CONTRACTS.md)
* [Quant Core Output Schema](docs/governance/QUANT_CORE_OUTPUT_SCHEMA.md)
* [Risk Governance Output Schema](docs/governance/RISK_GOVERNANCE_OUTPUT_SCHEMA.md)
* [Portfolio Allocator Output Schema](docs/governance/PORTFOLIO_ALLOCATOR_OUTPUT_SCHEMA.md)
* [Phase 3 Router Output Schema](docs/governance/PHASE3_ROUTER_OUTPUT_SCHEMA.md)
* [Full Decision Chain Smoke Runbook](docs/runbooks/RUN_FULL_DECISION_CHAIN_SMOKE.md)
* [Troubleshooting](docs/runbooks/TROUBLESHOOTING.md)
* [Documentation Inventory](docs/DOCS_INVENTORY.md)
* [Command Registry](docs/COMMAND_REGISTRY.md)
* [Reports Governance](docs/REPORTS_GOVERNANCE.md)
* [Docs README](docs/README.md)

---

## 7. Repository Structure

```text
src/        Core framework packages and diagnostic layers
scripts/    CLI entry points and local workflow runners
tests/      Unit tests and smoke-style validation suites
docs/       Active source of truth for architecture, schemas, runbooks, and roadmap
reports/    Historical snapshots and controlled remediation evidence
artifacts/  Generated workflow outputs; do not commit
```

### Documentation Governance

`docs/` is the active source of truth for architecture, schemas, command usage, and operational runbooks.

Most content under `reports/` should be treated as historical snapshot material. Controlled audit-remediation evidence under `reports/` is canonical only for remediation status and verification.

For details, see:

* [Reports Governance](docs/REPORTS_GOVERNANCE.md)

---

## 8. Current Status

The deterministic decision-diagnostic chain has been implemented. Documentation inventory and legacy governance archives have been normalized.

Before claiming full end-to-end runtime readiness, the full smoke validation process should be executed locally and verified through generated artifacts, manifests, and test results.

---

## 9. Development Rules

All development should follow the governance rules below:

1. Keep diagnostics separate from recommendations.
2. Preserve the authority boundary between research outputs and trading actions.
3. Keep canonical artifact names and schemas stable.
4. Update documentation whenever output schemas or runtime contracts change.
5. Do not commit generated artifacts under `artifacts/`.
6. Archive stale documentation instead of deleting it without review.
7. Validate runtime changes through tests, manifests, and reproducible commands.
8. Do not introduce learned meta-model authority unless explicitly governed and documented.

---

## 10. Disclaimer

VSEF is a research and diagnostic framework. Its outputs are intended for structured analysis, validation, and governance review. They should not be interpreted as financial advice, investment recommendations, or trading instructions.

Nothing in this repository constitutes investment advice, financial advice, trading advice, or a recommendation to buy, sell, hold, allocate, or execute any financial instrument.

Use of this framework is restricted by the proprietary notice and access restrictions stated above.
