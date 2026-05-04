# System Overview
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | System overview |
| Created / authored | Tuesday, 2026-05-05 00:00:00 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-05-05 00:00:00 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Timestamp source | Local deterministic decision-chain documentation refactor |
| Status | Active |

## Purpose

VSEF is a deterministic decision-diagnostic stock evaluation framework for
Vietnamese market research. The active chain turns forecast, scenario, risk,
candidate, allocation, and routing diagnostics into auditable artifacts for
review.

The system does not emit BUY or SELL recommendations. It does not execute
trades, certify production trading readiness, or train a learned meta-router.

## Implemented Chain

```text
Quant Core
-> Scenario Evaluation Engine v1
-> Risk Governance Layer v1
-> Decision Lane v2
-> Portfolio Allocator v1
-> Phase 3 Router v1
```

Each layer writes deterministic artifacts and preserves its authority boundary.
Downstream layers consume saved tables and manifests instead of hidden state.

## Layer Summary

| Layer | Role | Canonical outputs |
| --- | --- | --- |
| Quant Core | Runs governed forecast, regime, risk, consensus, policy, model-health, and packet diagnostics. | `run_manifest.json`, forecast tables, `analysis_packets.jsonl`, `decision_lane_candidates.csv` |
| Scenario Evaluation Engine v1 | Converts forecast and packet context into deterministic scenario probabilities, dominance, uncertainty, and calibration diagnostics. | `scenario_probability.csv`, `scenario_rankings.csv`, `scenario_dominance_summary.csv`, `scenario_uncertainty_summary.csv`, `scenario_calibration_summary.csv`, `scenario_manifest.json` |
| Risk Governance Layer v1 | Scores candidate context with normalized risk components and risk actions. | `risk_governance_summary.csv`, `risk_adjusted_candidates.csv`, `risk_override_log.csv`, `risk_manifest.json` |
| Decision Lane v2 | Builds enriched diagnostic candidates from candidate, scenario, and risk context. | `decision_lane_enriched_candidates.csv`, `decision_lane_manifest.json` |
| Portfolio Allocator v1 | Converts eligible enriched candidates into bounded diagnostic allocation candidates or a governed no-allocation state. | `portfolio_allocation.csv`, `portfolio_summary.csv`, `portfolio_risk_summary.csv`, `portfolio_decision_cards.jsonl`, `allocator_manifest.json` |
| Phase 3 Router v1 | Converts allocator rows into diagnostic route decisions. | `router_decisions.csv`, `router_summary.csv`, `router_manifest.json` |

## Allowed Output Classes

The active framework may emit:

- forecast diagnostics
- scenario diagnostics
- risk diagnostics
- diagnostic candidates
- `allocation_candidate` and `no_allocation`
- `route_allocation_candidate`, `hold`, `reject`, and `no_candidate`

These are review surfaces, not trading instructions.

## Canonical References

- [Decision Diagnostic Chain](DECISION_DIAGNOSTIC_CHAIN.md)
- [Authority Boundary](AUTHORITY_BOUNDARY.md)
- [Pipeline Contracts](governance/PIPELINE_CONTRACTS.md)
- [Quant Core Output Schema](governance/QUANT_CORE_OUTPUT_SCHEMA.md)
- [Risk Governance Output Schema](governance/RISK_GOVERNANCE_OUTPUT_SCHEMA.md)
- [Portfolio Allocator Output Schema](governance/PORTFOLIO_ALLOCATOR_OUTPUT_SCHEMA.md)
- [Phase 3 Router Output Schema](governance/PHASE3_ROUTER_OUTPUT_SCHEMA.md)
