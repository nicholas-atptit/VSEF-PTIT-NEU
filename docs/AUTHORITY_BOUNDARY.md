# Authority Boundary
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Governance policy |
| Created / authored | Tuesday, 2026-05-05 00:00:00 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-05-05 00:00:00 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Timestamp source | Local deterministic decision-chain documentation refactor |
| Status | Active |

## Core Rule

The active VSEF decision chain is diagnostic-only.

It may emit deterministic diagnostics and review labels. It must not emit BUY or
SELL recommendations, live execution instructions, production trading authority,
or learned meta-model authority.

## Allowed Outputs

The system is allowed to emit:

- forecast diagnostics
- scenario diagnostics
- risk diagnostics
- diagnostic candidates
- `allocation_candidate`
- `no_allocation`
- `route_allocation_candidate`
- `hold`
- `reject`
- `no_candidate`

These outputs are valid only as research and review artifacts.

## Disallowed Outputs

The system is not allowed to emit:

- BUY recommendation
- SELL recommendation
- live execution instruction
- production trading authority
- learned meta-model authority
- claims that a route decision is an order
- claims that an allocation candidate is a final portfolio decision

## Layer Boundaries

| Layer | Allowed authority | Explicitly disallowed authority |
| --- | --- | --- |
| Quant Core | Forecast, consensus, model-health, regime, risk, policy, packet, and legacy candidate diagnostics. | BUY/SELL recommendation or production trading readiness. |
| Scenario Evaluation Engine v1 | Scenario probability, ranking, dominance, uncertainty, and calibration diagnostics. | Scenario-driven trade action or confidence promotion beyond diagnostics. |
| Risk Governance Layer v1 | Risk score, risk level, risk action, confidence adjustment, block, force-hold, and reason diagnostics. | Trade instruction or final investment veto authority outside the diagnostic chain. |
| Decision Lane v2 | Enriched diagnostic candidate surface and reason summaries. | Direct candidate creation from non-chain sources without governance, or BUY/SELL authority. |
| Portfolio Allocator v1 | `allocation_candidate` or `no_allocation` diagnostics with bounded sizing and exposure context. | Forced trade, order sizing, broker instruction, or final portfolio authority. |
| Phase 3 Router v1 | `route_allocation_candidate`, `hold`, `reject`, or `no_candidate` diagnostics. | Live execution, final recommendation, learned meta-router, or production trading authority. |

## Language Rules

Use:

- diagnostic candidate
- allocation candidate
- route decision
- governed no-allocation
- review surface
- deterministic rule

Avoid:

- recommendation
- order
- trade signal
- execution approval
- production ready
- learned router

When historical docs or legacy modules use stronger trading language, active
decision-chain docs must restate the diagnostic-only boundary before referring
to those artifacts.
