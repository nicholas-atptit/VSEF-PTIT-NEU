# Phase 3 Router Output Schema
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Governance schema |
| Created / authored | Sunday, 2026-05-03 00:00:00 ICT (UTC+07:00) |
| Last updated | Sunday, 2026-05-03 00:00:00 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Status | Active |

## Authority Boundary

Phase 3 Router v1 is a deterministic diagnostic routing layer. It converts
Portfolio Allocator v1 diagnostics into route decisions only. It has no BUY or
SELL recommendation authority and does not train models.

## Inputs

Required router context:

- `portfolio_allocation.csv` or an equivalent DataFrame
- `portfolio_summary.csv` or an equivalent DataFrame
- `portfolio_risk_summary.csv` or an equivalent DataFrame

Optional context:

- `allocator_manifest.json`

If allocator outputs are unavailable, the router emits a valid diagnostic
`no_candidate` row with `missing_allocator_outputs`.

## Outputs

- `router_decisions.csv`
- `router_summary.csv`
- `router_manifest.json`

## Route Decisions

Valid `route_decision` values:

- `route_allocation_candidate`
- `hold`
- `reject`
- `no_candidate`

## Core Rules

The router applies rules in deterministic order:

- If `allocation_status != allocation_candidate`, emit `no_candidate`.
- If `risk_level == level_3_hard_override` or `risk_score >= 0.70`, emit
  `reject`.
- If `dominant_scenario` is `bear`, `drawdown`, or `high_volatility`, bias to
  `hold`; reject when elevated risk is also present.
- If volatility regime context is high, bias to `hold`; reject when elevated
  risk is also present.
- If `risk_adjusted_confidence` is medium, `disagreement_score` is medium, or
  conflict is mild, emit `hold`.
- If the candidate is valid, risk is low, confidence is high, disagreement is
  low, `final_weight > 0`, and portfolio exposure room is available, emit
  `route_allocation_candidate`.
- If `total_exposure` is near `max_total_exposure` or `cash_weight` is near
  `min_cash_buffer`, downgrade a routable candidate to `hold`.

## router_decisions.csv

Required fields:

- `router_decision_id`
- `allocation_id`
- `candidate_id`
- `source_packet_id`
- `timestamp`
- `ticker`
- `horizon`
- `route_decision`
- `route_reason`
- `allocation_status`
- `final_weight`
- `risk_level`
- `risk_score`
- `risk_adjusted_confidence`
- `disagreement_score`
- `dominance_score`
- `scenario_alignment`
- `dominant_scenario`
- `portfolio_status`
- `total_exposure`
- `cash_weight`
- `route_reason_codes`
- `diagnostic_only_authority`
- `no_buy_sell_recommendation_authority`

## router_summary.csv

Fields:

- `router_status`
- `source_allocation_count`
- `route_allocation_candidate_count`
- `hold_count`
- `reject_count`
- `no_candidate_count`
- `routed_final_weight`
- `total_exposure`
- `cash_weight`
- `diagnostic_only_authority`
- `no_buy_sell_recommendation_authority`

## router_manifest.json

The manifest records:

- version
- deterministic status
- config and thresholds
- input row counts
- output row counts
- artifact filenames and paths
- route decision counts
- required router decision fields
- allocator manifest context availability
- diagnostic-only authority
- no BUY/SELL recommendation authority
- missing allocator output status
