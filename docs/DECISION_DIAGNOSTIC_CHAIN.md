# Decision Diagnostic Chain
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Architecture note |
| Created / authored | Tuesday, 2026-05-05 00:00:00 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-05-05 00:00:00 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Timestamp source | Local deterministic decision-chain documentation refactor |
| Status | Active |

## Chain

```text
Quant Core
-> Scenario Evaluation Engine v1
-> Risk Governance Layer v1
-> Decision Lane v2
-> Portfolio Allocator v1
-> Phase 3 Router v1
```

The chain is diagnostic-only. It creates auditable research artifacts and
bounded review decisions. It does not create BUY or SELL recommendations.

## Quant Core

| Field | Contract |
| --- | --- |
| Purpose | Run governed forecast, risk, regime, policy, consensus, model-health, and analysis-packet diagnostics across the requested preset and run mode. |
| Inputs | Scenario matrix from CLI preset or explicit ticker, horizon, target, and model filters; market data available to the local evaluation stack. |
| Processing logic | Executes eligible models by role and run mode, records skipped models, computes summary tables, builds consensus and model-health diagnostics, and writes analysis packets. |
| Outputs | Base artifacts listed in `docs/governance/QUANT_CORE_OUTPUT_SCHEMA.md`, including `analysis_packets.jsonl` and `decision_lane_candidates.csv`. |
| Downstream consumer | Scenario Evaluation Engine v1 consumes forecasts, consensus, risk, regime, strategy metrics, analysis packets, and model health. Risk Governance consumes decision-lane candidates and analysis packets. |
| Missing-input behavior | Unsupported dependencies and model failures are recorded in `skipped_models` and `model_execution_log`; fallback risk and regime states are explicit. Empty scenario matrices are a runner error. |
| Diagnostic-only boundary | Forecasts, consensus, policy, and candidate surfaces are diagnostics. They are not BUY or SELL recommendations. |

## Scenario Evaluation Engine v1

| Field | Contract |
| --- | --- |
| Purpose | Convert Quant Core forecast and packet context into deterministic scenario probability, dominance, uncertainty, and calibration diagnostics. |
| Inputs | `full_model_predictions.csv`, `model_consensus_summary.csv`, `risk_summary.csv`, `regime_summary.csv`, `strategy_metrics.csv`, `analysis_packets.jsonl`, and `model_health_summary.csv` or equivalent DataFrames. |
| Processing logic | Assigns probabilities to the canonical scenario labels, ranks scenarios, computes dominance and uncertainty summaries, records calibration diagnostics, and enriches analysis packets with scenario fields. |
| Outputs | `scenario_probability.csv`, `scenario_rankings.csv`, `scenario_dominance_summary.csv`, `scenario_uncertainty_summary.csv`, `scenario_calibration_summary.csv`, `scenario_manifest.json`. |
| Downstream consumer | Risk Governance Layer v1 and Decision Lane v2 consume scenario dominance, uncertainty, probability, and packet fields. |
| Missing-input behavior | Missing or sparse calibration context is recorded as uncalibrated or low-confidence scenario context rather than promoted to authority. |
| Diagnostic-only boundary | Scenario labels and probabilities are uncertainty diagnostics only. They do not authorize trading actions. |

## Risk Governance Layer v1

| Field | Contract |
| --- | --- |
| Purpose | Apply deterministic risk scoring and action fields to diagnostic candidates. |
| Inputs | `decision_lane_candidates.csv`, `analysis_packets.jsonl`, and optional `risk_summary.csv`, `model_consensus_summary.csv`, `model_health_summary.csv`, `scenario_dominance_summary.csv`, `scenario_uncertainty_summary.csv`, and `scenario_probability.csv`. |
| Processing logic | Merges candidate and packet context, computes normalized risk components, calculates weighted `risk_score`, derives `risk_level`, `risk_action`, `confidence_adjustment_factor`, `block_candidate`, `force_hold`, and reason codes. |
| Outputs | `risk_governance_summary.csv`, `risk_adjusted_candidates.csv`, `risk_override_log.csv`, `risk_manifest.json`. |
| Downstream consumer | Decision Lane v2 consumes risk-adjusted candidates and governance fields. Portfolio Allocator v1 later consumes enriched Decision Lane outputs that include risk fields. |
| Missing-input behavior | If no candidates are available, emits empty schema-stable risk artifacts. Missing optional context contributes conservative defaults or `none` reason codes. |
| Diagnostic-only boundary | Risk actions are candidate filters and confidence adjustments. They are not recommendation or execution instructions. |

## Decision Lane v2

| Field | Contract |
| --- | --- |
| Purpose | Build an enriched diagnostic candidate surface by joining legacy candidate filters with scenario and risk context. |
| Inputs | `decision_lane_candidates.csv`, `analysis_packets.jsonl`, `risk_adjusted_candidates.csv`, `scenario_dominance_summary.csv`, and `scenario_probability.csv` or equivalent DataFrames. |
| Processing logic | Preserves candidate identity, joins by packet and context keys, adds scenario alignment, risk-adjusted confidence, candidate status, reason codes, and reason summary. |
| Outputs | `decision_lane_enriched_candidates.csv`, `decision_lane_manifest.json`. |
| Downstream consumer | Portfolio Allocator v1 consumes enriched candidates. |
| Missing-input behavior | Enriched Decision Lane outputs are written only when risk-governance context exists in the top-level runner. Missing scenario context maps to `unknown` alignment. |
| Diagnostic-only boundary | Candidates are review surfaces. They do not create BUY or SELL authority. |

## Portfolio Allocator v1

| Field | Contract |
| --- | --- |
| Purpose | Convert eligible enriched diagnostic candidates into bounded allocation candidates or a governed no-allocation state. |
| Inputs | `decision_lane_enriched_candidates.csv`, optional `risk_adjusted_candidates.csv`, optional `scenario_dominance_summary.csv`, and allocator config. |
| Processing logic | Applies deterministic gates, ranks candidates, sizes bounded final weights, enforces max exposure and cash buffer, and emits all-cash diagnostics when candidates fail gates. |
| Outputs | `portfolio_allocation.csv`, `portfolio_summary.csv`, `portfolio_risk_summary.csv`, `portfolio_decision_cards.jsonl`, `allocator_manifest.json`. |
| Downstream consumer | Phase 3 Router v1 consumes portfolio allocation, summary, risk summary, and optional allocator manifest. |
| Missing-input behavior | If enriched candidates are missing, emits a valid `no_allocation` all-cash diagnostic output with `missing_enriched_candidates`. |
| Diagnostic-only boundary | `allocation_candidate` is a bounded diagnostic allocation row, not a portfolio order or recommendation. |

## Phase 3 Router v1

| Field | Contract |
| --- | --- |
| Purpose | Convert Portfolio Allocator v1 rows into final diagnostic route decisions for review. |
| Inputs | `portfolio_allocation.csv`, `portfolio_summary.csv`, `portfolio_risk_summary.csv`, and optional `allocator_manifest.json`. |
| Processing logic | Standardizes allocator fields, evaluates deterministic route guards, applies risk, confidence, disagreement, scenario, conflict, and exposure checks, then writes canonical route artifacts. |
| Outputs | `router_decisions.csv`, `router_summary.csv`, `router_manifest.json`. |
| Downstream consumer | Human review, audit workflows, and future diagnostic-only downstream layers. |
| Missing-input behavior | Missing allocator outputs produce a valid `no_candidate` route row with missing-output context. Allocator `no_allocation` rows also route to `no_candidate`. |
| Diagnostic-only boundary | Route decisions are `route_allocation_candidate`, `hold`, `reject`, or `no_candidate`. They do not authorize BUY, SELL, live execution, production trading, or learned meta-model behavior. |
