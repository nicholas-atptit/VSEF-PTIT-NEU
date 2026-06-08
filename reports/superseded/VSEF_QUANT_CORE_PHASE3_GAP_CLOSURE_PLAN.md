# VSEF Quant Core Phase 3 Gap Closure Plan

## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Architecture gap-closure report |
| Created / authored | Thursday, 2026-04-30 20:24:45 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `b71000f830fa026cc1f0cd75ae420f6527c8030b` |
| Source reports | `reports/superseded/VSEF_1000_SEED_SMOKE_STABILITY_REPORT.md`; `reports/superseded/VSEF_1000_SEED_SMOKE_EXECUTIVE_SUMMARY.md`; `docs/audits/VSEF_QUANT_CORE_REPEATED_SEED_STABILITY_AUDIT.md`; `docs/governance/QUANT_CORE_GOVERNANCE.md`; `docs/governance/QUANT_CORE_OUTPUT_SCHEMA.md` |
| Source code inspected | `src/evaluation/quant_core.py`; `src/evaluation/consensus.py`; `src/reporting/analysis_packets.py`; `src/reporting/model_health.py`; `src/reporting/quant_core.py`; `src/core/model_governance.py` |
| Status | Active planning report; no code or training executed |

## Executive Summary

The completed 1000-seed smoke diagnostic closes an important stability and governance evidence layer for the Quant Core. It demonstrates that the bounded `smoke` / `research_core` workflow can be repeated across 1000 seeds with complete seed folders, complete aggregate files, and effectively zero seed-level dispersion in the saved stability outputs.

That does not close the decision architecture. The current Quant Core has governed forecasts, consensus summaries, model-health summaries, analysis packets, and conservative decision-lane candidates, but those candidates are diagnostic artifacts. They are not official BUY recommendations, they are not portfolio allocations, and they are not Phase 3 routed decisions.

The next architecture work should therefore separate three layers:

1. Decision-lane governance and schema clarity.
2. Portfolio Allocator v1.
3. Phase 3 deterministic routing, with learned meta-models deferred until enough validated out-of-sample history exists.

## What Has Been Solved

- The repeated-seed runner exists and executes real per-seed Quant Core commands rather than synthesizing fake metrics.
- The 1000-seed smoke run completed under `artifacts\quant_core_repeated_seed_1000_smoke`.
- Seed stability was measured through root aggregate files such as `model_seed_stability.csv`, `model_horizon_seed_stability.csv`, `strategy_seed_stability.csv`, `model_health_seed_stability.csv`, `consensus_seed_stability.csv`, and `decision_candidate_seed_stability.csv`.
- The 1000-seed report documents that seed-level dispersion is effectively zero for the bounded smoke setup.
- Decision candidates are explicitly diagnostic. They are generated from analysis packets using tradability, positive primary prediction, agreement bucket, and active signal filters.
- The reports explicitly reject best-seed cherry-picking.
- Generated artifacts remain evidence surfaces, not final recommendations.

## What Has Not Been Solved

| Gap | Current state | Missing capability |
| --- | --- | --- |
| Decision lane | Produces conservative `decision_lane_candidates.csv` rows for analyst review. | A formal separation between diagnostic candidates, allocation candidates, and final recommendations. |
| Portfolio allocator | Quant Core produces positions, trades, policy summaries, and strategy metrics, but no governed allocation layer. | A deterministic allocator that converts eligible candidates into bounded portfolio weights with explicit risk and exposure constraints. |
| Phase 3 routing / meta-model logic | Governance docs explicitly leave Phase 3 routing and meta-model decision logic out of scope. | A router that combines model health, consensus, risk, regime, strategy, and candidate scores into auditable route decisions. |

## Gap 1 - Decision Lane Governance

The current decision lane is a candidate generator. In `src/reporting/analysis_packets.py`, `build_decision_lane_candidates` filters packets that are tradable, positively predicted by the primary model, in a medium or high agreement bucket, and supported by at least one active policy signal. This is useful for analyst review, but it is not a final recommendation engine.

The next schema should distinguish:

| Schema object | Purpose | Status |
| --- | --- | --- |
| `CandidateCard` | Describes a diagnostic candidate emitted by the current decision lane. | Allowed now. |
| `RecommendationCard` | Describes a final recommendation after allocation, risk checks, and validation gates. | Disabled until allocator and validation exist. |

Required decision labels:

- `diagnostic_candidate`: current decision-lane output; for review and routing only.
- `allocation_candidate`: candidate accepted by allocator pre-checks but not a final recommendation.
- `final_recommendation`: disabled until Portfolio Allocator v1 and validation gates exist.

`final_recommendation` should remain unavailable in manifests, cards, and UI copy until portfolio allocation, risk checks, and validation evidence are implemented. This avoids implying BUY recommendation capability from a diagnostic candidate file.

## Gap 2 - Portfolio Allocator v1

Portfolio Allocator v1 should be deterministic, auditable, and conservative. It should not introduce new model families or train a learned policy. Its job is to convert candidate-level diagnostics into bounded allocation proposals or a no-allocation state.

Required allocator inputs:

- `decision_lane_candidates.csv`
- `model_consensus_summary.csv`
- `model_health_summary.csv`
- `risk_summary.csv`
- `strategy_metrics.csv`
- `regime_summary.csv` if available

Required allocator outputs:

- `portfolio_allocation.csv`
- `portfolio_summary.csv`
- `portfolio_risk_summary.csv`
- `portfolio_decision_cards.jsonl`
- `allocator_manifest.json`

Minimum allocation rules:

| Rule | Required behavior |
| --- | --- |
| Max ticker weight | Cap per-ticker allocation before normalization. |
| Max total exposure | Cap total invested weight across all candidates. |
| Cash buffer | Reserve explicit cash weight even when candidates pass all gates. |
| Confidence threshold | Require minimum candidate score and agreement before allocation. |
| Risk penalty | Reduce or block weights under high volatility, drawdown, or risk fallback states. |
| Model agreement penalty | Reduce or block weights when agreement is low or sign conflict is present. |
| No-allocation state | Emit a valid empty/all-cash output when no candidate passes gates. |
| Deterministic reproducibility | Identical inputs and config must produce identical allocations and manifests. |

Allocator v1 should produce allocation candidates, not final BUY recommendations. The output can say that a ticker receives a bounded proposed weight under a diagnostic allocation policy. It should not say the system recommends buying that ticker.

## Gap 3 - Phase 3 Routing / Meta-Model

The first Phase 3 version should be a deterministic router, not a trained meta-model. This keeps the architecture auditable while the project accumulates enough validated out-of-sample history for a future learned router.

Deterministic router inputs:

- model health
- consensus
- candidate score
- regime
- risk
- strategy metrics

Deterministic router outputs:

- `route_decision.csv`
- `phase3_decision_cards.jsonl`
- `routing_manifest.json`

Initial route labels should be explicit, for example:

- `reject_low_confidence`
- `reject_high_risk`
- `reject_unhealthy_model`
- `allocation_candidate`
- `hold_for_review`
- `no_candidate`

Future learned meta-model requirements:

- sufficient out-of-sample history across tickers, horizons, regimes, and time periods
- walk-forward validation of router decisions
- strict leakage controls
- benchmark against the deterministic router
- no test-period cherry-picking
- no seed selection based on test-period performance
- manifest records for training data, feature windows, labels, and evaluation windows

The learned meta-model should not replace the deterministic router until it demonstrates incremental value under leakage-safe validation.

## Recommended Implementation Order

| Phase | Task | Deliverable |
| --- | --- | --- |
| Phase A | Governance/schema clarification | Add `CandidateCard`, `RecommendationCard`, and decision-label rules; keep `final_recommendation` disabled. |
| Phase B | Portfolio Allocator v1 | Deterministic allocator with bounded exposure, risk penalties, no-allocation handling, and allocator manifest. |
| Phase C | Deterministic Phase 3 Router | Route decision tables and JSONL cards using health, consensus, risk, regime, and strategy gates. |
| Phase D | Medium/decision_core repeated-seed validation | Re-run stability diagnostics for allocator/router outputs after implementation. |
| Phase E | Optional learned meta-model | Train only after enough validated out-of-sample history exists and benchmark against the deterministic router. |

## Claim Boundaries

| Claim | Allowed now? | Reason |
| --- | --- | --- |
| The smoke workflow is stable across 1000 seeds. | Yes | The saved 1000-seed smoke report shows 1000 completed seed folders and effectively zero dispersion in the saved stability aggregates. |
| The system has final BUY recommendations. | No | Current `decision_lane_candidates.csv` is a diagnostic candidate file, not an allocator-backed recommendation surface. |
| The system has a complete portfolio allocator. | No | No governed allocator outputs such as `portfolio_allocation.csv` or `allocator_manifest.json` exist yet. |
| The system is production trading ready. | No | Smoke stability does not prove market edge, future profitability, live execution readiness, or full risk governance. |
| The system has candidate-level decision diagnostics. | Yes | Quant Core emits analysis packets and conservative decision-lane candidates with consensus, risk, regime, and policy context. |

## Next Codex Task Prompt

```text
You are working inside my local repository:

VSEF - Vietnam Stock Evaluation and Forecasting Framework

Task type: implement Portfolio Allocator v1 safely.

Do NOT run heavy training.
Do NOT run 1000 repeated seeds.
Do NOT change model families.
Do NOT claim BUY recommendation capability.
Do NOT commit artifacts under artifacts/, outputs/, tmp/, or models/.

Objective:
Implement a deterministic Portfolio Allocator v1 that consumes Quant Core diagnostic outputs and emits allocation-candidate artifacts only.

Inputs:
- decision_lane_candidates.csv
- model_consensus_summary.csv
- model_health_summary.csv
- risk_summary.csv
- strategy_metrics.csv
- regime_summary.csv if available

Outputs:
- portfolio_allocation.csv
- portfolio_summary.csv
- portfolio_risk_summary.csv
- portfolio_decision_cards.jsonl
- allocator_manifest.json

Rules:
- cap max ticker weight
- cap max total exposure
- reserve cash buffer
- require confidence threshold
- apply risk penalty
- apply model agreement penalty
- support no-allocation state
- deterministic reproducibility for identical inputs/config
- label outputs as allocation_candidate, not final_recommendation

Implementation guidance:
1. Inspect current Quant Core output schemas and tests.
2. Add allocator module under an appropriate src/ package.
3. Add a lightweight CLI or integrate behind an explicit optional Quant Core flag.
4. Add focused unit tests using small synthetic CSV fixtures.
5. Add documentation that final_recommendation remains disabled.
6. Run targeted tests and compileall.
7. Stage only source, tests, and docs; do not stage artifacts.
```
