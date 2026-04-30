# Phase 3 Router v1 Governance
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Governance note |
| Created / authored | Friday, 2026-05-01 01:36:14 ICT (UTC+07:00) |
| Last updated | Friday, 2026-05-01 01:36:14 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `dc76edd78a8b3ba8541a5b3691a345c49ceec67c` |
| Timestamp source | Local deterministic Phase 3 Router v1 implementation |
| Status | Active |

## Purpose

Phase 3 Router v1 is a deterministic post-processor for Portfolio Allocator v1 outputs and saved Quant Core diagnostics. It converts allocation candidates into governed route decisions that can be audited before any future recommendation or execution layer is considered.

Phase 3 Router v1 emits deterministic route decisions only. It does not emit final BUY recommendations.

## Inputs

Required allocator inputs:

- `portfolio_allocation.csv`
- `portfolio_summary.csv`
- `portfolio_risk_summary.csv`
- `portfolio_decision_cards.jsonl`
- `allocator_manifest.json`

Optional Quant Core diagnostics:

- `model_consensus_summary.csv`
- `model_health_summary.csv`
- `risk_summary.csv`
- `regime_summary.csv`
- `strategy_metrics.csv`

Missing optional diagnostics do not crash the router. Missing required allocator outputs produce a governed missing-data route.

## Outputs

The router writes:

- `route_decision.csv`
- `phase3_decision_cards.jsonl`
- `routing_summary.csv`
- `routing_manifest.json`

These outputs are diagnostic routing artifacts. They do not authorize trade execution.

## Route Labels

Allowed route labels:

- `route_allocation_candidate`
- `hold_for_review`
- `reject_low_confidence`
- `reject_high_risk`
- `reject_low_agreement`
- `reject_unhealthy_model`
- `reject_allocator_no_allocation`
- `no_candidate`
- `rejected_missing_required_data`

## Routing Rules

Default settings:

| Setting | Default |
| --- | ---: |
| `min_allocation_weight` | `0.01` |
| `min_candidate_score` | `0.0` |
| `min_model_agreement` | `0.5` |
| `max_risk_score` | `1.0` |
| `require_positive_allocation` | `true` |
| `allow_no_allocation` | `true` |
| `severe_drawdown_blocks` | `true` |
| `unhealthy_model_blocks` | `true` |
| `low_agreement_action` | `hold_for_review` |

Rules:

- If required allocator outputs are missing, emit `rejected_missing_required_data`.
- If the allocator emitted `no_allocation`, emit `no_candidate` by default, or `reject_allocator_no_allocation` when configured to reject that state.
- Hold candidates with non-positive allocation weight or allocation weight below the minimum threshold.
- Reject candidates below the confidence threshold.
- Hold or reject candidates below the agreement threshold according to `low_agreement_action`.
- Reject candidates above the risk threshold or in a severe drawdown state.
- Reject candidates linked to failing or weak model-health state.
- Route surviving candidates as `route_allocation_candidate`.
- Preserve deterministic reproducibility for identical inputs and configuration.

## No-Candidate Behavior

The router treats allocator no-allocation as a valid governed outcome. In the default configuration, it writes a `no_candidate` route card rather than forcing an allocation. This preserves the conservative behavior established by Portfolio Allocator v1.

## Why This Is Not a BUY Recommendation

The router is a diagnostic governance layer. It does not execute trades, does not train a learned meta-model, does not select random seeds, and does not certify production readiness. A routed allocation candidate is still only a route decision for review.

## Claim Boundaries

| Claim | Allowed? | Reason |
| --- | --- | --- |
| The router emits deterministic route decisions. | Yes | Outputs are generated from saved artifacts and explicit config only. |
| The router can reject or hold weak allocation candidates. | Yes | Risk, confidence, agreement, allocation-weight, and model-health gates are explicit. |
| The router creates final BUY recommendations. | No | Recommendation authority remains out of scope. |
| The router is a learned meta-model. | No | No model is trained in v1. |
| The system is production trading ready. | No | Live execution, monitoring, validation, and recommendation governance are not implemented. |

## Example CLI Command

```powershell
C:\Users\luong\.venv\Scripts\python.exe scripts/run_phase3_router.py `
  --input-dir tmp\portfolio_allocator_v1_smoke `
  --output-dir tmp\phase3_router_v1_smoke `
  --min-allocation-weight 0.01 `
  --min-candidate-score 0.0 `
  --min-model-agreement 0.5 `
  --max-risk-score 1.0 `
  --low-agreement-action hold_for_review `
  --allow-no-allocation
```

Use ignored local output directories such as `tmp/` or `artifacts/` for generated router artifacts.

## Limitations

- The router is deterministic and rule-based; it is not a trained routing model.
- It depends on Portfolio Allocator v1 and Quant Core artifact quality.
- Missing optional diagnostics reduce routing context.
- It does not prove future profitability.
- It does not replace formal portfolio validation or execution controls.

## Next Step

Run medium or `decision_core` validation after allocator and router real-output smoke checks. The validation should confirm that allocator outputs and router outputs remain schema-stable on real Quant Core artifacts before any broader Phase 3 routing audit is promoted.

## Future Learned Meta-Model Requirements

A learned Phase 3 meta-model should remain out of scope until the project has enough validated out-of-sample history. Before training a learned router, the repository should have:

- Walk-forward validation across multiple market regimes.
- Leakage controls that prevent test-period selection.
- Benchmark comparisons against this deterministic router.
- Seed-stability diagnostics for any stochastic route component.
- Clear governance labels that separate candidates from recommendations.
