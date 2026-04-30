# Portfolio Allocator v1 Governance
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Governance note |
| Created / authored | Thursday, 2026-04-30 20:37:39 ICT (UTC+07:00) |
| Last updated | Thursday, 2026-04-30 20:37:39 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `0c96f76aef54968f0978a4f13003b397a95715da` |
| Timestamp source | Local Portfolio Allocator v1 implementation |
| Status | Active |

## Purpose

Portfolio Allocator v1 is a deterministic post-processor for saved Quant Core diagnostic outputs. It converts eligible decision-lane candidates into bounded allocation candidates, or emits a valid no-allocation state when inputs are missing or candidates fail gates.

Portfolio Allocator v1 emits allocation candidates only. It does not emit final BUY recommendations.

## Inputs

The allocator reads a Quant Core output directory and supports these input artifacts:

- `decision_lane_candidates.csv`
- `model_consensus_summary.csv`
- `model_health_summary.csv`
- `risk_summary.csv`
- `strategy_metrics.csv`
- `regime_summary.csv` if available

`decision_lane_candidates.csv` is the required candidate source. Other files are optional context inputs; missing optional files do not crash the allocator.

## Outputs

The allocator writes:

- `portfolio_allocation.csv`
- `portfolio_summary.csv`
- `portfolio_risk_summary.csv`
- `portfolio_decision_cards.jsonl`
- `allocator_manifest.json`

The JSONL cards use only:

- `allocation_candidate`
- `no_allocation`

Rejected candidate labels are summarized in `allocator_manifest.json`.

## Allocation Rules

Allocator defaults:

| Setting | Default |
| --- | ---: |
| `max_ticker_weight` | `0.10` |
| `max_total_exposure` | `0.60` |
| `cash_buffer` | `0.40` |
| `min_candidate_score` | `0.0` |
| `min_model_agreement` | `0.5` |
| `max_risk_score` | `1.0` |
| `risk_penalty_strength` | `0.5` |
| `agreement_penalty_strength` | `0.5` |
| `allow_short` | `false` |

Rules:

- Start from `candidate_score`, falling back to primary prediction times agreement when needed.
- Reject candidates below the confidence threshold.
- Reject candidates below the model-agreement threshold.
- Reject candidates above the configured risk threshold or in a severe drawdown state.
- Reject candidates linked to failing or weak model-health state.
- Penalize accepted scores for risk, fallback risk provenance, and less-than-perfect agreement.
- Normalize surviving candidates deterministically.
- Enforce max ticker weight.
- Enforce max total exposure and reserve the cash buffer.
- Preserve deterministic reproducibility for identical inputs and config.

## No-Allocation State

The allocator emits `no_allocation` when:

- `decision_lane_candidates.csv` is missing.
- The candidate file is empty.
- Required candidate columns are missing.
- All candidates are rejected by confidence, agreement, risk, or model-health gates.
- Accepted candidates have non-positive adjusted scores.

The no-allocation output is all cash and is still a valid governed result.

## Why This Is Not a BUY Recommendation

The allocator is a diagnostic allocation-candidate layer. It does not execute trades, does not rank seeds, does not certify production readiness, and does not convert candidates into final investment advice. Its output should be read as a bounded proposal under explicit gates and constraints.

## Claim Boundaries

| Claim | Allowed? | Reason |
| --- | --- | --- |
| The allocator emits deterministic allocation candidates. | Yes | Outputs are generated from saved CSVs and config only. |
| The allocator supports a no-allocation state. | Yes | Missing or rejected candidates produce an all-cash diagnostic output. |
| The allocator emits final BUY recommendations. | No | This layer intentionally emits allocation candidates only. |
| The system is production trading ready. | No | Live execution, validation, monitoring, and formal recommendation governance are out of scope. |

## Example CLI Command

```powershell
C:\Users\luong\.venv\Scripts\python.exe scripts/run_portfolio_allocator.py `
  --input-dir artifacts\quant_core_smoke_audit\decision_core_smoke `
  --output-dir tmp\portfolio_allocator_v1_smoke `
  --max-ticker-weight 0.10 `
  --max-total-exposure 0.60 `
  --cash-buffer 0.40 `
  --min-candidate-score 0.0 `
  --min-model-agreement 0.5
```

Use ignored local output directories such as `tmp/` or `artifacts/` for generated allocator artifacts.

## Limitations

- The allocator is deterministic and rule-based; it is not a learned portfolio policy.
- It depends on saved Quant Core artifact quality and schema availability.
- Missing optional context reduces gating precision.
- It does not validate future profitability.
- It does not replace Phase 3 routing.

## Next Step

Build the deterministic Phase 3 Router. The router should consume allocation candidates plus model health, consensus, risk, regime, and strategy metrics, then emit route decisions and routing manifests without introducing learned meta-model logic until enough validated out-of-sample history exists.
