# VSEF Phase 3 Allocator-Router Real-Output Smoke Report
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Real-output smoke validation report |
| Created / authored | Friday, 2026-05-01 03:15:10 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `b9a4fd29913b568b79d662b3d9d238a1e3add1f8` |
| Source artifact root | `artifacts\phase3_real_smoke` |
| Seed range | 1-10 |
| Status | Complete conservative smoke validation |

## Executive Summary

Portfolio Allocator v1 and Phase 3 Router v1 were validated on real saved Quant Core artifacts for 10 smoke seeds. The validation used the existing saved outputs under `artifacts\phase3_real_smoke` and did not rerun Quant Core, train models, rerun repeated seeds, or modify artifacts.

All 10 allocator output folders and all 10 router output folders were present with the required files. The allocator consistently emitted a governed `no_allocation` / `CASH` state because all decision-lane candidates were rejected by allocator gates. The router then consistently emitted `no_candidate` / `CASH` route decisions.

Portfolio Allocator v1 and Phase 3 Router v1 emit governed candidates and route decisions only; they do not emit final BUY recommendations.

## Validation Scope

This report validates artifact compatibility and conservative orchestration behavior on 10 real saved Quant Core seed outputs. It is not a heavy training run, not a 1000-seed rerun, not medium or `decision_core` validation, not production validation, and not final recommendation generation.

The smoke result should be interpreted as schema and governance validation. It confirms that the allocator and router can consume real saved artifacts and preserve conservative no-allocation behavior when no allocation candidate passes the configured gates.

## Artifact Verification

Required allocator files checked per seed:

- `portfolio_allocation.csv`
- `portfolio_summary.csv`
- `portfolio_risk_summary.csv`
- `portfolio_decision_cards.jsonl`
- `allocator_manifest.json`

Required router files checked per seed:

- `route_decision.csv`
- `phase3_decision_cards.jsonl`
- `routing_summary.csv`
- `routing_manifest.json`

| Seed | Allocator folder | Allocator files | Router folder | Router files | Allocator label | Router label |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | `seed_000001_allocator` | 5/5 present | `seed_000001_router` | 4/4 present | `no_allocation` | `no_candidate` |
| 2 | `seed_000002_allocator` | 5/5 present | `seed_000002_router` | 4/4 present | `no_allocation` | `no_candidate` |
| 3 | `seed_000003_allocator` | 5/5 present | `seed_000003_router` | 4/4 present | `no_allocation` | `no_candidate` |
| 4 | `seed_000004_allocator` | 5/5 present | `seed_000004_router` | 4/4 present | `no_allocation` | `no_candidate` |
| 5 | `seed_000005_allocator` | 5/5 present | `seed_000005_router` | 4/4 present | `no_allocation` | `no_candidate` |
| 6 | `seed_000006_allocator` | 5/5 present | `seed_000006_router` | 4/4 present | `no_allocation` | `no_candidate` |
| 7 | `seed_000007_allocator` | 5/5 present | `seed_000007_router` | 4/4 present | `no_allocation` | `no_candidate` |
| 8 | `seed_000008_allocator` | 5/5 present | `seed_000008_router` | 4/4 present | `no_allocation` | `no_candidate` |
| 9 | `seed_000009_allocator` | 5/5 present | `seed_000009_router` | 4/4 present | `no_allocation` | `no_candidate` |
| 10 | `seed_000010_allocator` | 5/5 present | `seed_000010_router` | 4/4 present | `no_allocation` | `no_candidate` |

All expected allocator and router folders were present for seeds 1-10. No missing required smoke output files were found.

## Recommendation Boundary Scan

A broad forbidden-term scan was run across the smoke output root. The scan found the lowercase word `buy` only inside negative governance text such as `not a buy recommendation` in allocator manifests and allocator decision cards.

No `BUY`, `SELL`, `final_recommendation`, or `production_ready` labels or recommendation outputs were emitted by the saved allocator or router artifacts.

## Allocator Output Summary

Across the 10 allocator outputs:

| Metric | Value |
| --- | ---: |
| Allocator folders checked | 10 |
| Allocator folders with required files | 10 |
| Total source candidates | 220 |
| Accepted candidates | 0 |
| Rejected candidates | 220 |
| `no_allocation` rows | 10 |
| `CASH` allocator rows | 10 |

Each allocator `portfolio_summary.csv` reported:

- `portfolio_label`: `no_allocation`
- `accepted_candidate_count`: `0`
- `no_allocation_reason`: `all_candidates_rejected`

Each allocator `portfolio_allocation.csv` used:

- `decision_label`: `no_allocation`
- `ticker`: `CASH`
- `allocation_weight`: `1.0`
- `invested_weight`: `0.0`
- `cash_weight`: `1.0`

This confirms conservative gating and schema compatibility on real saved Quant Core outputs. It does not demonstrate active portfolio construction because no allocation candidates survived.

## Router Output Summary

Across the 10 router outputs, `route_decision.csv` produced 10 total rows. All rows were `no_candidate` routes with no active allocation.

Route label count:

| Route label | Count |
| --- | ---: |
| `no_candidate` | 10 |

Each router row used:

- `route_label`: `no_candidate`
- `ticker`: `CASH`
- `allocation_weight`: `0.0`
- `candidate_score`: `0.0`
- `route_score`: `0.0`
- `reason`: `all_candidates_rejected`

This confirms that Phase 3 Router v1 preserved the allocator's no-allocation state instead of forcing a route candidate.

## Interpretation

The validated chain was:

```text
Quant Core diagnostic outputs
-> Portfolio Allocator v1 no_allocation state
-> Phase 3 Router v1 no_candidate route
```

This is a successful conservative smoke validation, not a failure. The allocator and router processed real saved artifacts, wrote their expected output surfaces, preserved governance boundaries, and did not emit recommendation labels.

The observed no-allocation/no-candidate behavior means the current gates were conservative for this 10-seed smoke sample. It should be treated as evidence that the governance path is working, not as evidence of market edge or production readiness.

## Limitations

- This was only a 10-seed smoke check.
- No active allocation candidates were observed.
- This was not medium or `decision_core` validation.
- This was not production trading validation.
- This did not generate final recommendations.
- The allocator and router are deterministic heuristic layers, not learned meta-models.

## Recommended Next Step

Run a medium / `decision_core` validation after confirming schema stability on the allocator and router real-output path. If the project needs active allocation candidates for research review, adjust thresholds only as a separate governed experiment with explicit before/after documentation, leakage controls, and no claim of improved accuracy unless separately validated.

## Appendix A - Files Used

Source root:

```text
artifacts\phase3_real_smoke
```

Expected allocator folders:

```text
seed_000001_allocator
seed_000002_allocator
seed_000003_allocator
seed_000004_allocator
seed_000005_allocator
seed_000006_allocator
seed_000007_allocator
seed_000008_allocator
seed_000009_allocator
seed_000010_allocator
```

Expected router folders:

```text
seed_000001_router
seed_000002_router
seed_000003_router
seed_000004_router
seed_000005_router
seed_000006_router
seed_000007_router
seed_000008_router
seed_000009_router
seed_000010_router
```
