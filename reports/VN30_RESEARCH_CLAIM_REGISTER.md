# VN30 Research Claim Register

This file is an index over existing VN30 claim registers and claim boundaries. It does not add or upgrade empirical claims.

## Global Rules

- No trading, profitability, investment-recommendation, or live-deployment claim is made.
- No broad benchmark, model training, data fetch, or paper/DOCX generation is implied by this register.
- Claims must cite the exact scope, universe, frequency, horizon, date window, row count, baseline, and source artifact.
- Stock hourly, stock daily, index-only, top-k ranking, and paper-source artifacts are separate evidence families.

## Active Claim Sources

| evidence family | claim source | status |
| --- | --- | --- |
| VN30 hourly available-window and selected candidate | `reports/VN30_HOURLY_2015_BENCHMARK_CLAIM_REGISTER.md` | Active hourly boundary for available-window evidence. |
| VN30 hourly selected candidate | `reports/VN30_HOURLY_SELECTED_CANDIDATE_CLAIM_BOUNDARY.md` | Active boundary for the reproduced 61.51% L2 Logistic h40 selected candidate. |
| VN30 hourly selected candidate stability | `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_CLAIM_BOUNDARY.md` | Active stability boundary; target62 and final65 claims remain unsupported. |
| VN30 daily 2015 | `reports/VN30_DAILY_2015_CLAIM_REGISTER.md` | Active daily boundary; daily result remains below 60%. |
| VN30 index benchmark context | `reports/VN30_INDEX_BENCHMARK_CLAIM_REGISTER.md` | Active index-only boundary; index claims do not support stock claims. |
| VN30 hourly top-k ranking | `reports/VN30_HOURLY_2015_TOPK_RANKING_CLAIM_REGISTER.md` | Separate metric family; top-k results are not pooled directional accuracy. |
| VN30 stock-index joint panel | `reports/VN30_STOCK_INDEX_JOINT_PANEL_TRAINING_V1_CLAIM_REGISTER.md` | Joint-panel training boundary; separate from selected-candidate evidence. |

## Current Selected-Candidate Boundary

- Final pooled directional accuracy: 61.51%.
- Majority baseline accuracy: 50.44%.
- Majority baseline lift: +11.07 percentage points.
- Full VN30 stock coverage: 30/30.
- Claim level: exploratory improved baseline60 evidence.
- Target62 claim: no.
- Final65 claim: no.

The selected-candidate boundary remains governed by `reports/VN30_HOURLY_SELECTED_CANDIDATE_CLAIM_BOUNDARY.md` and `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_AUDIT_RESULT.md`.
