# Phase F1.5 Narrow Forecast Rehab Decision
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Historical archive |
| Created / authored | Sunday, 2026-04-19 11:54:17 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-04-28 22:28:23 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `8800ce6e3780c7978856737e70cb5e3b999eacee` |
| Timestamp source | Git history |
| Status | Historical reference |

## Scope Locked

| dimension | in-scope | comparator-only | baseline-only | out/de-emphasized | reason |
| --- | --- | --- | --- | --- | --- |
| ticker_group | `small_banks` |  |  | `mixed_large_cap`, `vn100_subset` | F1 showed the only repeatable daily edge cluster lived in small banks. |
| horizons | `5`, `10` |  |  | `1` | Horizon 10 was the least-bad slice and horizon 5 remained materially stronger than horizon 1. |
| feature_families | `tech_core_v1`, `compact_v1`, `compact_plus_longlag_v1`, `compact_plus_longlag_v2` |  |  | `technical_plus_market`, `technical_plus_market_plus_sector`, `current_full`, `short_lag` | The narrowed rehab should increase edge density, not re-open broad context expansion. |
| model_families | `lightgbm`, `random_forest`, `xgboost` | `ets`, `sarimax` | `naive`, `moving_average` | `linear`, `ridge`, `lasso` | Tree models remained the primary rehab family; ETS and SARIMAX stayed as comparators only. |
| target_framing | `forward_return`, `direction_binary` |  |  | `forward_log_return`, `future_realized_volatility` | F1 left forward_return vs direction_binary unresolved and this cycle needed a clean comparison. |

## What Improved

- Narrowing to `small_banks` plus horizons `5/10` materially improved the post-cost profile versus the broader F1 rehab reference.
- The medium narrow run produced baseline positive-Sharpe share `96.43%` and elevated-cost positive-Sharpe share `76.79%`.
- Forward-return horizon `10` was the strongest narrow slice, with baseline median Sharpe `2.719614` and elevated-cost median Sharpe `2.083128`.
- The primary tree family improved materially in the narrowed scope:
  - `xgboost` + `lightgbm` baseline median Sharpe `2.185184`
  - `xgboost` + `lightgbm` elevated-cost median Sharpe `1.226288`
  - `xgboost` + `lightgbm` horizon-10 baseline positive-Sharpe share `100%`

## What Stayed Weak

- The edge is still narrow and not broad-daily robust.
- Comparator models still matter:
  - `ets` was the most stable model across the full narrow matrix.
  - `sarimax` remained economically credible in the narrow slice.
- Forecast quality and monetization still do not line up perfectly.
  - `tech_core_v1` underperformed on directional-accuracy summaries at horizon 10.
  - The same family still produced some of the best horizon-10 policy outcomes with `xgboost` and `lightgbm`.
- Horizon `5` remained more cost-sensitive than horizon `10`.

## Feature Readout

- Best balanced family across the full narrow matrix: `compact_plus_longlag_v1`
- Strongest primary-model horizon-10 slice: `tech_core_v1` with `xgboost` and `lightgbm`
- Practical takeaway:
  - keep `compact_plus_longlag_v1` as the default narrowed family
  - keep `tech_core_v1` as a challenge set for horizon-10 tree-model runs
  - keep `compact_plus_longlag_v2` as the secondary long-memory variant

## Target Framing Readout

- `forward_return` beat `direction_binary` inside the narrow scope on both usefulness and downstream credibility.
- `direction_binary` remained useful as a diagnostic framing, but not as the next default execution framing.
- Next-stage target framing should be `forward_return`.

## Recommendation

Direct recommendation: `freeze all but the best one or two model families`

Interpretation:

- Continue one more narrow rehab cycle.
- Keep `small_banks` only.
- Standardize on `forward_return`.
- Make horizon `10` the primary research horizon, with horizon `5` retained only as a guardrail.
- Freeze primary models to:
  - `xgboost`
  - `lightgbm`
- Keep comparators:
  - `ets`
  - optional `sarimax`
- Freeze baseline-only models to baseline monitoring and stop treating them as primary rehab candidates.

## Phase 3 Status

- Phase 3 remains blocked.
- The narrow slice is materially better than the broader F1 rehab, but the evidence is still too concentrated to justify routing, stacking, portfolio allocation, or deep sequence expansion.
