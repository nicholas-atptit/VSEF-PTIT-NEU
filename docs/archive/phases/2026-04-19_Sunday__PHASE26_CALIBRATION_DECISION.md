# Phase 2.6 Calibration Decision
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Historical archive |
| Created / authored | Sunday, 2026-04-19 01:39:56 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-04-28 22:28:23 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `8800ce6e3780c7978856737e70cb5e3b999eacee` |
| Timestamp source | Git history |
| Status | Historical reference |

## Question

Is the current weakness mainly caused by thresholding and sizing policy, or is the forecast edge itself too weak to monetize after costs?

## Short Answer

Calibration improved the decision layer, but it did not fix the core economic problem.

- bounded regime-aware adaptive sizing beat the old fixed-fraction and Phase 2.5-style current risk-only path on drawdown and turnover
- regime execution helped enough in the medium matrix to keep it viable as an execution input
- the full bounded matrix still produced negative median Sharpe across every policy variant
- forecast-versus-policy analysis showed some monetizable pockets, but not enough stable edge across windows, horizons, and ticker groups

## Default Candidate

Current default candidate for any further bounded benchmarking:

- `regime_threshold_adaptive_drawdown::adaptive_current`

Reason:

- it was the best balanced trade-off across median Sharpe, median CAGR, median max drawdown, and turnover
- lighter drawdown haircuts produced a slightly less-bad median Sharpe, but gave back too much drawdown control and turnover discipline
- elevated-cost robustness favored the same regime-plus-risk family, but with `adaptive_lighter_vol`

## Recommendation

- direct recommendation: `stop policy work and revisit forecast layer`
- Phase 3 remains blocked

## Why Phase 3 Is Still Blocked

- the threshold sweep found a least-bad setting around `0.010`, but all threshold bands remained economically weak
- fixed-fraction was simpler, but it did not solve the post-cost weakness
- adaptive sizing can be improved with lighter or bounded variants, but calibration gains were not large enough to rescue the full stack
- regime conditioning improved drawdown frequently and Sharpe more than half the time versus risk-only, but the entire calibrated stack still stayed below a credible go/no-go bar

## Artifact Reference

See:

- `artifacts/phase26_calibration/summary.md`
- `artifacts/phase26_calibration/phase26_report.md`
- `artifacts/phase26_calibration/policy_ablation_summary.csv`
- `artifacts/phase26_calibration/threshold_calibration_summary.csv`
- `artifacts/phase26_calibration/sizing_calibration_summary.csv`
- `artifacts/phase26_calibration/regime_value_summary.csv`
- `artifacts/phase26_calibration/forecast_vs_policy_summary.csv`
