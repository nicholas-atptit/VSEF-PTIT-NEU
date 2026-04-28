# VSEF Broader Feature Governance Audit
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Audit |
| Created / authored | Sunday, 2026-04-26 00:00:00 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-04-28 22:34:05 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `8800ce6e3780c7978856737e70cb5e3b999eacee` |
| Timestamp source | existing document date |
| Status | Active |

## Purpose

This audit extends the earlier SSI-only governance check to a four-ticker walk-forward run. It reviews the existing diagnostic outputs for coefficient stability, feature importance stability, linear-vs-importance alignment, feature governance flags, and context coverage.

This is a research governance audit only. It does not add model families, change model governance status, remove features, prove absence of leakage, or claim improved trading performance.

## Command Used

```bash
python scripts/run_walkforward_all_models_stacking_eval.py --tickers SSI FPT ACB HPG --history-start 2020-12-21 --history-end 2025-02-28 --initial-train-start 2020-12-21 --initial-train-end 2024-12-31 --forecast-start 2025-01-02 --forecast-end 2025-01-24 --horizons short_5d --step-sizes 1 --algorithms cart,xgboost,lightgbm --output-dir outputs/walkforward_governance_audit_broader --max-workers 1 --max-depth 3 --meta-min-samples 1 --epochs 1
```

## Output Directory

Primary diagnostic output directory:

```text
outputs/walkforward_governance_audit_broader/ssi_fpt_acb_hpg/_combined_internal/csv/
```

Step-level outputs were also written under:

```text
outputs/walkforward_governance_audit_broader/ssi_fpt_acb_hpg/step_1/
```

## Data Source Status

Real provider data was not used at runtime. The environment reported `vnstock_data_not_installed`, and the runner used local CSV fallback files.

| ticker | source | rows | available range |
| --- | --- | ---: | --- |
| `ACB` | `csv_fallback` | 1044 | 2020-12-21 through 2025-02-28 |
| `FPT` | `csv_fallback` | 1044 | 2020-12-21 through 2025-02-28 |
| `HPG` | `csv_fallback` | 1044 | 2020-12-21 through 2025-02-28 |
| `SSI` | `csv_fallback` | 1044 | 2020-12-21 through 2025-02-28 |

The local CSV cache is a semi-real repository artifact, not a live-provider fetch in this run.

## Generated CSV Files

The combined diagnostic folder contained:

- `linear_coefficient_diagnostics.csv`
- `linear_coefficient_stability_summary.csv`
- `feature_importance_diagnostics.csv`
- `feature_importance_stability_summary.csv`
- `linear_vs_importance_feature_comparison.csv`
- `feature_governance_review.csv`
- `context_coverage_diagnostics.csv`
- `context_coverage_summary.csv`

It also contained standard prediction, summary, backtest, coverage, and metadata outputs.

## Governance Summary

| governance_category | count |
| --- | ---: |
| `safe_trailing` | 12 |
| `requires_review` | 9 |

| risk_level | count |
| --- | ---: |
| `low` | 11 |
| `medium` | 10 |

| recommended_action | count |
| --- | ---: |
| `keep` | 11 |
| `review_timing` | 9 |
| `keep_but_document` | 1 |

No `high` or `unknown` risk rows appeared in this broader audit. No `review_redundancy` or `exclude_until_verified` rows appeared.

## Timing-Review Features

The following features were flagged with `recommended_action = review_timing`.

| feature | category | risk | linear stability | importance stability | reason |
| --- | --- | --- | --- | --- | --- |
| `breadth_member_count` | `requires_review` | `medium` | yes | yes | Joined context feature requires source-date alignment review and availability metadata. |
| `breadth_thrust_10` | `requires_review` | `medium` | no | yes | Joined context feature requires source-date alignment review and availability metadata. |
| `declining_share` | `requires_review` | `medium` | no | yes | Joined context feature requires source-date alignment review and availability metadata. |
| `down_volume` | `requires_review` | `medium` | no | yes | Joined context feature requires source-date alignment review and availability metadata. |
| `new_high_low_spread_5` | `requires_review` | `medium` | no | yes | Joined context feature requires source-date alignment review and availability metadata. |
| `pct_above_ma20` | `requires_review` | `medium` | yes | yes | Joined context feature requires source-date alignment review and availability metadata. |
| `pct_above_ma50` | `requires_review` | `medium` | no | yes | Joined context feature requires source-date alignment review and availability metadata. |
| `up_down_volume_ratio_5` | `requires_review` | `medium` | no | yes | Joined context feature requires source-date alignment review and availability metadata. |
| `up_volume` | `requires_review` | `medium` | no | yes | Joined context feature requires source-date alignment review and availability metadata. |

These flags are conservative. They do not prove leakage.

## Context Coverage Summary

| ticker | horizon | fold_count | mean breadth missing | max breadth missing | mean foreign-flow missing | max foreign-flow missing | warning |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ACB` | `short_5d` | 17 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | `weak_coverage` |
| `FPT` | `short_5d` | 17 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | `weak_coverage` |
| `HPG` | `short_5d` | 17 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | `weak_coverage` |
| `SSI` | `short_5d` | 17 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | `weak_coverage` |

Breadth coverage was complete for this cached run. Foreign-flow coverage was absent for all four tickers in the selected window, so any `foreign_*` feature interpretation should remain weak unless a source with ticker/date coverage is supplied and reviewed.

## Stable Feature Observations

Top stable importance features by normalized importance included:

| model | task | feature | fold_count | mean normalized importance | top-10 ratio | stability |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `cart` | `profit` | `rolling_min_5` | 68 | 0.3701 | 1.0000 | `high` |
| `cart` | `trend` | `new_high_low_spread_5` | 68 | 0.3210 | 1.0000 | `high` |
| `cart` | `return` | `ema_50` | 68 | 0.2641 | 1.0000 | `high` |
| `cart` | `profit` | `breadth_thrust_10` | 68 | 0.2177 | 1.0000 | `high` |
| `lightgbm` | `profit` | `rolling_min_5` | 68 | 0.2004 | 1.0000 | `high` |

Top stable linear coefficient features by mean absolute coefficient included:

| model | task | feature | fold_count | mean abs coefficient | sign consistency | stability |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `linear` | `return` | `dist_ma_20` | 68 | 0.6382 | 1.0000 | `high` |
| `linear` | `return` | `close_return_10d` | 68 | 0.2801 | 1.0000 | `high` |
| `linear` | `return` | `bb_width` | 68 | 0.0928 | 1.0000 | `high` |
| `ridge` | `return` | `dist_ma_20` | 68 | 0.0427 | 1.0000 | `high` |
| `ridge` | `return` | `pct_above_ma20` | 68 | 0.0298 | 1.0000 | `high` |

These are diagnostic stability observations only. They do not establish predictive superiority, causality, or trading performance.

## Linear-vs-Importance Alignment

The comparison output labeled the following `short_5d` return features as `aligned_stable`:

- `bb_width`
- `breadth_member_count`
- `close_return_10d`
- `close_to_sma_200`
- `dist_ma_20`
- `dist_ma_60`
- `ema_50`
- `pct_above_ma20`
- `range_20`
- `rolling_max_60`
- `rolling_volatility_60`
- `turnover_ma_60`

Aligned stability means the feature appeared stable under both the linear coefficient and tree/boosting importance diagnostics. It does not prove the feature is causal or execution-ready.

## Runtime Warnings

The run emitted repeated LightGBM no-positive-gain warnings and scikit-learn feature-name warnings during inference. These warnings did not prevent the governance CSVs from being generated, but they limit performance interpretation and should be considered before drawing model-quality conclusions from this run.

## Changes Made

No feature registry, feature selection, feature engineering, model implementation, or model governance behavior was changed based on this audit.

No features were deleted or excluded automatically.

## Limitations

- The audit used four tickers, one horizon, one step size, and a January 2025 forecast window.
- Runtime data came from local CSV fallback files, not a live `vnstock_data` provider call.
- The output is broader than the SSI-only audit, but it is still a workflow and governance sample rather than broad market evidence.
- Coverage diagnostics measure source-row availability, not official source release timestamps.
- The review does not prove leakage absence, causality, or tradable performance.
- `weak_coverage` for foreign-flow context means foreign-flow interpretation should remain conservative.

## Next Recommended Task

Review why foreign-flow context coverage is absent in this cached run:

- confirm whether local foreign-flow artifacts exist for the selected ticker/date window
- decide whether future audits should require minimum context coverage thresholds for interpretation
- add report-level summaries for context coverage warnings
- keep coverage metadata as diagnostics, not model inputs

## Follow-Up: Foreign-Flow Coverage Investigation

`docs/audits/VSEF_FOREIGN_FLOW_COVERAGE_INVESTIGATION.md` investigates the absent foreign-flow coverage. The local `data/foreign_flow.csv` artifact exists but contains only a `TEST` row dated 2026-04-24, so it does not support exact ticker/date joins for the January 2025 `SSI`, `FPT`, `ACB`, and `HPG` audit window.

`docs/audits/VSEF_FOREIGN_FLOW_REAL_COVERAGE_AUDIT.md` documents a later provider-artifact rerun. The provider-backed artifact improved foreign-flow coverage from complete absence to partial availability, but fold-level coverage remained `weak_coverage`.
