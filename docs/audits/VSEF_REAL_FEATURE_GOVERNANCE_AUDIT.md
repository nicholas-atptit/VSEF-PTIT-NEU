# VSEF Real Feature Governance Audit
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

This audit runs a small walk-forward governance check using the feature diagnostics added in the preceding branches. The goal is to verify that `feature_governance_review.csv` is produced by the workflow and to review the resulting feature flags.

This audit is for research governance only. It does not change model governance status, remove features, or claim improved trading performance.

## Command Used

```bash
python scripts/run_walkforward_all_models_stacking_eval.py --tickers SSI --history-start 2020-12-21 --history-end 2025-01-20 --initial-train-start 2020-12-21 --initial-train-end 2024-12-31 --forecast-start 2025-01-02 --forecast-end 2025-01-10 --horizons short_5d --step-sizes 1 --algorithms cart,xgboost,lightgbm --output-dir outputs/walkforward_governance_audit --max-workers 1 --max-depth 3 --meta-min-samples 1 --epochs 1
```

## Output Directory

Primary diagnostic output directory:

```text
outputs/walkforward_governance_audit/ssi/_combined_internal/csv/
```

The script also writes step-level standard performance outputs under:

```text
outputs/walkforward_governance_audit/ssi/step_1/
```

## Data Source

Real provider data was not used at runtime. The environment reported `vnstock_data_not_installed`, and the runner used the local CSV fallback:

- ticker: `SSI`
- source: `csv_fallback`
- rows: `1020`
- available range: `2020-12-21` through `2025-01-20`

The CSV cache is a semi-real local artifact, not a live-provider fetch in this run.

## Generated CSV Files

The combined diagnostic folder contained:

- `linear_coefficient_diagnostics.csv`
- `linear_coefficient_stability_summary.csv`
- `feature_importance_diagnostics.csv`
- `feature_importance_stability_summary.csv`
- `linear_vs_importance_feature_comparison.csv`
- `feature_governance_review.csv`

It also contained the standard prediction, summary, backtest, coverage, and metadata CSV/JSON outputs.

## Governance Category Summary

| governance_category | count |
| --- | ---: |
| `requires_review` | 10 |
| `safe_trailing` | 11 |

## Risk-Level Summary

| risk_level | count |
| --- | ---: |
| `low` | 11 |
| `medium` | 10 |

No `high` or `unknown` risk rows appeared in this small audit.

## Recommended Action Summary

| recommended_action | count |
| --- | ---: |
| `keep` | 11 |
| `review_timing` | 10 |

No `review_redundancy` or `exclude_until_verified` rows appeared in this small audit.

## Timing-Review Features

The following features were flagged with `recommended_action = review_timing`.

| feature | category | risk | linear stability | importance stability | reason |
| --- | --- | --- | --- | --- | --- |
| `breadth_member_count` | `requires_review` | `medium` | yes | yes | External or joined context feature requires date-alignment confirmation. |
| `breadth_thrust_10` | `requires_review` | `medium` | no | yes | External or joined context feature requires date-alignment confirmation. |
| `declining_share` | `requires_review` | `medium` | no | yes | External or joined context feature requires date-alignment confirmation. |
| `down_volume` | `requires_review` | `medium` | no | yes | External or joined context feature requires date-alignment confirmation. |
| `new_high_low_spread_5` | `requires_review` | `medium` | no | yes | External or joined context feature requires date-alignment confirmation. |
| `pct_above_ma20` | `requires_review` | `medium` | yes | yes | External or joined context feature requires date-alignment confirmation. |
| `pct_above_ma50` | `requires_review` | `medium` | no | yes | External or joined context feature requires date-alignment confirmation. |
| `turnover_ma_60` | `requires_review` | `medium` | yes | yes | Flow or context-like feature should have timing assumptions documented. |
| `up_down_volume_ratio_5` | `requires_review` | `medium` | no | yes | External or joined context feature requires date-alignment confirmation. |
| `up_volume` | `requires_review` | `medium` | no | yes | External or joined context feature requires date-alignment confirmation. |

These flags are conservative. They do not prove leakage.

## High, Unknown, Redundancy, and Exclusion Findings

- `risk_level = high`: none observed
- `risk_level = unknown`: none observed
- `recommended_action = review_redundancy`: none observed
- `recommended_action = exclude_until_verified`: none observed

## Changes Made

No feature registry, feature selection, feature engineering, or model governance behavior was changed based on this audit.

No features were deleted or excluded automatically.

## Limitations

- The audit used one ticker, one horizon, one step size, and a short forecast period.
- Runtime data came from the local CSV fallback, not a live `vnstock_data` provider call.
- The output should be treated as a workflow validation and feature-governance sample, not broad market evidence.
- Review labels are rule-based and conservative; they are not proof of leakage, causality, or tradable performance.
- LightGBM emitted repeated no-positive-gain warnings in this short-window run; those warnings do not invalidate the governance CSV but do limit performance interpretation.

## Next Recommended Task

Review the timing assumptions for market breadth and flow-style features:

- confirm source-date alignment for breadth-derived fields
- document whether `turnover_ma_60` should remain a timing-review feature or be treated as a local trailing OHLCV-derived feature
- add targeted tests around source-date and forward-fill behavior for context joins before changing feature-selection rules

## Follow-Up: Context Timing Governance

The follow-up branch `vsef-context-timing-governance` documents the timing policy for the flagged breadth and flow-style features. It confirms that `turnover_ma_60` is local trailing OHLCV-derived under the current feature builder, while joined breadth and foreign-flow features remain timing-review candidates until source-date availability is governed more explicitly.

## Follow-Up: Context Availability Metadata

The follow-up branch `vsef-context-availability-metadata` adds explicit support columns for breadth and foreign-flow source availability. These columns separate measured zero values from missing-context fallback values without automatically changing model features or feature-selection behavior.

## Follow-Up: Broader Governance Audit

`docs/audits/VSEF_BROADER_FEATURE_GOVERNANCE_AUDIT.md` extends this SSI-only audit to `SSI`, `FPT`, `ACB`, and `HPG` over a longer January 2025 window using the same governance and context-coverage diagnostics.
