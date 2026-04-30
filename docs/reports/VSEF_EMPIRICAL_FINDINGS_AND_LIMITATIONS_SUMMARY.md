# VSEF Empirical Findings and Limitations Summary
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Empirical findings verification report |
| Created / authored | Thursday, 2026-04-30 11:27:01 ICT (UTC+07:00) |
| Last updated | Thursday, 2026-04-30 11:27:01 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `e9062c13edf34a63637e1b6f208cf4f871ebecef` |
| Timestamp source | Local verification session |
| Status | Active |

## Scope Clarification

This report validates the archived empirical findings summary against the exact saved CSV artifact paths listed below. It does not rerun model pipelines, does not alter model code, and does not treat generated artifacts as source-controlled documentation.

The base summary is `docs/archive/root/2026-04-15_Wednesday__RESEARCH_FINDINGS_AND_LIMITATIONS.md`. Its claims are preserved here as historical summary claims, but they are marked `unverified` where the corresponding CSV artifact is missing from the current checkout or cannot be read.

No listed CSV artifact was present at the requested path during this verification pass. The repository root also did not contain an `artifacts/` directory, and a filename search did not find matching CSV filenames elsewhere in the checkout. Therefore, no header, first-row, schema, or value-level verification could be performed for the requested artifact files.

## Artifact Verification Table

| Artifact | Exists at requested path | Header read | First rows read | Verification status |
| --- | --- | --- | --- | --- |
| `artifacts/backtest_model_comparison/overall_model_ranking.csv` | No | No | No | `unverified`: missing artifact |
| `artifacts/backtest_forward_return/overall_horizon_ranking.csv` | No | No | No | `unverified`: missing artifact |
| `artifacts/strategy_backtest/summary/overall_strategy_ranking.csv` | No | No | No | `unverified`: missing artifact |
| `artifacts/dual_task/summary/cross_task_model_ranking.csv` | No | No | No | `unverified`: missing artifact |
| `artifacts/combined_signal/summary/overall_combined_signal_summary.csv` | No | No | No | `unverified`: missing artifact |
| `artifacts/regime_aware_analysis/summary/overall_regime_summary.csv` | No | No | No | `unverified`: missing artifact |
| `artifacts/walk_forward_regime_robustness/summary/overall_robustness_report.csv` | No | No | No | `unverified`: missing artifact |
| `artifacts/meta_selector/summary/overall_meta_selector_report.csv` | No | No | No | `unverified`: missing artifact |
| `artifacts/meta_selector_audit/benchmark_audit_report.csv` | No | No | No | `unverified`: missing artifact |
| `artifacts/context_meta_selector/summary/overall_context_selector_report.csv` | No | No | No | `unverified`: missing artifact |

## Verified Findings

No empirical winner, metric value, or selector result from the archived summary could be verified from the requested CSV artifacts because all ten artifact files are missing from the current checkout.

The following narrower findings are verified by local repository inspection:

- The archived summary exists at `docs/archive/root/2026-04-15_Wednesday__RESEARCH_FINDINGS_AND_LIMITATIONS.md`.
- The requested report path did not already exist before this report was created.
- The requested CSV artifact paths are absent in the current checkout.
- Later 15-year reports in `docs/reports/` and `docs/audits/` document a separate long-window, multi-horizon audit series where `stacking_final` is strongest on MAE/RMSE across the evaluated horizons, while directional leadership remains horizon-sensitive.

These repository-context findings do not verify the older artifact-specific CSV claims. They only constrain how the older claims should be interpreted.

## Unverified or Missing Artifact Claims

The claims below are inherited from the archived summary. They should not be cited as current CSV-verified facts unless the matching artifacts are restored and rechecked.

| Artifact source | Archived claim summary | Current verification result |
| --- | --- | --- |
| `artifacts/backtest_model_comparison/overall_model_ranking.csv` | `naive_previous_close` has the best overall average rank; `cart` slightly edges naive on RMSE; `xgboost` and `lightgbm` are competitive but not decisive winners in close-level framing. | `unverified`: file missing |
| `artifacts/backtest_forward_return/overall_horizon_ranking.csv` | `xgboost` is best RMSE on `3d`, `5d`, and `20d`; `xgboost` is best MAPE on `3d` and `20d`; `lightgbm` leads `5d` MAPE; no horizon has `any_model_beats_naive_overall = True`. | `unverified`: file missing |
| `artifacts/strategy_backtest/summary/overall_strategy_ranking.csv` | `20d / cart / threshold 0.02` is the best saved sample with `best_total_return = 0.0387`, `best_sharpe_ratio = 1.6586`, `any_model_beats_buy_and_hold = True`, and `any_model_positive_net_return_after_costs = True`; `3d / cart / threshold 0.02` is weaker but positive; `5d` remains negative after costs. | `unverified`: file missing |
| `artifacts/dual_task/summary/cross_task_model_ranking.csv` | `xgboost` is best regression RMSE on `3d`, `5d`, and `20d`; best classification model is `lightgbm` on `3d`, `cart` on `5d`, and `cart` on `20d`; `20d` has the highest saved actionability score. | `unverified`: file missing |
| `artifacts/combined_signal/summary/overall_combined_signal_summary.csv` | Combined signals help on selected slices such as `3d / xgboost`, `5d / lightgbm`, and `5d / xgboost`, but gains are mixed or absent for other slices. | `unverified`: file missing |
| `artifacts/regime_aware_analysis/summary/overall_regime_summary.csv` | Best regression model by regime is `xgboost / 3d` in bull, bear, and sideway regimes; best classifier differs by regime; `combined_weighted_linear_gated` remains the best combined method family while exact winners vary. | `unverified`: file missing |
| `artifacts/walk_forward_regime_robustness/summary/overall_robustness_report.csv` | `xgboost` is most frequent regression winner with `low` stability; `cart` is most frequent classification winner with `low` stability; `combined_weighted_linear_gated` is most frequent combined-method winner; `5d` is most frequent winning horizon with `low` stability. | `unverified`: file missing |
| `artifacts/meta_selector/summary/overall_meta_selector_report.csv` | Best selector mode is `fallback_global`; selector edge is narrow and labeled `low` stability; most frequent selected setup is `xgboost+20d+predicted_return`. | `unverified`: file missing |
| `artifacts/meta_selector_audit/benchmark_audit_report.csv` | Recomputed summaries match saved summaries exactly; no grouping bug was found; some baseline rows are effectively identical because they resolve to the same selected `model/horizon` row universe. | `unverified`: file missing |
| `artifacts/context_meta_selector/summary/overall_context_selector_report.csv` | Best context selector mode is `context_knn_selector`; fixed baselines are stronger overall; context selector does not clearly beat the simpler regime selector; most frequent selected setup is `xgboost+20d+predicted_return`. | `unverified`: file missing |

## Relationship to 15-Year Stacking Audits

Older saved artifacts show that XGBoost, CART, and naive baselines remain strong in specific fixed-window, forward-return, strategy, and selector layers. Later 15-year multi-horizon stacking audits show that `stacking_final` is strongest on MAE/RMSE in that specific long-window all-model setup. Together, these results support a conservative conclusion: method-family patterns are more stable than exact model-horizon-policy winners.

This wording is intentionally conservative. The older artifact claims are not verified by the missing CSVs in this checkout, while the later 15-year reports are separate documented audits. The combined interpretation should not imply that one model, horizon, selector, or trading policy universally wins.

The later audit documents also keep directional performance separate from point-error performance. They report `stacking_final` strength on MAE/RMSE while showing that directional accuracy and F1 leadership can shift by horizon and ticker basket. That distinction prevents the older fixed-window and selector claims from contradicting the 15-year stacking reports.

## Current Defensible Claims

- The current checkout does not contain the ten requested CSV artifacts, so the archived summary cannot be treated as newly CSV-verified.
- The archived summary remains useful as a historical claim inventory, not as current artifact evidence.
- The later 15-year multi-horizon audit series supports a limited claim that `stacking_final` is strongest on MAE/RMSE in that long-window all-model setup.
- Directional and strategy-style conclusions remain more fragile than point-error conclusions.
- Method-family patterns are more defensible than exact model-horizon-threshold or selector-policy winners.
- VSEF should be described as a research and evaluation framework, not as proof of live-trading readiness.

## Claims Not Yet Defensible

- A universal best model across all horizons, tasks, regimes, and evaluation layers.
- A current artifact-verified claim that `naive_previous_close`, `cart`, `xgboost`, `lightgbm`, or any selector mode wins the older saved artifact tables.
- The archived strategy values, including `best_total_return = 0.0387` and `best_sharpe_ratio = 1.6586`, without the matching CSV.
- Robust selector superiority over strong fixed baselines.
- Stable bear-regime conclusions.
- Production trading performance, causality, or live deployment readiness.

## Recommended Next Steps

1. Restore the ten CSV artifacts, or regenerate them through the documented workflows without committing generated artifact outputs.
2. Rerun this verification after the CSVs are present and record each file's header, first rows, schema, and checked values.
3. Keep older fixed-window artifact findings and later 15-year stacking audit findings in separate evidence buckets.
4. Add a lightweight verification script or checklist if these summaries need to be refreshed repeatedly.
5. Continue to treat exact winners as conditional until broader, reproducible, long-window evidence supports them.
