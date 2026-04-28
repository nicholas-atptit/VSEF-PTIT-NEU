# Research Findings and Limitations
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Historical archive |
| Created / authored | Wednesday, 2026-04-15 11:19:48 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-04-28 22:51:14 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `ef20ce73b466d75a61ca4768d4f4129405df7fb0` |
| Timestamp source | Git history |
| Status | Archived |

This document summarizes the current empirical findings from the saved artifacts and the main limitations that still constrain the system.

The tone here is intentionally conservative. The repository has become much stronger as a research framework, but the results do not justify strong claims about a universal winning setup.

## 1. Current Findings

### 1.1 Real-data fixed-window close backtest

Current result from `artifacts/backtest_model_comparison/overall_model_ranking.csv`:

- `naive_previous_close` still has the best overall average rank.
- `cart` slightly edges out naive on RMSE, but not on MAPE or directional accuracy.
- `xgboost` and `lightgbm` are competitive, but neither is a decisive winner in that close-level framing.

Takeaway:

- The repo now has a real and honest baseline.
- Simple persistence is still hard to beat on broad aggregate metrics.

### 1.2 Forward-return forecasting

Current result from `artifacts/backtest_forward_return/overall_horizon_ranking.csv`:

- `xgboost` is the best RMSE model on `3d`, `5d`, and `20d`.
- `xgboost` is also the best MAPE model on `3d` and `20d`.
- `lightgbm` edges `5d` MAPE.
- No horizon has `any_model_beats_naive_overall = True`.

Takeaway:

- Forward-return modeling is informative, but the naive flat-return baseline remains strong overall.
- There is still no universally dominant return model once all metrics are considered together.

### 1.3 Strategy backtest layer

Current result from `artifacts/strategy_backtest/summary/overall_strategy_ranking.csv`:

- `20d / cart / threshold 0.02` is the best saved sample:
  - `best_total_return = 0.0387`
  - `best_sharpe_ratio = 1.6586`
  - `any_model_beats_buy_and_hold = True`
  - `any_model_positive_net_return_after_costs = True`
- `3d / cart / threshold 0.02` is positive but much weaker.
- `5d` remains unattractive in the saved sample because the best total return is still negative after costs.

Takeaway:

- Some strategy-style configurations are usable in the specific window tested.
- That does not make them robust. The strategy evidence is still short-window and sample-dependent.

### 1.4 Dual-task forecasting

Current result from `artifacts/dual_task/summary/cross_task_model_ranking.csv`:

- `xgboost` is the best regression model by RMSE on `3d`, `5d`, and `20d`.
- The best classification model differs by horizon:
  - `lightgbm` on `3d`
  - `cart` on `5d`
  - `cart` on `20d`
- `20d` has the highest saved actionability score among the three horizons.

Takeaway:

- Best return forecasting and best profit classification are not the same problem.
- The repo is right to keep them as separate evaluation tasks.

### 1.5 Combined-signal analysis

Current result from `artifacts/combined_signal/summary/overall_combined_signal_summary.csv`:

- Combined signals help on some slices, but not uniformly.
- Clearer positive cases:
  - `3d / xgboost` improved over return-only and probability-only ranking
  - `5d / lightgbm` improved over both
  - `5d / xgboost` improved over both
- Mixed or weak cases:
  - `3d / cart` and `3d / lightgbm` did not improve
  - `20d / lightgbm` and `20d / xgboost` improved versus return-only but not versus probability-only
  - `20d / cart` did not improve

Takeaway:

- Combining `predicted_return` and `predicted_profit_probability` can help signal quality.
- The benefit is local, not universal.

### 1.6 Regime-aware analysis

Current result from `artifacts/regime_aware_analysis/summary/overall_regime_summary.csv`:

- Best regression model by regime:
  - `xgboost / 3d` in `bull`
  - `xgboost / 3d` in `bear`
  - `xgboost / 3d` in `sideway`
- Best classification model by regime:
  - `xgboost / 5d` in `bull`
  - `cart / 20d` in `bear`
  - `cart / 20d` in `sideway`
- Best combined method is still `combined_weighted_linear_gated`, but the winning model and horizon vary by regime.

Takeaway:

- The best regression model is relatively stable.
- The best classifier and best combined setup are regime-dependent.
- Sideway remains harder to summarize cleanly than bull or bear.

### 1.7 Walk-forward robustness

Current result from `artifacts/walk_forward_regime_robustness/summary/overall_robustness_report.csv`:

- `xgboost` is the most frequent regression winner, but stability is still labeled `low`.
- `cart` is the most frequent classification winner, also with `low` stability.
- `combined_weighted_linear_gated` is the most frequent combined-method winner.
- `5d` is the most frequent winning horizon in the saved walk-forward summary, but still with `low` stability.

Takeaway:

- The most repeatable pattern is at the method-family level.
- Exact model or horizon winners remain sample-sensitive.

### 1.8 Regime-conditioned meta-selector

Current result from `artifacts/meta_selector/summary/overall_meta_selector_report.csv`:

- Best selector mode in the saved report is `fallback_global`.
- The meta-selector edge versus fixed setups is narrow and explicitly labeled `low` stability.
- The most frequent selected setup is `xgboost+20d+predicted_return`.

Takeaway:

- Adaptive selection helps only slightly in the saved folds.
- Much of the selector behavior still collapses back toward globally strong families.

### 1.9 Benchmark audit

Current result from `artifacts/meta_selector_audit/benchmark_audit_report.csv`:

- Recomputed summaries match the saved summaries exactly.
- No grouping bug was found.
- Some baseline rows are effectively identical in practice because they resolve to the same selected `model/horizon` row universe.

Takeaway:

- The suspiciously similar baseline summaries were mostly an interpretation issue, not a broken aggregation.
- Broad averages and top-k metrics should be read together.

### 1.10 Context-conditioned selector

Current result from `artifacts/context_meta_selector/summary/overall_context_selector_report.csv`:

- Best context selector mode is `context_knn_selector`.
- Fixed baselines are still stronger overall.
- The context selector also does not clearly beat the simpler regime selector layer.
- The most frequent context-selected setup is still `xgboost+20d+predicted_return`.

Takeaway:

- Richer context features did not produce a broad selector improvement in the saved sample.
- The current gains are narrow and still sample-sensitive.

## 2. What Looks More Stable

The following findings look more repeatable than the rest:

- `xgboost` is frequently strong on the regression side.
- `combined_weighted_linear_gated` is the most repeatable combined-method family in walk-forward summaries.
- Method-family conclusions are more stable than exact configuration conclusions.
- The classification branch remains materially different from the regression branch, which supports the dual-task design.

## 3. What Still Looks Unstable

The following findings remain unstable or conditional:

- the exact best model-horizon pair
- the exact best classifier
- whether `3d`, `5d`, or `20d` is best overall
- whether adaptive selectors truly beat strong fixed baselines
- whether bear-regime conclusions can be trusted beyond the limited saved folds

## 4. Limitations

### 4.1 Short and uneven evaluation coverage

- The single-window workflows use short recent holdouts.
- Even the walk-forward stack covers a limited number of folds.
- That makes apparent winners vulnerable to period-specific noise.

### 4.2 Sparse bear samples

- Bear observations are still relatively scarce in the saved studies.
- Some bear conclusions look strong only because there are too few folds and too few matched rows.

### 4.3 Calibration is still imperfect

- Profit probabilities are useful ranking inputs, but the calibration summaries remain uneven.
- Some probability buckets are overconfident or underconfident depending on regime and model.

### 4.4 Combined-layer gains are selective

- The combined layer improves signal quality in some horizon/model slices.
- It does not beat return-only or probability-only ranking everywhere.

### 4.5 Classifier quality is still limited

- Positive-class precision is not yet strong enough to treat the classification branch as a hard gating oracle.
- The classification branch is directionally useful, but not decisively reliable.

### 4.6 Strategy conclusions are sample-limited

- The strategy layer shows that some forecast configurations can survive costs in the saved sample.
- That is still a paper backtest, not proof of robust tradability.

### 4.7 No universal winner

- There is still no single model, horizon, or selector mode that wins broadly across all layers and all conditions.

## 5. Recommended Next Steps

### 5.1 Increase history and fold coverage

- Extend walk-forward evaluation over more periods.
- Prioritize periods with deeper bearish coverage.

### 5.2 Improve classifier calibration and precision

- Add calibration-specific tuning or post-processing.
- Focus on positive-class precision and false-positive control, not just accuracy.

### 5.3 Add stronger exogenous context

- Benchmark-relative features are already in place.
- The next gains are more likely to come from better external context than from more selector complexity alone.

Examples:

- cleaner benchmark and sector features
- event and sentiment signals
- uncertainty-aware features
- model-dispersion features

### 5.4 Keep benchmark audits in the loop

- The selector audit did not find a grouping bug, but it did show that some baselines become effectively identical in practice.
- Future selector work should continue to write audit traces so summary tables stay interpretable.

### 5.5 Prefer method-family decisions over fragile exact picks

- Current evidence supports using model families and combination families as the more stable unit of comparison.
- For example, "boosting regressor + gated combined method" is more defensible than claiming one exact model-horizon-threshold tuple is robust.

## 6. Bottom Line

The repository has matured into a strong research framework, but the empirical picture is still mixed.

What is defensible today:

- the evaluation stack is much more rigorous than before
- some improvements are real in specific slices
- combined-signal and regime-aware views add useful structure
- exact winners are still unstable

What is not yet defensible:

- claiming a universal best model
- claiming the selector layer is robustly superior
- claiming bear-regime conclusions are well established
- treating the saved strategy outcomes as production-ready trading evidence
