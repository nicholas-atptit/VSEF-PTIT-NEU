# VSEF 15-Year Daily and Multi-Horizon Walk-Forward Technical Report

Date: 2026-04-28

Branch: `vsef-15y-technical-report`

Source audit branches:

- `vsef-15y-daily-walkforward-audit`
- `vsef-15y-daily-audit-no-foreign-flow`
- `vsef-15y-multihorizon-audit`
- `vsef-15y-broader-ticker-audit`

Source documents:

- `docs/audits/VSEF_15Y_DAILY_WALKFORWARD_AUDIT.md`
- `docs/audits/VSEF_15Y_DAILY_WALKFORWARD_NO_FOREIGN_FLOW_AUDIT.md`
- `docs/audits/VSEF_15Y_MULTIHORIZON_WALKFORWARD_AUDIT.md`
- `docs/audits/VSEF_15Y_BROADER_TICKER_MULTIHORIZON_AUDIT.md`
- `docs/governance/VSEF_FOREIGN_FLOW_DISABLE_MODE.md`
- `docs/governance/VSEF_CONTEXT_COVERAGE_DIAGNOSTICS.md`
- `docs/governance/VSEF_FOREIGN_FLOW_ARTIFACT_POLICY.md`
- `docs/README.md`
- `README.md`

## Executive Summary

This report synthesizes the 15-year VSEF walk-forward audit series for Vietnamese equity forecasting over the 2010-2025 historical window. The evaluated tickers are `SSI`, `FPT`, `BVH`, and `VNM`. The OHLCV data source is a staged local refresh produced through the canonical `vnstock_data` provider and read through the explicit staged data directory `tmp\ohlcv_15y_refresh_probe_data`.

The evaluation design uses daily walk-forward steps with `--step-sizes 1`, meaning each available trading session in the forecast window is evaluated rather than each calendar day. The model set is limited to the existing CART, XGBoost, LightGBM, and `stacking_final` components. No new model family is introduced in this report.

Foreign-flow context is intentionally disabled because no governed long-window foreign-flow artifact exists for the requested ticker/date window. Disabled foreign-flow is an explicit exclusion control, not evidence of complete foreign-flow coverage. Breadth context remains measured and interpretable.

Across the evaluated horizons, `stacking_final` consistently achieves the lowest MAE and RMSE. Directional accuracy is more horizon-sensitive and is not consistently dominated by one model. These findings are research and evaluation evidence only. They do not prove live-trading readiness or trading performance.

A follow-up broader-ticker audit is documented in `docs/audits/VSEF_15Y_BROADER_TICKER_MULTIHORIZON_AUDIT.md`. It requested `SSI`, `FPT`, `BVH`, `VNM`, `ACB`, `HPG`, `MWG`, `DGC`, `VCB`, and `VIC`; coverage supported the full 2010-2025 window for 8 tickers, while `MWG` and `DGC` were excluded because their staged histories begin in 2014. The broader fallback run preserves the `stacking_final` MAE/RMSE pattern and keeps directional performance horizon-sensitive.

## 1. Project Context

VSEF stands for Vietnam Stock Evaluation and Forecasting Framework. The repository is a private and proprietary research framework for Vietnamese market data evaluation, forecasting experiments, and model-governance diagnostics.

The framework emphasizes data governance, explicit walk-forward evaluation, model comparison, feature and context coverage diagnostics, and conservative interpretation. This report consolidates the recent 15-year audit sequence into one technical narrative suitable for supervisor or lecturer review.

This report does not claim live-trading readiness. Backtest-style metrics, prediction summaries, and diagnostic outputs are treated as empirical evaluation evidence rather than deployable trading proof.

## 2. Data Source and Coverage

Runtime and data source:

| item | value |
| --- | --- |
| Python runtime | `C:\Users\luong\.venv\Scripts\python.exe` |
| Provider | canonical `vnstock_data` |
| Tickers | `SSI`, `FPT`, `BVH`, `VNM` |
| Staged OHLCV directory | `tmp\ohlcv_15y_refresh_probe_data` |
| File range | `2009-11-23` to `2026-04-24` |
| Run-loaded range | `2010-01-04` to `2025-12-31` |
| Staged rows per ticker | `4097` |
| Matched requested business dates | `3993` |
| Missing generic business dates | `181` |
| Coverage rate | `0.956636` |
| All tickers support requested window | yes |

Coverage by ticker:

| ticker | staged rows | file min date | file max date | run-loaded range | matched business dates | missing generic business dates | coverage rate | supports requested window |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| `SSI` | 4097 | 2009-11-23 | 2026-04-24 | 2010-01-04 to 2025-12-31 | 3993 | 181 | 0.956636 | yes |
| `FPT` | 4097 | 2009-11-23 | 2026-04-24 | 2010-01-04 to 2025-12-31 | 3993 | 181 | 0.956636 | yes |
| `BVH` | 4097 | 2009-11-23 | 2026-04-24 | 2010-01-04 to 2025-12-31 | 3993 | 181 | 0.956636 | yes |
| `VNM` | 4097 | 2009-11-23 | 2026-04-24 | 2010-01-04 to 2025-12-31 | 3993 | 181 | 0.956636 | yes |

The missing generic business dates are calculated against a Monday-Friday denominator. They are not based on an official Vietnam exchange trading calendar. Therefore, the missing count should be interpreted as a conservative generic-business-day audit signal rather than a definitive exchange-session gap count.

## 3. Evaluation Design

The audit uses the following split:

| segment | date range |
| --- | --- |
| Training history | `2010-01-01` to `2022-12-31` |
| Forecast/evaluation window | `2023-01-01` to `2025-12-31` |

The walk-forward configuration uses `--step-sizes 1`. In this runner, step size 1 means that each available trading session in the forecast window is evaluated. It does not mean every calendar day.

The evaluated model set is:

| role | model |
| --- | --- |
| Base model | CART |
| Base model | XGBoost |
| Base model | LightGBM |
| Existing ensemble layer | `stacking_final` |

No GRU, TCN, Transformer, TFT, N-BEATS, N-HiTS, CatBoost, EGARCH, GJR-GARCH, DQN, PPO, or A2C model is added or evaluated in this report. Generated outputs are retained under ignored workflow directories and are not committed.

## 4. Horizon Setup

The evaluated horizons are:

| requested horizon | evaluated runner horizon | target length | note |
| --- | --- | ---: | --- |
| `short_5d` | `short_5d` | 5 trading days | supported exactly |
| `short_10d` | `short_10d` | 10 trading days | supported exactly |
| `short_20d` | `short_20d` | 20 trading days | supported exactly |
| `medium_60d` | `long_3m` | 63 trading days | supported proxy |

The requested `medium_60d` name is not a supported runner horizon. The existing supported proxy `long_3m` was used instead, representing approximately 63 trading days. No horizon logic was changed to force a new name.

## 5. Foreign-Flow Governance

No governed long-window foreign-flow artifact exists for `SSI`, `FPT`, `BVH`, and `VNM` across 2010-2025. The earlier 15-year daily audit used the default foreign-flow loader path, where the local default artifact was a fixture or local cache artifact rather than a governed long-window source. That run consequently reported `foreign_flow_missing_rate = 1.0` and weak coverage warnings driven by absent foreign-flow context.

The cleaner single-horizon and multi-horizon reruns use:

```powershell
--foreign-flow-mode disabled
```

Disabled mode means:

- foreign-flow artifact loading is intentionally skipped
- the default `data/foreign_flow.csv` fixture or scratch artifact is not loaded
- foreign-flow missing rates are unavailable by design
- foreign-flow features must not be interpreted
- breadth coverage remains visible and interpretable

Clean context coverage summary:

| diagnostic | result |
| --- | --- |
| Breadth missing rate | `0.0` |
| Foreign-flow mode | `disabled` |
| Foreign-flow coverage status | `disabled` |
| Weak coverage folds | `0` |
| Warning level | `ok` |

This governance control improves interpretation discipline. It does not improve or complete foreign-flow coverage.

## 6. Single-Horizon Result Summary

The clean single-horizon audit evaluates `short_5d` with foreign-flow intentionally disabled. Overall model metrics are:

| model | observations | MAE | RMSE | correlation | directional accuracy | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cart` | 2972 | 0.041548 | 0.059189 | -0.063782 | 0.496972 | 0.500501 |
| `lightgbm` | 2972 | 0.038627 | 0.051548 | -0.037671 | 0.508748 | 0.495159 |
| `stacking_final` | 2972 | 0.027669 | 0.038603 | -0.068472 | 0.491925 | 0.626053 |
| `xgboost` | 2972 | 0.041812 | 0.056113 | -0.047804 | 0.512786 | 0.482487 |

For `short_5d`, `stacking_final` has the lowest MAE, lowest RMSE, and highest F1. XGBoost has the highest directional accuracy at `0.512786`. Directional accuracy remains close to 50 percent, so this result should not be interpreted as a strong directional trading edge.

## 7. Multi-Horizon Result Summary

The multi-horizon audit evaluates `short_5d`, `short_10d`, `short_20d`, and `long_3m` using the same staged OHLCV data and disabled foreign-flow policy.

| horizon | model | observations | MAE | RMSE | correlation | directional accuracy | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `short_5d` | `cart` | 2972 | 0.041548 | 0.059189 | -0.063782 | 0.496972 | 0.500501 |
| `short_5d` | `lightgbm` | 2972 | 0.038627 | 0.051548 | -0.037671 | 0.508748 | 0.495159 |
| `short_5d` | `stacking_final` | 2972 | 0.027669 | 0.038603 | -0.068472 | 0.491925 | 0.626053 |
| `short_5d` | `xgboost` | 2972 | 0.041812 | 0.056113 | -0.047804 | 0.512786 | 0.482487 |
| `short_10d` | `cart` | 2952 | 0.061638 | 0.081398 | -0.128745 | 0.493902 | 0.519305 |
| `short_10d` | `lightgbm` | 2952 | 0.058106 | 0.074869 | -0.101849 | 0.489499 | 0.496492 |
| `short_10d` | `stacking_final` | 2952 | 0.040564 | 0.055338 | -0.033149 | 0.509824 | 0.644734 |
| `short_10d` | `xgboost` | 2952 | 0.059716 | 0.077254 | -0.047748 | 0.489160 | 0.483208 |
| `short_20d` | `cart` | 2816 | 0.085939 | 0.112797 | -0.113473 | 0.492543 | 0.491278 |
| `short_20d` | `lightgbm` | 2816 | 0.081734 | 0.108591 | -0.106519 | 0.493253 | 0.501920 |
| `short_20d` | `stacking_final` | 2816 | 0.061643 | 0.080093 | -0.035315 | 0.488991 | 0.623002 |
| `short_20d` | `xgboost` | 2816 | 0.083253 | 0.109862 | -0.092116 | 0.489347 | 0.479740 |
| `long_3m` | `cart` | 2139 | 0.154071 | 0.202510 | -0.083331 | 0.549322 | 0.594958 |
| `long_3m` | `lightgbm` | 2139 | 0.153814 | 0.203395 | -0.078701 | 0.571763 | 0.625511 |
| `long_3m` | `stacking_final` | 2139 | 0.124236 | 0.168430 | -0.186810 | 0.458626 | 0.582552 |
| `long_3m` | `xgboost` | 2139 | 0.156275 | 0.205545 | -0.079354 | 0.576438 | 0.634677 |

Prediction eligibility decreases as horizon length increases because realized targets near the end of 2025 require more future trading sessions.

| horizon | total predictions | eligible predictions | ineligible predictions |
| --- | ---: | ---: | ---: |
| `short_5d` | 11968 | 11888 | 80 |
| `short_10d` | 11968 | 11808 | 160 |
| `short_20d` | 11584 | 11264 | 320 |
| `long_3m` | 9564 | 8556 | 1008 |

## 8. Best Model by Horizon

| horizon | lowest MAE/RMSE | best directional accuracy | best F1 |
| --- | --- | --- | --- |
| `short_5d` | `stacking_final` | `xgboost` | `stacking_final` |
| `short_10d` | `stacking_final` | `stacking_final` | `stacking_final` |
| `short_20d` | `stacking_final` | `lightgbm` | `stacking_final` |
| `long_3m` | `stacking_final` | `xgboost` | `xgboost` |

The stacking layer is strongest for error minimization. Directional performance is horizon-sensitive and changes across the model set. XGBoost warrants further investigation for directional behavior, especially at `short_5d` and `long_3m`. The `short_20d` horizon is directionally weak across all evaluated models.

## 9. Ticker-Level Observations

The following table summarizes `stacking_final` by ticker and horizon:

| ticker | horizon | observations | MAE | RMSE | correlation | directional accuracy | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BVH` | `short_5d` | 743 | 0.028009 | 0.039988 | -0.066548 | 0.492598 | 0.627838 |
| `FPT` | `short_5d` | 743 | 0.027134 | 0.036798 | -0.027676 | 0.506057 | 0.618106 |
| `SSI` | `short_5d` | 743 | 0.035143 | 0.047482 | -0.088597 | 0.523553 | 0.678182 |
| `VNM` | `short_5d` | 743 | 0.020390 | 0.027443 | -0.121414 | 0.445491 | 0.572614 |
| `BVH` | `short_10d` | 738 | 0.041791 | 0.058421 | -0.063606 | 0.491870 | 0.641834 |
| `FPT` | `short_10d` | 738 | 0.038317 | 0.051918 | 0.040913 | 0.548780 | 0.663296 |
| `SSI` | `short_10d` | 738 | 0.052321 | 0.068165 | 0.008942 | 0.542005 | 0.677481 |
| `VNM` | `short_10d` | 738 | 0.029827 | 0.038654 | -0.160376 | 0.456640 | 0.594540 |
| `BVH` | `short_20d` | 704 | 0.061831 | 0.079795 | -0.082038 | 0.474432 | 0.621677 |
| `FPT` | `short_20d` | 704 | 0.059423 | 0.074327 | 0.140036 | 0.561080 | 0.686930 |
| `SSI` | `short_20d` | 704 | 0.079584 | 0.103375 | -0.049660 | 0.497159 | 0.617711 |
| `VNM` | `short_20d` | 704 | 0.045734 | 0.055511 | -0.214605 | 0.423295 | 0.561555 |
| `BVH` | `long_3m` | 539 | 0.112011 | 0.126191 | 0.007687 | 0.454545 | 0.591667 |
| `FPT` | `long_3m` | 503 | 0.137562 | 0.181366 | -0.455252 | 0.461233 | 0.577223 |
| `SSI` | `long_3m` | 541 | 0.169103 | 0.238614 | -0.295948 | 0.452865 | 0.596730 |
| `VNM` | `long_3m` | 556 | 0.080375 | 0.092419 | 0.039087 | 0.465827 | 0.562592 |

`VNM` has the lowest `short_5d` stacking MAE at `0.020390`, indicating lower point-error magnitude in that slice. `SSI` has the highest `short_5d` stacking F1 at `0.678182`. `FPT` shows relatively stronger stacking directional accuracy at `short_10d` and `short_20d`, reaching `0.548780` and `0.561080` respectively.

Longer horizons expose larger ticker-level differences. `SSI` has the highest `long_3m` stacking MAE and RMSE among the four tickers, while `VNM` remains low-error but directionally weak across several horizons. These ticker-level differences support the need for broader universe testing before drawing general conclusions.

## 10. Feature Governance and Stability

Feature governance summary:

| category | count |
| --- | ---: |
| `safe_trailing` | 12 |
| `requires_review` | 10 |
| `alias_or_redundant` | 2 |

Recommended actions:

| action | count |
| --- | ---: |
| `keep` | 12 |
| `review_timing` | 10 |
| `review_redundancy` | 2 |

Stability diagnostics:

| diagnostic | high | medium | low |
| --- | ---: | ---: | ---: |
| feature importance | 288 | 98 | 70 |
| linear coefficient | 56 | 74 | 38 |

Linear-vs-importance comparison:

| label | count |
| --- | ---: |
| `importance_only` | 95 |
| `aligned_stable` | 55 |
| `unstable_or_missing` | 2 |

The governance findings are diagnostic and do not establish causality. Breadth and market-context features remain marked for timing review where appropriate. Stability evidence can help prioritize follow-up review, but it does not imply a trading edge.

## 11. Interpretation

The combined evidence supports six main interpretations.

First, `stacking_final` consistently reduces point-forecast error across the evaluated horizons. It has the lowest MAE and RMSE for `short_5d`, `short_10d`, `short_20d`, and `long_3m`.

Second, directional prediction remains fragile. The best directional-accuracy values are modest and vary by horizon. A model that reduces point error is not automatically the best directional classifier.

Third, XGBoost shows relatively stronger directional behavior in `short_5d` and `long_3m`, where it leads directional accuracy. It also leads F1 in `long_3m`. This makes XGBoost a reasonable candidate for further diagnostic slicing, but not evidence of trading superiority.

Fourth, longer horizons increase forecast error and reduce eligible observations. This is expected because longer realized targets require more future trading sessions and aggregate more market uncertainty.

Fifth, multi-horizon evaluation is necessary. A single `short_5d` audit captures only one target length and would miss the ranking changes observed at `short_10d`, `short_20d`, and `long_3m`.

Sixth, foreign-flow should remain excluded from long-window interpretation until a governed artifact exists. Disabled mode improves audit clarity by preventing fixture or scratch-cache artifacts from producing misleading weak-coverage warnings.

Signal-effectiveness backtesting is the next bridge from forecast evaluation to investment-style decision support. It should test whether strict BUY rules have useful precision after explicit cost and slippage assumptions, while preserving the interpretation that these are diagnostic outputs rather than trading-performance proof.

Held-out threshold selection is required before treating BUY precision targets as policy candidates. Descriptive threshold frontiers can identify promising slices, but thresholds must be selected on an earlier period and evaluated unchanged on a later held-out period before stronger interpretation.

Rolling held-out threshold selection is required before treating a BUY precision target as stable. Regime-conditioned evaluation is also needed before deciding whether BUY rules should be active in all market states or only in favorable regimes. Current rolling diagnostics remain analysis-only and do not establish trading-performance proof.

## 12. Limitations

This report has the following limitations:

- only four tickers are evaluated
- the selected ticker basket is not a full-market sample
- only CART, XGBoost, LightGBM, and the existing stacking layer are evaluated
- no GRU, TCN, Transformer, TFT, N-BEATS, N-HiTS, CatBoost, EGARCH, GJR-GARCH, DQN, PPO, or A2C model is included
- foreign-flow context is disabled and must not be interpreted
- the coverage denominator uses generic business days, not an official Vietnam exchange calendar
- generated outputs are diagnostic artifacts, not trading strategy proof
- transaction costs, slippage, liquidity, borrow constraints, turnover, and portfolio construction constraints are not proven here
- correlations are weak or negative in several settings
- directional accuracy and one-period BUY precision are not strong enough by themselves to support trading claims

## 13. Recommended Next Steps

Recommended follow-up work, in priority order:

1. Rolling held-out threshold selection for saved forecast outputs before promoting any BUY precision target as stable.
2. Regime-aware performance slicing: bull, bear, sideways, and high-volatility regimes where safe labels exist.
3. Broader VN100-style audit if runtime resources and long-window coverage allow.
4. Long-window foreign-flow artifact curation if foreign-flow interpretation is required.
5. Optional technical report export to Word or PDF for advisor review.

## 14. Conclusion

VSEF now has a governed 15-year daily and multi-horizon evaluation baseline for `SSI`, `FPT`, `BVH`, and `VNM`. The audit sequence establishes reproducible staged OHLCV loading, explicit daily walk-forward evaluation, disabled foreign-flow governance for long-window runs, and multi-horizon comparison.

The strongest conclusion is that `stacking_final` is most reliable on error metrics across the evaluated horizons. Directional performance is mixed, horizon-dependent, and not strong enough to justify trading-performance claims. The framework is suitable for further research and audit-driven development, not live deployment.

Controls such as explicit OHLCV directories and foreign-flow disabled mode improve reproducibility and interpretation discipline. Future work should prioritize regime-aware analysis, strategy-level validation with realistic frictions, broader coverage only where long-window data supports it, and long-window foreign-flow curation only if that context is required for governed interpretation.
