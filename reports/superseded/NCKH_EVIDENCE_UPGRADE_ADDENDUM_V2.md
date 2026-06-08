# NCKH Evidence Upgrade Addendum V2

## Purpose of Addendum

This addendum documents the evidence-upgrade work completed after the first submission package. It does not rewrite the full paper and does not introduce new model families, production/runtime changes, or heavy benchmark reruns. Its purpose is to show how the evidence-gap-closure artifacts affect the paper's Chapter 4 discussion, limitations, and future-work framing.

The addendum should be treated as a paper update guide and supervisor review artifact. It should not replace the existing manuscript until the supervisor accepts derived diagnostics as appropriate evidence for the final paper.

## Summary of Evidence-Gap-Closure Phase

The evidence-gap-closure phase used official 2025 artifacts under `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff` and generated paper-readable outputs under `reports/generated/evidence_gap_closure/`.

Main outcomes:

- Cache audit still leaves only 7 benchmark-usable 2025 tickers: ANV, BCM, BID, BMP, BVH, BWE, and CII.
- Derived v2 confidence sweep covers 32 available frequency/model/horizon combinations from 154,048 prediction rows.
- Best >=50% coverage candidate: daily XGBoost h=20, confidence threshold 0.86, 60.55% accuracy, 53.02% coverage.
- Lagged ex-ante bear-regime daily LightGBM h=20 reaches 66.34% over 309 rows.
- Lagged ex-ante bear-regime daily XGBoost h=20 reaches 65.05% over 309 rows.
- Cost/slippage proxy artifacts now exist, but they do not establish executable trading readiness.
- Multi-window validation remains unavailable because 2022-2024 official prediction artifacts were not generated.

## Gap Closure Status Table

| Gap | Status after upgrade | Evidence source | Interpretation |
|---|---|---|---|
| Seven evaluated tickers | Not closed | `reports/generated/evidence_gap_closure/vn100_cache_coverage_audit.md` | The official 2025 evidence remains limited to seven benchmark-usable tickers. |
| Missing daily confidence-threshold rows | Partially closed | `reports/generated/evidence_gap_closure/vn100_full_confidence_sweep_summary.csv` | Daily rows now exist as derived diagnostics from official prediction rows. |
| Sweep limited to hourly stacking h=1 | Partially closed | `reports/generated/evidence_gap_closure/vn100_full_confidence_sweep_report.md` | The derived v2 sweep covers all available frequency/model/horizon combinations. |
| Selected confidence slice concentration | Partially closed | `reports/generated/evidence_gap_closure/vn100_full_confidence_sweep_report.md`; `reports/generated/vn100_ticker_concentration_summary.md` | The original hourly selected slice remains concentrated; newer daily candidates show lower concentration at higher coverage floors. |
| No cost/slippage artifacts | Partially closed | `reports/generated/evidence_gap_closure/vn100_cost_slippage_validation_report.md` | Cost/slippage proxy diagnostics exist, but they are not execution-ready trading artifacts. |
| Single 2025 evaluation window | Not closed | `reports/generated/evidence_gap_closure/vn100_multiwindow_validation_report.md` | 2022-2024 official prediction artifacts remain unavailable. |
| No ex-ante regime validation | Partially closed | `reports/generated/evidence_gap_closure/vn100_exante_regime_validation_report.md` | A lagged ex-ante proxy supports the 2025 daily bear-regime diagnostic, but not multi-window stability. |

## New Derived Confidence-Sweep Findings

The original submission package treated the confidence-threshold evidence as partial because the official sweep artifact contained rows only for hourly stacking h=1. The v2 derived sweep broadens this diagnostic by recomputing thresholds from existing official prediction rows.

Key findings:

- The v2 sweep uses 154,048 official prediction rows.
- It covers 32 available frequency/model/horizon combinations.
- It sweeps confidence thresholds from 0.50 to 0.90.
- The best >=50% coverage candidate is daily XGBoost h=20 at threshold 0.86, with 60.55% accuracy and 53.02% coverage.
- The best >=40% coverage candidate is daily XGBoost h=20 at threshold 0.90, with 60.78% accuracy and 44.85% coverage.
- The best >=30% coverage candidate is daily stacking h=20 at threshold 0.69, with 62.03% accuracy and 30.28% coverage.
- The best >=20% coverage candidate is daily stacking h=20 at threshold 0.71, with 64.53% accuracy and 21.61% coverage.

Interpretation: these results strengthen the conditional confidence-filtering discussion. They do not convert the official global benchmark into a pass because they are selected, filtered, derived diagnostics from one 2025 evaluation window.

## New Ex-Ante Regime Proxy Findings

The original manuscript treated bear-regime performance as post-hoc diagnostic evidence. The evidence-upgrade phase adds a lagged ex-ante proxy using only prior realized target returns within each ticker/frequency/model/horizon sequence.

Key findings:

- Daily LightGBM h=20 in lagged ex-ante bear regime: 66.34% accuracy over 309 rows.
- Daily XGBoost h=20 in lagged ex-ante bear regime: 65.05% accuracy over 309 rows.
- Both rows pass 63% in the 2025 official prediction artifact.
- Each key slice covers seven tickers, with four reliable tickers passing 63%.

Interpretation: this reduces the concern that the daily h=20 bear-regime result is purely post-hoc. It does not establish a stable full-market 63% method because the ex-ante proxy has not been validated across 2022-2024 windows or a broader benchmark-usable ticker set.

## New Cost/Slippage Proxy Findings

The original submission package correctly stated that cost/slippage artifacts were missing. The evidence-upgrade phase adds a diagnostic proxy for selected slices using existing official target returns, a long/flat signal mapping, and cost/slippage grids.

Key findings:

- The proxy includes transaction cost bps 5, 10, 15, and 20 crossed with slippage bps 5, 10, 15, and 20.
- It includes model-signal, buy-and-hold, flat/no-trade, always-up, moving-average, and previous-direction baselines.
- It reports gross return, net return, turnover, drawdown, profit factor, win rate, trade count, average trade return, exposure, equity curve, cost-adjusted return, and benchmark comparison.
- The hourly stacking h=1 threshold 0.57 slice is positive at the 10/10 bps diagnostic grid but negative at the 20/20 bps grid.

Interpretation: the cost/slippage gap is partially closed as a diagnostic artifact category. It still does not establish executable trading readiness because the proxy lacks entry/exit execution prices, liquidity filters, fill assumptions, deployment constraints, and multi-window validation.

## Gaps Still Not Closed

- Full VN100 representativeness remains unproven because only seven tickers are benchmark-usable for 2025.
- Expanded official 2025 benchmark artifacts do not exist.
- Multi-window validation remains unavailable for 2022, 2023, and 2024.
- Confidence-sweep findings remain derived diagnostics from existing prediction rows, not a new official benchmark run.
- Cost/slippage evidence remains a proxy, not a deployable execution backtest.
- The ex-ante regime proxy remains single-window and has not been tested out of sample across years.

## Claim Boundary After Upgrade

Safe after upgrade:

- The global daily and hourly official benchmark still did not pass 60%.
- The evaluated official universe still contains seven benchmark-usable tickers.
- Derived v2 confidence diagnostics identify daily h=20 candidates above 60% at meaningful coverage floors.
- Lagged ex-ante bear-regime diagnostics support the 2025 daily h=20 regime finding.
- Cost/slippage proxy diagnostics exist but do not establish trading readiness.

Still unsafe:

- Claiming a global benchmark pass.
- Claiming representativeness for the full VN100 universe.
- Claiming a stable multi-window 63% method.
- Claiming live trading readiness or profitability.

## How This Changes Chapter 4 Discussion

Chapter 4 should add two concise subsections:

1. Derived confidence-sweep expansion: report the v2 sweep coverage, the 32 frequency/model/horizon combinations, and the best coverage-floor candidates. State clearly that these are derived diagnostics from official predictions.
2. Lagged ex-ante regime validation: report the daily LightGBM and XGBoost h=20 ex-ante bear-regime rows. Explain that this strengthens the regime discussion but does not prove multi-window stability.

The cost/slippage proxy should be added as an appendix or limitations subsection, not as evidence of trading readiness.

## How This Changes Future-Work Section

The future-work section should become more specific:

- Expand cache coverage and rerun the official 2025 benchmark only after more tickers become benchmark-usable.
- Produce official 2022-2024 walk-forward artifacts to test signal stability.
- Convert cost/slippage proxy diagnostics into execution-aware backtests with explicit entry/exit prices, liquidity filters, and fill assumptions.
- Validate ex-ante regime rules across multiple windows before treating regime slices as deployable filters.
- Decide with supervisor review whether derived v2 diagnostics should enter the abstract or remain in Chapter 4.
