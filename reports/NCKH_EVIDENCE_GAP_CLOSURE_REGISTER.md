# NCKH Evidence Gap Closure Register

## Source Phase

This register summarizes the VN100 evidence-gap-closure phase on branch `research/vn100-evidence-hardening-v1`.
The phase used existing official artifacts under `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff`.
No new model family was added, no production/runtime behavior was changed, and no heavy benchmark rerun was performed.

## Gap Closure Table

| Original gap | Work performed | Output artifact | Status | Evidence summary | Remaining limitation | Paper claim boundary changes |
|---|---|---|---|---|---|---|
| Seven evaluated tickers | Audited official cache summary and local cache file presence for all considered VN100 tickers. | `reports/generated/evidence_gap_closure/vn100_cache_coverage_audit.md`; `reports/generated/evidence_gap_closure/vn100_cache_coverage_audit.csv` | Not closed | 104 tickers were considered; 60 have local daily cache files and 86 have local hourly cache files, but only 7 are benchmark-usable for 2025. | No additional benchmark-usable 2025 tickers were found without cache expansion or a rerun. | No. The paper must still state that the official evaluated set is seven tickers and is not representative of the full VN100 universe. |
| Missing daily confidence sweep rows | Recomputed deterministic threshold diagnostics from official prediction rows for every available daily and hourly model/horizon combination. | `reports/generated/evidence_gap_closure/vn100_full_confidence_sweep_summary.csv`; `reports/generated/evidence_gap_closure/vn100_full_confidence_sweep_report.md` | Partially closed | Derived v2 sweep includes daily rows. Best >=50% coverage candidate is daily XGBoost h=20 threshold 0.86 with 60.55% accuracy and 53.02% coverage. | The official runner's daily threshold-sweep file remains header-only; v2 rows are derived analysis, not a new official benchmark run. | Conditional claims can mention derived v2 confidence diagnostics if clearly labeled. Global benchmark claims do not change. |
| Confidence sweep only hourly stacking h=1 | Swept all available frequencies, models, horizons, and thresholds 0.50-0.90 from official prediction rows. | `reports/generated/evidence_gap_closure/vn100_full_confidence_sweep_summary.csv`; `reports/generated/evidence_gap_closure/vn100_confidence_threshold_coverage_accuracy_v2.png` | Partially closed | The v2 derived sweep covers 32 available frequency/model/horizon combinations from 154,048 prediction rows. Several daily h=20 rows pass 60% at coverage floors from 20% to 50%. | Derived from existing predictions only; no expanded-cache or multi-window official rerun exists. | Conditional confidence-sweep discussion becomes broader, but the global benchmark still does not pass. |
| Selected confidence slice concentration | Added concentration checks for best v2 coverage-floor candidates. | `reports/generated/evidence_gap_closure/vn100_full_confidence_sweep_report.md`; previous `reports/generated/vn100_ticker_concentration_summary.md` | Partially closed | New best >=50% and >=40% daily XGBoost h=20 candidates have low prediction-count concentration. Best >=30% daily stacking candidate is moderate; best >=20% is high. | Original hourly stacking h=1 threshold 0.57 selected slice remains high-concentration and the evaluated universe remains seven tickers. | Paper can distinguish the original concentrated hourly slice from broader v2 daily candidates. No full-market representativeness claim is allowed. |
| No cost/slippage trading artifacts | Generated cost/slippage-aware diagnostic proxy for selected slices and baselines across a 5/10/15/20 bps cost and slippage grid. | `reports/generated/evidence_gap_closure/vn100_cost_slippage_validation_report.md`; `reports/generated/evidence_gap_closure/vn100_cost_slippage_summary.csv`; `reports/generated/evidence_gap_closure/vn100_trade_list.csv`; `reports/generated/evidence_gap_closure/vn100_equity_curve.csv`; `reports/generated/evidence_gap_closure/vn100_equity_curve.png` | Partially closed | Cost/slippage, turnover, drawdown, profit factor, win rate, trade count, exposure, equity-curve, and baseline-comparison diagnostics now exist. Hourly stacking h=1 threshold 0.57 is positive at 10/10 bps but negative at 20/20 bps in the diagnostic proxy. | Uses target-return proxies rather than executable entry/exit prices, liquidity filters, fills, or deployment constraints. | Practical trading readiness remains not established. |
| Single 2025 evaluation window | Created a multi-window availability script and report for requested 2022-2025 windows. | `reports/generated/evidence_gap_closure/vn100_multiwindow_validation_report.md`; `reports/generated/evidence_gap_closure/vn100_multiwindow_accuracy_summary.csv`; `reports/generated/evidence_gap_closure/vn100_multiwindow_stability_matrix.csv` | Not closed | 2025 official artifacts are available; 2022, 2023, and 2024 official prediction artifacts are unavailable. | Heavy benchmark reruns were not performed, so stability across windows is not established. | No. The paper must continue to describe the evidence as single-window 2025 evidence. |
| No ex-ante regime validation | Derived lagged ex-ante regime labels using prior realized target returns only and compared them with post-hoc regime diagnostics. | `reports/generated/evidence_gap_closure/vn100_exante_regime_validation_report.md`; `reports/generated/evidence_gap_closure/vn100_exante_regime_accuracy_summary.csv`; `reports/generated/evidence_gap_closure/vn100_regime_posthoc_vs_exante_comparison.csv` | Partially closed | Daily LightGBM h=20 ex-ante bear accuracy is 66.34% over 309 rows; daily XGBoost h=20 ex-ante bear accuracy is 65.05% over 309 rows. Both are reliable in the 2025 artifact and pass 63%. | The ex-ante rule is a lagged diagnostic proxy and has not been validated across 2022-2024 windows or a larger ticker universe. | Regime evidence becomes stronger for 2025 conditional diagnostics, but not for global, stable, or trading-readiness claims. |

## Overall Claim Boundary

The evidence-upgrade phase strengthens conditional diagnostics but does not change the core manuscript boundary:

- Global benchmark pass: no.
- Full-market VN100 representativeness: no.
- Stable multi-window 63% method: no.
- Practical trading readiness: not established.
- Conditional signal evidence: stronger, especially derived daily confidence-sweep diagnostics and lagged ex-ante daily bear-regime diagnostics.

## New Output Directories

The following ignored output directories were created locally as requested:

- `outputs/vn100_multiwindow_walkforward/`
- `outputs/vn100_cost_slippage_validation/`

Committed paper-readable outputs are under `reports/generated/evidence_gap_closure/`.
