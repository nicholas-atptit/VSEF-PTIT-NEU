# NCKH Full Paper V2 Update Plan

## Purpose

This plan gives exact update instructions for a later V2 manuscript revision. It does not rewrite `reports/NCKH_FULL_PAPER_DRAFT_VN100_CLEAN.md` yet.

## Tables to Update

| Target | Required update | Source artifact | Status wording |
|---|---|---|---|
| Table 1: Dataset and evaluation scope | Add cache audit counts: 104 considered tickers, 60 local daily cache files, 86 local hourly cache files, and 7 benchmark-usable 2025 tickers. | `reports/generated/evidence_gap_closure/vn100_cache_coverage_audit.md` | Ready as audit evidence; coverage gap remains open. |
| Table 5: Confidence-filtered strategy diagnostics | Add derived v2 confidence-sweep rows for daily XGBoost h=20 thresholds 0.86 and 0.90, and daily stacking h=20 thresholds 0.69 and 0.71. | `reports/generated/evidence_gap_closure/vn100_full_confidence_sweep_summary.csv` | Partial/derived; not a new official benchmark run. |
| Table 6: Regime-specific diagnostics | Add lagged ex-ante bear-regime rows: daily LightGBM h=20 at 66.34% over 309 rows and daily XGBoost h=20 at 65.05% over 309 rows. | `reports/generated/evidence_gap_closure/vn100_exante_regime_accuracy_summary.csv` | Partial; 2025-only ex-ante proxy. |
| Table 8: Robustness and limitation matrix | Update gap status: confidence sweep partially closed, ex-ante regime partially closed, cost/slippage proxy partially closed, cache expansion not closed, multi-window not closed. | `reports/NCKH_EVIDENCE_GAP_CLOSURE_REGISTER.md` | Ready as limitation matrix update. |

## Figures to Update

| Target | Required update | Source artifact | Caution |
|---|---|---|---|
| Figure 4: Confidence threshold vs coverage/accuracy | Replace or append the original figure with the v2 sweep figure. | `reports/generated/evidence_gap_closure/vn100_confidence_threshold_coverage_accuracy_v2.png` | Label as derived from official prediction rows. |
| Figure 5: Regime-specific accuracy | Optionally append lagged ex-ante bear-regime markers or add a small supporting table instead of changing the figure. | `reports/generated/evidence_gap_closure/vn100_exante_regime_accuracy_summary.csv` | Do not imply multi-window stability. |
| Appendix figure: Cost/slippage equity proxy | Add only as appendix material if space allows. | `reports/generated/evidence_gap_closure/vn100_equity_curve.png` | Do not present as executable trading backtest. |

## Chapter 4 Additions

### Add Subsection: Derived Confidence-Sweep Expansion

Place after the current confidence-filtered diagnostics subsection.

Required content:

- State that the v2 sweep is derived from existing official prediction rows.
- Report that it covers 32 available frequency/model/horizon combinations from 154,048 prediction rows.
- Report the best >=50% coverage candidate: daily XGBoost h=20 threshold 0.86, 60.55% accuracy, 53.02% coverage.
- Mention the other coverage-floor candidates only as supporting diagnostics.
- State that this strengthens conditional signal evidence but does not change the global benchmark result.

Allowed wording:

> A derived v2 confidence sweep over official prediction rows broadens the confidence-filtering evidence beyond the original hourly stacking h=1 artifact.

Forbidden wording:

> The official benchmark now passes 60%.

### Add Subsection: Lagged Ex-Ante Regime Validation

Place after the current regime-specific diagnostics subsection.

Required content:

- Define the ex-ante proxy as a lagged rule based on prior realized target returns only.
- Report daily LightGBM h=20 bear-regime accuracy of 66.34% over 309 rows.
- Report daily XGBoost h=20 bear-regime accuracy of 65.05% over 309 rows.
- State that this reduces post-hoc regime-selection concern for the 2025 diagnostic.
- State that it does not establish a stable full-market 63% method.

Allowed wording:

> The lagged ex-ante proxy supports the 2025 daily bear-regime diagnostic for h=20.

Forbidden wording:

> The model has a stable 63% full-market method.

## Appendix Addition

### Add Appendix Subsection: Cost/Slippage Proxy Diagnostics

Required content:

- State that cost/slippage proxy diagnostics now exist.
- List cost grid: transaction cost bps 5, 10, 15, 20 and slippage bps 5, 10, 15, 20.
- List baselines: buy-and-hold, flat/no-trade, always-up, moving-average signal, previous-direction signal.
- State that the proxy uses target-return-derived returns, not execution-ready entry/exit prices.
- State that practical trading readiness remains not established.

Recommended placement: appendix after artifact map, or Chapter 4 limitations if no appendix space is available.

## Abstract and Conclusion

Keep abstract and conclusion cautious unless the supervisor explicitly accepts derived diagnostics for headline wording.

Safe possible addition:

> Additional derived diagnostics broaden the confidence-sweep evidence and support a lagged ex-ante bear-regime result in the 2025 window, while coverage and multi-window limitations remain.

Do not add:

- Any global 60% benchmark pass claim.
- Any full-market VN100 representativeness claim.
- Any stable multi-window 63% claim.
- Any trading-readiness or profitability claim.

## Update Sequence

1. Update Table 1 with cache audit counts.
2. Update Table 5 with derived v2 confidence sweep rows.
3. Update Table 6 with lagged ex-ante bear-regime rows.
4. Update Table 8 with gap closure status.
5. Replace or append Figure 4 with the v2 sweep figure.
6. Add Chapter 4 subsection: Derived confidence-sweep expansion.
7. Add Chapter 4 subsection: Lagged ex-ante regime validation.
8. Add appendix subsection: Cost/slippage proxy diagnostics.
9. Recheck all abstract and conclusion wording against `reports/NCKH_RESULTS_CLAIM_REGISTER_V2.md`.
