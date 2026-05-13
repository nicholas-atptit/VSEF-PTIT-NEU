# NCKH Post Gap Closure Paper Update Notes

## Evidence Upgrade Summary

The evidence-gap-closure phase adds derived diagnostics from official 2025 prediction artifacts. It does not add new model families, does not change runtime/model behavior, and does not rerun the heavy benchmark pipeline.

Main evidence changes:

- Cache audit confirms the seven-ticker limitation remains.
- Derived v2 confidence sweep now covers all available daily/hourly model/horizon combinations.
- Daily confidence-filtered h=20 candidates appear stronger than the original hourly-only sweep, but they remain derived single-window diagnostics.
- Lagged ex-ante daily bear-regime diagnostics remain above 63% for LightGBM and XGBoost h=20 in 2025.
- Cost/slippage proxy artifacts now exist, but executable trading readiness is still not established.
- Multi-window validation remains unavailable for 2022-2024.

## Tables Needing Updates

| Table | Update needed | Suggested source |
|---|---|---|
| Table 1: Dataset and evaluation scope | Add cache-audit counts: 104 considered tickers, 60 daily cache files, 86 hourly cache files, 7 benchmark-usable tickers. | `reports/generated/evidence_gap_closure/vn100_cache_coverage_audit.md` |
| Table 3: Global benchmark results | No numeric change unless an expanded official benchmark is later run. Keep global pass as no. | Existing `daily/hourly/benchmark_summary.json` |
| Table 5: Confidence-filtered strategy diagnostics | Add derived v2 rows for daily XGBoost h=20 and daily stacking h=20 coverage-floor candidates. Mark status as derived/partial. | `reports/generated/evidence_gap_closure/vn100_full_confidence_sweep_summary.csv` |
| Table 6: Regime-specific diagnostics | Add lagged ex-ante bear-regime rows for daily LightGBM h=20 and daily XGBoost h=20. | `reports/generated/evidence_gap_closure/vn100_exante_regime_accuracy_summary.csv` |
| Table 8: Robustness and limitation matrix | Update cost/slippage from missing to diagnostic proxy available; keep trading readiness not established. Add multi-window still missing. | `reports/NCKH_EVIDENCE_GAP_CLOSURE_REGISTER.md` |

## Figures Needing Updates

| Figure | Update needed | Suggested source |
|---|---|---|
| Figure 4: Confidence threshold vs coverage/accuracy | Replace or append the v2 confidence sweep figure. Label it as derived from official predictions. | `reports/generated/evidence_gap_closure/vn100_confidence_threshold_coverage_accuracy_v2.png` |
| Figure 5: Regime-specific accuracy | Consider adding ex-ante bear-regime markers or a separate panel. | `reports/generated/evidence_gap_closure/vn100_exante_regime_accuracy_summary.csv` |
| New optional figure: Cost/slippage equity proxy | Use only as diagnostic appendix material, not as a trading-readiness figure. | `reports/generated/evidence_gap_closure/vn100_equity_curve.png` |

## Claims That Become Stronger

- Confidence filtering is no longer limited to the original hourly stacking h=1 artifact at the derived-analysis level. The v2 sweep covers all 32 available frequency/model/horizon combinations.
- Daily XGBoost h=20 has derived confidence-filtered candidates above 60% at >=50% and >=40% coverage floors.
- Daily stacking h=20 has stronger but more selective derived confidence candidates at >=30% and >=20% coverage floors.
- The bear-regime daily h=20 finding is less purely post-hoc because a lagged ex-ante proxy also exceeds 63% for LightGBM and XGBoost in 2025.
- Cost/slippage diagnostics can now be discussed as a preliminary proxy rather than a completely missing artifact category.

## Claims That Remain Unsafe

- The VN100 benchmark passed globally.
- The evidence represents the full VN100 universe.
- The method is stable across 2022-2025 windows.
- The system is ready for live trading.
- The study proves profitability after realistic costs.
- The study establishes a stable full-market 63% method.

## Abstract and Conclusion Guidance

The abstract and conclusion should not be rewritten automatically yet. If updated, they should:

- Keep the global benchmark pass as no.
- Keep the seven-ticker limitation explicit.
- Add that derived v2 confidence diagnostics identify daily h=20 candidates above 60% at meaningful coverage floors.
- Add that lagged ex-ante bear-regime diagnostics support the 2025 conditional regime finding.
- State that cost/slippage proxy diagnostics are preliminary and do not establish trading readiness.
- State that multi-window validation remains unavailable.

## Submission Package V1 Status

Submission package v1 should be superseded by an evidence-upgrade addendum if the supervisor accepts derived diagnostics as paper evidence. It should not be replaced by a fully rewritten paper until the claim register and tables are updated consistently.

## Recommended Next Paper Action

Create a v2 paper addendum first:

1. Update Tables 1, 5, 6, and 8.
2. Replace Figure 4 with the v2 sweep figure.
3. Add a short Chapter 4 subsection on derived confidence-sweep expansion.
4. Add a short Chapter 4 subsection on lagged ex-ante regime validation.
5. Keep cost/slippage results in limitations or appendix unless execution-quality assumptions are added.
