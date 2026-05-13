# NCKH Supervisor Brief After Gap Closure

## One-Page Summary

The evidence-gap-closure phase strengthens the VN100 paper, but it does not overturn the original claim boundaries. The strongest new material is a derived confidence-threshold sweep across all available prediction-row combinations and a lagged ex-ante regime proxy for the daily bear-regime h=20 finding.

The paper can be updated before submission if the supervisor accepts derived diagnostics as supporting Chapter 4 evidence. The full paper should not be rewritten automatically yet; the safer next step is to add a V2 addendum, update selected tables/figures, and preserve the original limitations.

## What Improved

- Confidence-sweep evidence is broader. The derived v2 sweep covers 32 available frequency/model/horizon combinations from 154,048 official prediction rows.
- Daily confidence-filtered diagnostics now have usable rows. The best >=50% coverage candidate is daily XGBoost h=20 at threshold 0.86, with 60.55% accuracy and 53.02% coverage.
- Regime evidence is stronger. A lagged ex-ante bear-regime proxy reaches 66.34% for daily LightGBM h=20 and 65.05% for daily XGBoost h=20, both over 309 rows.
- Cost/slippage evidence moved from missing to preliminary proxy diagnostics with cost grid, slippage grid, baselines, turnover, drawdown, profit factor, trade count, exposure, and equity curve.
- The evidence package now has a clear closure register and paper update plan.

## What Did Not Improve

- Cache coverage did not expand. The cache audit still leaves only seven benchmark-usable 2025 tickers.
- No expanded official 2025 benchmark was produced.
- Multi-window validation remains unavailable for 2022, 2023, and 2024.
- The global official benchmark still does not pass 60%.
- The evidence still does not represent the full VN100 universe.
- Cost/slippage evidence remains a proxy rather than an executable trading backtest.
- The ex-ante regime finding is still single-window evidence.

## Should the Paper Be Updated Before Submission?

Recommended answer: yes, but narrowly.

The paper should be updated if the supervisor accepts derived diagnostics as Chapter 4 supporting evidence. The update should not change the main conclusion. It should add:

- Table 1 cache-audit counts.
- Table 5 derived v2 confidence-sweep rows.
- Table 6 lagged ex-ante bear-regime rows.
- Table 8 gap-closure status.
- Figure 4 v2 confidence sweep figure.
- Chapter 4 subsection on derived confidence-sweep expansion.
- Chapter 4 subsection on lagged ex-ante regime validation.
- Appendix subsection on cost/slippage proxy diagnostics.

The abstract and conclusion should remain cautious unless the supervisor wants the derived diagnostics mentioned explicitly.

## Recommended Cautious Wording

Use:

> The official 2025 global benchmark does not pass the 60% threshold, but derived post-benchmark diagnostics broaden the confidence-filter evidence and support a lagged ex-ante daily bear-regime result.

Use:

> The evidence upgrade strengthens conditional diagnostic findings while leaving full-market representativeness, multi-window stability, and trading readiness unresolved.

Use:

> The cost/slippage results are preliminary proxy diagnostics and should not be interpreted as executable trading-readiness evidence.

Do not use:

> The VN100 benchmark passes 60%.

Do not use:

> The model achieves a stable full-market 63% accuracy.

Do not use:

> The strategy is ready for live trading or profitable after costs.

## Supervisor Decision Needed

The main decision is whether derived v2 diagnostics should appear in the abstract and conclusion or remain in Chapter 4 and appendix. The safest academic positioning is to keep them in Chapter 4 as evidence upgrades and keep the abstract focused on the official benchmark, conditional diagnostics, and limitations.
