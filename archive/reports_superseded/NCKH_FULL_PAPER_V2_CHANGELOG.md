# NCKH Full Paper V2 Changelog

## What Changed From Clean V1

- Created a full V2 manuscript at `reports/NCKH_FULL_PAPER_DRAFT_VN100_V2_WITH_FIGURES.md`.
- Rewrote the abstract to include evidence-upgrade findings while preserving cautious claim boundaries.
- Updated Chapter 3 with cache-audit counts and the V2 explanation of limited benchmark-usable coverage.
- Rewrote Chapter 4 to include original official benchmark results, original confidence diagnostics, V2 confidence-sweep expansion, original regime diagnostics, lagged ex-ante regime validation, statistical significance, ticker concentration, cost/slippage proxy diagnostics, and evidence-gap closure.
- Rewrote Chapter 5 to distinguish global benchmark failure from stronger conditional diagnostics.
- Added References and Appendices A-C.
- Embedded key PNG figures and copied Markdown schematic figures as text.

## New V2 Evidence Added

- Cache audit confirms 104 considered tickers, 60 local daily cache files, 86 local hourly cache files, and only 7 benchmark-usable 2025 tickers.
- Derived V2 confidence sweep covers 32 available frequency/model/horizon combinations from 154,048 prediction rows.
- Best >=50% coverage derived confidence candidate is daily XGBoost h=20 threshold 0.86 with 60.55% accuracy and 53.02% coverage.
- Lagged ex-ante bear-regime daily LightGBM h=20 reaches 66.34% over 309 rows.
- Lagged ex-ante bear-regime daily XGBoost h=20 reaches 65.05% over 309 rows.
- Cost/slippage proxy diagnostics now include cost/slippage grid, baselines, turnover, drawdown, profit factor, win rate, trade count, exposure, and equity curve.

## Claims Strengthened

- Conditional confidence evidence is stronger after the derived V2 sweep.
- The 2025 daily bear-regime h=20 finding is stronger after lagged ex-ante proxy validation.
- Cost/slippage discussion is stronger because proxy artifacts now exist.
- Concentration discussion is stronger because V2 coverage-floor candidates can be contrasted with the original concentrated hourly selected slice.

## Claims Unchanged

- The official global benchmark does not pass 60%.
- The official evaluated universe contains seven benchmark-usable tickers.
- Full VN100 representativeness is not established.
- Stable multi-window performance is not established.
- Trading readiness is not established.
- Profitability is not established.

## Claims Still Unsafe

- Global 60% benchmark success.
- Full VN100 representativeness.
- Stable multi-window 63% full-market performance.
- Live-trading readiness.
- Profitability after realistic execution costs.
- New-model or runtime/model-improvement claims for V2.

## Remaining Evidence Gaps

- Seven-ticker benchmark-usable coverage remains unresolved.
- Expanded official 2025 benchmark artifacts do not exist.
- Official 2022-2024 walk-forward artifacts are unavailable.
- Derived V2 confidence sweep is not a new official benchmark rerun.
- Ex-ante regime validation remains single-window.
- Cost/slippage diagnostics remain proxy-based and are not execution-aware backtests.

## Files Added in This Phase

- `reports/NCKH_FULL_PAPER_DRAFT_VN100_V2_WITH_FIGURES.md`
- `reports/NCKH_FULL_PAPER_V2_TABLE_FIGURE_MAP.md`
- `reports/NCKH_FULL_PAPER_V2_CHANGELOG.md`
