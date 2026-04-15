# Changelog Summary

This document summarizes the major ML and evaluation changes that turned the repository from a basic forecasting project into a layered research framework for Vietnamese equities.

The emphasis here is not only what changed, but why each change mattered to the research process.

## 1. Real-Data `vnstock` Integration

What changed:

- Added `src/data/adapters/vnstock_adapter.py` as the main real-data adapter.
- Standardized fetched data into a consistent daily schema:
  - `date`
  - `ticker`
  - `open`
  - `high`
  - `low`
  - `close`
  - `volume`
- Added sorting, de-duplication, type coercion, date filtering, and retry handling for transient fetch failures.

Why it mattered:

- Removed dependence on manual CSV downloads and mock data.
- Made the evaluation stack reproducible from code.
- Established a single source of truth for both ticker OHLCV and benchmark/index data used later by regime analysis.

## 2. Fixed-Window Real-Data Backtest

What changed:

- Added `src/ml/backtest/real_data.py` and `scripts/run_backtest_real_data.py`.
- Implemented a fixed train/eval split using real daily data.
- Added artifact writing for:
  - `predicted_vs_actual.csv`
  - `metrics_summary.csv`
  - `fetch_summary.csv`
  - `training_summary.csv`
  - `run_config.json`
  - per-ticker charts
- Added a naive previous-close baseline to the close-level evaluation.

Why it mattered:

- Gave the repo a deterministic baseline workflow.
- Made leakage checks and data-coverage reporting explicit.
- Exposed a hard truth early: naive persistence is strong and must be beaten honestly.

## 3. Multi-Model Comparison

What changed:

- Added `src/ml/backtest/model_comparison.py` and `scripts/run_backtest_model_comparison.py`.
- Reused the fixed-window backtest path to compare `cart`, `xgboost`, `lightgbm`, and optional `sarimax` / `ets`.
- Added `model_comparison.csv` and `overall_model_ranking.csv`.

Why it mattered:

- Shifted the repo from single-model experimentation to repeatable model-family comparison.
- Made optional dependency handling explicit by skipping unavailable algorithms instead of breaking the run.
- Showed that family-level behavior is often more stable than exact configuration-level behavior.

## 4. Forward-Return Forecasting for `3d` / `5d` / `20d`

What changed:

- Added `src/ml/backtest/forward_return.py` and `scripts/run_backtest_forward_return.py`.
- Extended target generation to multi-horizon forward returns:
  - `ret_3d`
  - `ret_5d`
  - `ret_20d`
- Applied the evaluation window to realized `target_date`, not anchor `prediction_date`.
- Added a naive flat-return baseline and an optional momentum-continuation baseline.

Why it mattered:

- Moved the project away from next-close-only prediction toward more realistic trading-day horizons.
- Preserved strict time ordering while making target semantics explicit.
- Established a common artifact layout under `artifacts/backtest_forward_return/{3d,5d,20d}/`.

## 5. Strategy Backtest Layer

What changed:

- Added `src/ml/backtest/strategy_backtest.py` and `scripts/run_strategy_backtest.py`.
- Converted saved forward-return forecasts into analysis-only threshold-based strategy backtests.
- Added conservative execution assumptions:
  - signal at `prediction_date` close
  - entry at next tradable open
  - exit at `target_date` close
  - costs on entry and exit
- Added benchmark strategies such as buy-and-hold and naive flat strategy.

Why it mattered:

- Made it possible to ask whether forecast outputs are actionable, not just statistically accurate.
- Added a bridge from prediction quality to signal usability.
- Kept the layer analysis-only rather than mixing research results with execution claims.

## 6. Dual-Task Architecture

What changed:

- Added `src/ml/backtest/dual_task.py` and `scripts/run_dual_task_backtest.py`.
- Extended `src/ml/trainer.py` so the same trainer path can build:
  - regression targets for forward return
  - classification targets for profit/loss after costs
- Added cost-aware profit labels using:
  - next tradable open for entry
  - target-date close for exit
  - per-side transaction fee and slippage
- Added separate regression and classification artifacts under `artifacts/dual_task/`.

Why it mattered:

- Split the problem into two useful views:
  - market movement forecasting
  - trade actionability forecasting
- Kept both tasks aligned on the same underlying data and time-safe split logic.
- Clarified that return prediction alone is not enough once costs are included.

## 7. Combined Signal Layer

What changed:

- Added `src/ml/backtest/combined_signal.py` and `scripts/run_combined_signal_analysis.py`.
- Combined `predicted_return` and `predicted_profit_probability` into analysis-only signal-quality views.
- Added multiple combination styles:
  - weighted linear score
  - gated threshold logic
  - rank-based combination
- Added bucket summaries, top-k ranking summaries, and probability calibration summaries.

Why it mattered:

- Shifted the repo from "single metric per row" to "decision-support signal quality".
- Allowed direct comparison between:
  - return-only ranking
  - probability-only ranking
  - combined ranking
- Showed that the combined layer can help in some slices, but not uniformly.

## 8. Regime-Aware Analysis

What changed:

- Added `src/ml/backtest/regime_aware_analysis.py` and `scripts/run_regime_aware_analysis.py`.
- Added market regime labels:
  - `bull`
  - `bear`
  - `sideway`
- Used rolling benchmark return thresholds on `VNINDEX`, with market-proxy fallback support.
- Joined regimes into regression, classification, and combined-signal outputs using information available up to `prediction_date`.

Why it mattered:

- Exposed regime dependence that global averages were hiding.
- Showed that best classifier and best combined setup can differ by regime even when the best regression model is relatively stable.
- Kept regime assignment explicit and auditable.

## 9. Walk-Forward Regime-Aware Robustness

What changed:

- Added `src/ml/backtest/walk_forward_regime_robustness.py` and `scripts/run_walk_forward_regime_robustness.py`.
- Repeated the forecasting, dual-task, combined-signal, and regime-aware stack across multiple folds.
- Added global stability summaries:
  - model stability
  - horizon stability
  - regime stability
  - combined-method stability
- Added fold-level summaries and cross-fold charts.

Why it mattered:

- Replaced one-window conclusions with repeated out-of-sample checks.
- Made it possible to distinguish:
  - repeatable method-family behavior
  - sample-specific winner noise
- Confirmed that many exact winners remain unstable across folds.

## 10. Regime-Conditioned Meta-Selector

What changed:

- Added `src/ml/backtest/meta_selector.py` and `scripts/run_meta_selector.py`.
- Built an analysis-only selector that chooses candidate setups using prior folds only.
- Added selector modes:
  - `simple_regime_lookup`
  - `regime_weighted_rank`
  - `fallback_global`
- Added audit-friendly fold outputs with reasons and fallback usage.

Why it mattered:

- Turned regime-aware summaries into an adaptive selection experiment.
- Preserved no-leakage by limiting selection history to earlier folds.
- Showed that adaptive selection can help slightly, but the edge is narrow and still sample-sensitive.

## 11. Benchmark Audit and Context-Conditioned Selector

What changed:

- Added `src/ml/backtest/context_meta_selector.py` and `scripts/run_context_meta_selector.py`.
- Added benchmark-audit artifacts under `artifacts/meta_selector_audit/`.
- Added context-conditioned selector modes:
  - `context_knn_selector`
  - `context_bin_lookup`
  - `context_meta_score`
- Expanded the selector context beyond coarse regime labels to continuous market and prediction-state features.

Why it mattered:

- Audited whether suspiciously similar benchmark summaries were a bug or a real consequence of overlapping selected row universes.
- Confirmed that some baseline rows are effectively identical in practice without implying a grouping bug.
- Tested a richer selector design and found that context conditioning is still not clearly stronger than the simpler regime-based layer.

## Current Takeaway

The most important change is not a single model improvement. It is the transition to a disciplined evaluation stack:

- real data instead of placeholders
- explicit train/eval boundaries
- forward-return horizons instead of next-close-only framing
- cost-aware classification
- combined-signal analysis
- regime-aware slicing
- walk-forward robustness
- selector experiments with audit trails

That evolution makes the repository much more useful for research, even though it has not produced a strong universal winner.

## Related Documentation

- [USAGE_GUIDE.md](USAGE_GUIDE.md)
- [ML_IMPLEMENTATION_GUIDE.md](ML_IMPLEMENTATION_GUIDE.md)
- [EVALUATION_WORKFLOWS.md](EVALUATION_WORKFLOWS.md)
- [RESEARCH_FINDINGS_AND_LIMITATIONS.md](RESEARCH_FINDINGS_AND_LIMITATIONS.md)
