# VN100 Cost and Slippage Readiness Review

## Reviewed Evidence

- Official VN100 artifact family: `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff`.
- Official prediction files: `daily/predicted_vs_actual.csv` and `hourly/predicted_vs_actual.csv`.
- Official confidence and regime files: `confidence_threshold_sweep_summary.csv`, `daily/regime_accuracy_summary.csv`, and `hourly/regime_accuracy_summary.csv`.
- Backtest modules inspected: `src/ml/backtest/strategy_backtest.py`, `src/ml/backtest/signal_effectiveness.py`, `src/ml/backtest/event_driven.py`, `src/ml/benchmark/evaluator.py`, and `src/ml/benchmark/system_benchmark.py`.

## Findings

| Question | Current answer | Evidence |
|---|---|---|
| Have selected signal slices been evaluated with transaction cost? | No official evidence for the selected VN100 slices. | The official artifact files contain prediction, confidence, regime, and directional-correctness fields, but no transaction-cost configuration or cost-adjusted strategy output. |
| Is slippage modeled? | Supported in backtest modules, but not applied to the official VN100 selected slices. | `strategy_backtest.py` has `transaction_fee_bps` and `slippage_bps`; `event_driven.py` has `simulate_execution_cost`; `signal_effectiveness.py` supports cost/slippage grids. |
| Do turnover metrics exist? | Supported outside the official selected-slice artifacts. | `strategy_backtest.py` computes turnover and portfolio turnover definitions; official VN100 artifacts do not include turnover fields. |
| Do drawdown metrics exist? | Supported outside the official selected-slice artifacts. | `strategy_backtest.py` and `src/ml/benchmark/evaluator.py` compute max drawdown; official VN100 artifacts do not include drawdown fields. |
| Does profit factor exist? | Supported outside the official selected-slice artifacts. | `strategy_backtest.py`, `signal_effectiveness.py`, and `src/ml/benchmark/evaluator.py` compute profit factor; official VN100 artifacts do not include it. |
| Do cost-adjusted returns exist for the official selected slices? | No. | No official selected-slice artifact provides net return, cost-adjusted return, PnL, equity curve, trade list, or portfolio result. |

## Selected VN100 Slices Reviewed

| Slice | Official evidence available | Trading-readiness status |
|---|---|---|
| Hourly stacking h=1, confidence threshold 0.57 | Filtered accuracy 0.6003482803656944, coverage 0.3129854203569969, evaluated rows 2,297. | Directional diagnostic only; no cost-adjusted backtest. |
| Daily LightGBM h=20, bear regime | Regime accuracy 0.6959459459459459, observations 444. | Regime diagnostic only; no ex-ante trading rule or cost-adjusted backtest. |
| Hourly stacking h=1, high-volatility regime | Regime accuracy 0.5651093439363817, observations 2,012. | Regime diagnostic only; no cost-adjusted backtest. |

## What Is Still Missing

Before making practical trading-readiness claims, the project still needs:

1. A deterministic mapping from official selected prediction rows to executable signals.
2. Entry and exit price conventions tied to available OHLCV data.
3. Transaction-cost and slippage assumptions applied to every trade.
4. Net return, equity curve, turnover, drawdown, profit factor, exposure, and trade-count artifacts.
5. Buy-and-hold, flat, and simple signal baselines under the same cost assumptions.
6. Liquidity and partial-fill assumptions for VN100 tickers.
7. Multi-window validation showing that selected confidence and regime slices are not single-window artifacts.
8. A rule that prevents post-hoc regime selection from being treated as an ex-ante deployable strategy.

## Claim Boundary

The current official VN100 evidence supports directional-accuracy diagnostics only. It does not establish trading profitability, cost-adjusted performance, deployment readiness, or practical investment suitability.
