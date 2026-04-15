# Usage Guide: Running Workflows and Commands

This guide explains how to run each major workflow in the stock forecasting framework, including expected inputs, outputs, and artifact locations.

---

## Prerequisites

1. **Python 3.8+** with a virtual environment configured
2. **Dependencies installed**:
   ```bash
   pip install -r requirements.txt
   ```
3. **vnstock configured**: The adapter is pre-configured; ensure you can fetch data:
   ```bash
   python -c "from src.data.adapters.vnstock_adapter import VnstockAdapter; a = VnstockAdapter(['ACB']); print('vnstock OK')"
   ```
4. **Data path**: Ensure `data/` directory exists with any pre-downloaded CSV files (optional; vnstock fetches on demand)

---

## 1. Fixed-Window Real-Data Backtest

### Purpose
Train a model once on a fixed historical window, then evaluate daily on a holdout window.

### Command
```bash
python -m scripts.run_backtest_real_data
```

### Configuration
Edit script defaults or pass config via Python:
```python
from src.ml.backtest.real_data import FixedWindowBacktestConfig, RealDataBacktestRunner

config = FixedWindowBacktestConfig(
    tickers=["ACB", "MWG", "DGC"],
    train_start="2020-01-01",
    train_end="2023-12-31",
    eval_start="2024-01-01",
    eval_end="2026-04-10",
    algorithms=["cart", "xgboost", "lightgbm", "sarimax", "ets"],
    output_dir="artifacts/backtest",
)

runner = RealDataBacktestRunner(config)
runner.run()
```

### Outputs
- **Model directory**: `artifacts/backtest/models/{ticker}/` — serialized model manifests
- **Predictions table**: `artifacts/backtest/predicted_vs_actual.csv`
- **Charts**: `artifacts/backtest/charts/` — prediction time series plots
- **Summary**: `artifacts/backtest/summary.json` — aggregate metrics

### Key Metrics
- **Regression**: MAE, RMSE, MAPE, directional accuracy
- **Algorithm comparison**: per-algorithm performance summary

### Dependencies
Optional models skip gracefully if libraries unavailable (e.g., SARIMAX requires `statsmodels`, LSTM requires `tensorflow`).

---

## 2. Forward-Return Multi-Horizon Backtest

### Purpose
Predict forward returns at 3-day, 5-day, and 20-day horizons on real data.

### Command
```bash
python -m scripts.run_backtest_forward_return
```

### Configuration
```python
from src.ml.backtest.forward_return import ForwardReturnBacktestConfig, ForwardReturnBacktestRunner

config = ForwardReturnBacktestConfig(
    tickers=["ACB", "MWG", "DGC"],
    train_start="2020-01-01",
    train_end="2023-12-31",
    eval_start="2024-01-01",
    eval_end="2026-04-10",
    algorithms=["cart", "xgboost", "lightgbm", "sarimax", "ets"],
    horizons=["3d", "5d", "20d"],
    output_dir="artifacts/backtest_forward_return",
)

runner = ForwardReturnBacktestRunner(config)
runner.run()
```

### Outputs
- **Per-horizon results**: `artifacts/backtest_forward_return/{3d,5d,20d}/`
  - `predicted_vs_actual.csv` — daily predictions and actuals
  - `summary.json` — horizon-specific metrics
  - `charts/` — visualization

### Key Metrics
Same as fixed-window, but computed per horizon independently.

### Notes
- Targets are time-safe: `close[t+horizon] / close[t] - 1`
- Last `horizon` rows will be NaN (no future data to predict)
- Momentum baseline included for reference

---

## 3. Dual-Task Backtest (Regression + Classification)

### Purpose
Simultaneously train regression (forward return) and classification (profit/loss after costs) branches.

### Command
```bash
python -m scripts.run_dual_task_backtest
```

### Configuration
```python
from src.ml.backtest.dual_task import DualTaskBacktestConfig, DualTaskBacktestRunner

config = DualTaskBacktestConfig(
    tickers=["ACB", "MWG", "DGC"],
    train_start="2020-01-01",
    train_end="2023-12-31",
    eval_start="2024-01-01",
    eval_end="2026-04-10",
    algorithms=["cart", "xgboost", "lightgbm", "sarimax", "ets"],
    horizons=["3d", "5d", "20d"],
    transaction_fee_bps=15.0,       # basis points
    slippage_bps=20.0,               # basis points
    output_dir="artifacts/dual_task",
)

runner = DualTaskBacktestRunner(config)
runner.run()
```

### Outputs
- **Regression branch**: `artifacts/dual_task/regression/{horizon}/`
  - `predicted_vs_actual.csv`
  - Regression metrics (MAE, RMSE, MAPE, directional accuracy)
- **Classification branch**: `artifacts/dual_task/classification/{horizon}/`
  - `predicted_vs_actual.csv` (with probability column)
  - Classification metrics (accuracy, precision, recall, F1, ROC-AUC)

### Key Metrics
**Regression**:
- MAE, RMSE, MAPE
- Directional accuracy (% of up/down predictions correct)

**Classification**:
- Accuracy, Precision, Recall, F1
- ROC-AUC (if probability available)
- Confusion matrix breakdown

### Notes
- Profit label computed as: `(close[t+horizon] / close[t]) * (1 - costs) > 1.0`
- Costs = transaction_fee_bps / 10000 + slippage_bps / 10000
- Imbalanced classes expected (fewer big winners)

---

## 4. Combined Signal Analysis

### Purpose
Fuse predicted forward return and predicted profit probability into a single decision-support score.

### Command
```bash
python -m scripts.run_combined_signal_analysis
```

### Configuration
```python
from src.ml.backtest.combined_signal import CombinedSignalConfig, CombinedSignalAnalysisRunner

config = CombinedSignalConfig(
    dual_task_dir="artifacts/dual_task",
    output_dir="artifacts/combined_signal",
    horizons=["3d", "5d", "20d"],
    return_thresholds=[0.0, 0.005, 0.01, 0.02],
    probability_thresholds=[0.50, 0.55, 0.60, 0.65],
    w_return=0.5,                  # weight for return signal
    w_profit=0.5,                  # weight for profit probability
    top_k_values=[1, 3, 5],        # for top-k ranking
)

runner = CombinedSignalAnalysisRunner(config)
runner.run()
```

### Outputs
- **Combined table**: `artifacts/combined_signal/{horizon}/combined_signal_table.csv`
  - Columns: date, ticker, model_name, horizon, actual_return, predicted_return, actual_profit_label, predicted_profit_label, predicted_profit_probability, combined_score
  - Rankings per date (top-K signal rankings)

### Key Metrics
- **Combined score**: weighted average of normalized return and profit scores
- **Top-K accuracy**: % of top-K selected positions that were profitable
- **Hit rate**: % of positive signals that landed in positive direction

### Notes
- Requires successful dual_task_backtest run
- Supports multiple ranking methods: simple, weighted linear, gated thresholds
- Useful for portfolio construction (rank by combined score, take top K)

---

## 5. Regime-Aware Analysis

### Purpose
Attach market regime labels (bull/bear/sideway) to evaluation outputs and summarize metrics by regime.

### Command
```bash
python -m scripts.run_regime_aware_analysis
```

### Configuration
```python
from src.ml.backtest.regime_aware_analysis import RegimeAwareAnalysisConfig, RegimeAwareAnalysisRunner

config = RegimeAwareAnalysisConfig(
    dual_task_dir="artifacts/dual_task",
    combined_signal_dir="artifacts/combined_signal",
    output_dir="artifacts/regime_aware_analysis",
    horizons=["3d", "5d", "20d"],
    benchmark_symbol="VNINDEX",
    benchmark_source="vnindex_or_market_proxy",
    regime_method="rolling_return_threshold",
    regime_lookback_days=20,
    bull_threshold=0.03,           # 3% return over lookback = bull
    bear_threshold=-0.03,          # -3% return over lookback = bear
)

runner = RegimeAwareAnalysisRunner(config)
runner.run()
```

### Outputs
- **Regime-conditioned metrics**: `artifacts/regime_aware_analysis/summary/{horizon}/`
  - `bull.csv` — metrics for bull regime
  - `bear.csv` — metrics for bear regime
  - `sideway.csv` — metrics for sideway regime
  - `regime_distribution.json` — % time in each regime

### Key Metrics (Per Regime)
- Regression: MAE, RMSE, MAPE, directional accuracy
- Classification: accuracy, precision, recall, F1, ROC-AUC
- Win rates per model-horizon pair

### Notes
- Regime detection based on VNINDEX or market proxy rolling return
- Can be customized to use other regime definitions (volatility-based, etc.)
- Exposes which models thrive in trending vs. sideways markets

---

## 6. Walk-Forward Regime-Aware Robustness

### Purpose
Run multiple temporal folds (expanding or rolling) through the entire stack and measure cross-fold stability.

### Command
```bash
python -m scripts.run_walk_forward_regime_robustness
```

### Configuration
```python
from src.ml.backtest.walk_forward_regime_robustness import WalkForwardRegimeRobustnessConfig, WalkForwardRegimeRobustnessRunner

config = WalkForwardRegimeRobustnessConfig(
    tickers=["ACB", "MWG", "DGC"],
    train_start="2020-01-01",
    first_eval_start="2023-01-01",
    last_eval_end="2026-04-10",
    eval_window_days=60,            # 60-day eval window per fold
    step_size_days=30,              # 30-day advance between folds
    max_folds=4,
    horizons=["3d", "5d", "20d"],
    algorithms=["cart", "xgboost", "lightgbm"],
    training_window_mode="expanding",  # or "rolling"
    rolling_train_window_days=None,    # set if using rolling mode
    output_dir="artifacts/walk_forward_regime_robustness",
)

runner = WalkForwardRegimeRobustnessRunner(config)
runner.run()
```

### Outputs
- **Fold results**: `artifacts/walk_forward_regime_robustness/fold_{N}/`
  - Each fold = complete dual_task + combined_signal + regime_analysis
- **Cross-fold summary**: `artifacts/walk_forward_regime_robustness/summary/`
  - `cross_fold_stability.csv` — win rates across folds per model-horizon
  - `fold_breakdown.csv` — fold-by-fold comparison
  - `regime_stability.json` — stability by regime

### Key Metrics
- **Win rate**: % of folds where model outperforms baseline
- **Stability level**: high (≥70%), medium (45-70%), low (<45%)
- **Consistency**: which models perform well across ALL folds vs. just peak period

### Notes
- **Expanding mode**: each fold uses all prior data up to that point (realistic for live trading)
- **Rolling mode**: uses fixed-size recent window (stabler but less data per fold)
- Long-running: each fold requires full train + eval cycle
- Can retry failed folds automatically with backoff

---

## 7. Meta-Selector: Regime-Conditioned Candidate Selection

### Purpose
From walk-forward results, select the best model-horizon-threshold combination for each market regime.

### Command
```bash
python -m scripts.run_meta_selector
```

### Configuration
```python
from src.ml.backtest.meta_selector import MetaSelectorConfig, MetaSelectorRunner

config = MetaSelectorConfig(
    walk_forward_dir="artifacts/walk_forward_regime_robustness",
    output_dir="artifacts/meta_selector",
    selector_modes=["simple_regime_lookup", "regime_weighted_rank", "fallback_global"],
    minimum_prior_folds_per_regime=2,
    minimum_samples_per_regime=30,
    primary_top_k=3,
    utility_weight_topk_avg_return=0.40,
    utility_weight_topk_profit_rate=0.30,
    utility_weight_positive_class_precision=0.20,
    utility_weight_directional_accuracy=0.10,
)

runner = MetaSelectorRunner(config)
runner.run()
```

### Outputs
- **Candidates table**: `artifacts/meta_selector/candidates.csv`
  - Ranked candidates per regime (bull, bear, sideway)
  - Scores: topk_avg_return, topk_profit_rate, precision, directional accuracy
  - Recommendation label (rank 1, 2, 3)
- **Selector modes**: per mode, best recommendations
- **Summary**: `artifacts/meta_selector/selector_summary.json`

### Key Metrics
- **Utility score**: weighted blend of return, profit_rate, precision, directional_accuracy
- **Rank scores**: normalized ranking of each metric
- **Weighted rank**: final composite score for recommendation

### Notes
- Uses walk-forward results to avoid double-dipping
- Multiple selector modes support different deployment scenarios:
  - `simple_regime_lookup`: if market is bull, use bull-best model
  - `regime_weighted_rank`: blend across regimes based on current regime confidence
  - `fallback_global`: if regime-specific data sparse, use global winner
- Minimal improvement expected (walk-forward already selected good models)

---

## 8. Model Comparison Workflow

### Purpose
Compare performance of different model families on a fixed dataset.

### Command
```bash
python -m scripts.run_backtest_model_comparison
```

### Outputs
- Per-model summary: `artifacts/backtest_model_comparison/`
  - Algorithm family comparison chart
  - Cross-model ranking by metric

---

## 9. Strategy Backtest (Optional)

### Purpose
Simulate portfolio allocation and rebalancing strategies.

### Command
```bash
python -m scripts.run_strategy_backtest
```

### Notes
Requires successful prior runs of combined_signal or regime_aware_analysis.

---

## Complete End-to-End Workflow

```bash
# 1. Fixed-window baseline
python -m scripts.run_backtest_real_data

# 2. Multi-horizon forward returns
python -m scripts.run_backtest_forward_return

# 3. Add profit/loss classification
python -m scripts.run_dual_task_backtest

# 4. Fuse return + profit signals
python -m scripts.run_combined_signal_analysis

# 5. Attach market regimes
python -m scripts.run_regime_aware_analysis

# 6. Test temporal stability (hours → days to run)
python -m scripts.run_walk_forward_regime_robustness

# 7. Select best per-regime model
python -m scripts.run_meta_selector
```

All artifacts automatically saved to `artifacts/` with clear subdirectories.

---

## Expected Output Structure

```
artifacts/
├── backtest/
│   ├── models/
│   ├── predicted_vs_actual.csv
│   └── charts/
├── backtest_forward_return/
│   ├── 3d/, 5d/, 20d/
│   │   ├── predicted_vs_actual.csv
│   │   ├── summary.json
│   │   └── charts/
├── dual_task/
│   ├── regression/
│   │   └── {horizon}/
│   └── classification/
│       └── {horizon}/
├── combined_signal/
│   └── {horizon}/
│       └── combined_signal_table.csv
├── regime_aware_analysis/
│   └── summary/
│       └── {horizon}/
│           ├── bull.csv
│           ├── bear.csv
│           └── sideway.csv
├── walk_forward_regime_robustness/
│   ├── fold_1/, fold_2/, fold_3/, fold_4/
│   └── summary/
│       └── cross_fold_stability.csv
└── meta_selector/
    ├── candidates.csv
    └── selector_summary.json
```

---

## Troubleshooting

### Missing vnstock data
- Check internet connection
- Verify ticker symbols: use uppercase (ACB, MWG, DGC)
- Data is cached; delete `data/` subdirectories to force refetch

### Model training fails
- Optional models skip gracefully; check logs for missing library warnings
- Ensure at least 60 training samples per ticker-horizon
- Check date windows don't contain gaps

### Low eval performance
- Normal for short horizons and Vietnamese market conditions
- Regime-aware analysis will show which regimes enable better predictions
- Meta-selector focuses on stability, not peak numbers

### Out of memory
- Reduce number of tickers
- Reduce eval window in walk-forward (e.g., from 60 to 30 days)
- Use rolling instead of expanding window mode

---

## Related Documentation

- [CHANGELOG_SUMMARY.md](./CHANGELOG_SUMMARY.md) — System evolution
- [ML_IMPLEMENTATION_GUIDE.md](./ML_IMPLEMENTATION_GUIDE.md) — Internal details
- [EVALUATION_WORKFLOWS.md](./EVALUATION_WORKFLOWS.md) — Detailed workflow specs
