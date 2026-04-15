# Evaluation Workflows: Detailed Specifications

This document describes each evaluation workflow: its purpose, inputs, outputs, metrics, and artifact locations.

---

## 1. Real-Data Backtest Workflow

### Purpose
Train a single model on a fixed historical window, then evaluate daily on a held-out test window. Baseline method that avoids temporal leakage.

### Input Data
- **Source**: vnstock OHLCV (daily bars)
- **Tickers**: Configurable list (e.g., ["ACB", "MWG", "DGC"])
- **Train period**: Fixed (e.g., 2020-01-01 to 2023-12-31)
- **Eval period**: Fixed (e.g., 2024-01-01 to 2026-04-10)

### Process

1. **Fetch OHLCV data** for all tickers + dates
2. **Compute features** (technical indicators, context, risk) for all dates
3. **Split data**: 
   - Train: rows 0 to train_stop
   - Validation: rows train_stop to val_stop (15% default)
   - Test: rows val_stop to end
4. **Train models** for each algorithm (CART, XGBoost, LightGBM, SARIMAX, ETS)
   - Each model trained on train set
   - Validated on validation set
   - Early stopping (for LSTM)
5. **Evaluate daily** on test set
   - For each day, each model predicts forward return + profit probability
   - Collect predictions
6. **Compute metrics**:
   - Regression: MAE, RMSE, MAPE, directional accuracy
   - Classification: accuracy, precision, recall, F1, ROC-AUC
7. **Generate artifacts**

### Outputs

**Directory**: `artifacts/backtest/`

```
artifacts/backtest/
├── models/
│   ├── {ticker}/
│   │   ├── cart.json                    # Manifest
│   │   ├── xgboost.json
│   │   └── ...
├── predicted_vs_actual.csv              # Compiled daily predictions
├── summary.json                         # Aggregate metrics
└── charts/
    └── {ticker}_predictions.png
```

**predicted_vs_actual.csv columns**:
- date, ticker, model_name, horizon
- predicted_return, actual_return
- predicted_profit_label, actual_profit_label
- predicted_profit_probability

**summary.json structure**:
```json
{
  "overall_metrics": {
    "cart": {"mae": 0.045, "rmse": 0.062, "mape": 2.3, "directional_accuracy": 0.48},
    "xgboost": {"mae": 0.041, "rmse": 0.058, ...}
  },
  "per_ticker": {
    "ACB": {...},
    "MWG": {...}
  },
  "model_family_comparison": {...}
}
```

### Metrics

| Metric | Calculation | Interpretation |
|--------|-----------|---|
| MAE | mean(\|actual - predicted\|) | Avg absolute error in returns |
| RMSE | sqrt(mean((actual - predicted)²)) | Penalizes large errors |
| MAPE | mean(\|actual - predicted\| / \|actual\|) | % error (scale-agnostic) |
| Directional accuracy | % of up/down predictions correct | "Did we get the direction right?" |
| Accuracy (classification) | % of correct profit/loss predictions | Overall classifier performance |
| Precision | TP / (TP + FP) | Of predicted profits, % were correct |
| Recall | TP / (TP + FN) | Of actual profits, % were found |
| F1 | 2 * (precision * recall) / (precision + recall) | Harmonic mean |
| ROC-AUC | Area under receiver-operator curve | Performance at all thresholds |

---

## 2. Forward-Return Multi-Horizon Workflow

### Purpose
Predict forward returns at multiple time horizons (3d, 5d, 20d) independently. Shows how prediction quality varies with forecast horizon.

### Input Data
- **Same source**: vnstock OHLCV
- **Horizons**: 3-day, 5-day, 20-day (configurable)
- **Train/eval period**: Same as real-data backtest

### Process

For each horizon h in [3d, 5d, 20d]:
1. **Compute horizon-specific target**: `forward_return[t] = close[t+h] / close[t] - 1`
2. **Train models** (CART, XGBoost, LightGBM, SARIMAX, ETS) on this target
3. **Evaluate** on hold-out test window
4. **Compute metrics** per horizon
5. **Compare across horizons**: Which is most predictable?

### Outputs

**Directory**: `artifacts/backtest_forward_return/`

```
artifacts/backtest_forward_return/
├── 3d/
│   ├── predicted_vs_actual.csv
│   ├── summary.json
│   └── charts/
├── 5d/
│   └── ...
├── 20d/
│   └── ...
└── cross_horizon_comparison.json      # Metrics per horizon
```

**summary.json per horizon** includes:
- Model-by-model metrics (MAE, RMSE, MAPE, directional accuracy)
- Best performing model per horizon
- Metric comparison: easier to predict at which horizon?

### Metrics

Same as real-data backtest, computed per horizon independently.

**Key question answered**: "Is it easier to predict 3-day moves or 20-day moves?"

Typical finding: Longer horizons often more predictable (trends clearer), but fewer data points.

---

## 3. Dual-Task Backtest Workflow

### Purpose
Train AND evaluate both regression (forward return) and classification (profit/loss) branches in parallel. Understand cost-inclusive profitability, not just returns.

### Configuration
```python
transaction_fee_bps = 15.0       # Entry/exit fees
slippage_bps = 20.0              # Unfavorable execution vs. closing price
```

Total cost = (15 + 20) / 10000 = 0.35% per trade.

### Process

For each horizon h:
1. **Regression target**: `forward_return[t] = close[t+h] / close[t] - 1`
2. **Classification target**: 
   ```
   profit_label[t] = 1 if (close[t+h] / close[t]) * (1 - 0.0035) > 1.0 else 0
   ```
3. **Train regression models** on forward_return
4. **Train classification models** on profit_label
5. **Evaluate both branches** on test set
6. **Collect metrics** separately per branch

### Outputs

**Directory**: `artifacts/dual_task/`

```
artifacts/dual_task/
├── regression/
│   ├── 3d/
│   │   ├── predicted_vs_actual.csv
│   │   ├── summary.json
│   │   └── charts/
│   ├── 5d/
│   ├── 20d/
├── classification/
│   ├── 3d/
│   │   ├── predicted_vs_actual.csv
│   │   ├── summary.json (includes confusion matrix)
│   │   └── charts/
│   ├── 5d/
│   ├── 20d/
└── dual_task_comparison.json         # Branch-by-branch comparison
```

**predicted_vs_actual.csv** columns:
- date, ticker, model_name, horizon
- (regression) predicted_return, actual_return
- (classification) predicted_profit_label, predicted_profit_probability, actual_profit_label

### Metrics

**Regression branch**:
- MAE, RMSE, MAPE, directional accuracy (same as forward-return)

**Classification branch**:
- Accuracy, precision, recall, F1, ROC-AUC
- Confusion matrix (TP, FP, FN, TN)

### Key Insights

- **Difference between branches**: Regression MAE < 2% but classification accuracy only 52% → High precision required to be profitable
- **Class imbalance**: If positive class is 45%, baseline accuracy is 55% (always predict loss)
- **Threshold sensitivity**: Moving decision threshold changes false positive rate

---

## 4. Combined Signal Analysis Workflow

### Purpose
Merge regression predictions and classification probabilities into a single decision signal. Reduces noise via consensus.

### Configuration
```python
w_return = 0.5                           # Weight for return signal
w_profit = 0.5                           # Weight for profit probability
return_thresholds = [0.0, 0.005, 0.01]  # Threshold levels for return signal
probability_thresholds = [0.50, 0.55, 0.60]  # Threshold levels for profit signal
top_k_values = [1, 3, 5]                # For top-K portfolio construction
```

### Process

For each date and horizon:
1. **Collect regression predictions** (predicted_return)
2. **Collect classification probabilities** (predicted_profit_probability)
3. **Normalize both signals** to [0, 1]
4. **Combine**: `combined_score = w_return * norm_return + w_profit * norm_profit`
5. **Apply thresholds** to classify signal strength
6. **Rank by combined score** and select top-K
7. **Evaluate top-K portfolio**: % of top-K selections that were profitable

### Outputs

**Directory**: `artifacts/combined_signal/`

```
artifacts/combined_signal/
├── 3d/
│   ├── combined_signal_table.csv
│   ├── summary.json
│   └── charts/
├── 5d/
├── 20d/
└── signal_analysis.json              # Threshold sensitivity analysis
```

**combined_signal_table.csv** columns:
- date, ticker, model_name, horizon
- predicted_return, predicted_profit_probability
- combined_score (0-1)
- signal_strength (weak/medium/strong based on thresholds)
- rank_today (1-inf, ranked by combined_score per date)
- actual_return, actual_profit_label

### Metrics

**Top-K accuracy**:
```
top_k_accuracy = (# of top-K selections that were profitable) / (# of top-K selections)
```

**Hit rate per signal strength**:
- weak: % accuracy if signal_strength == 'weak'
- medium: % accuracy if signal_strength == 'medium'
- strong: % accuracy if signal_strength == 'strong'

### Key Insights
- Expecting ~55-60% accuracy (baseline is 50% for binary)
- Stronger signal should correlate with higher accuracy
- Threshold sensitivity: changing return_threshold may improve precision but reduce recall

---

## 5. Regime-Aware Analysis Workflow

### Purpose
Partition evaluation results by market regime (bull/bear/sideway) and compute metrics per regime. Uncover regime-specific model performance.

### Configuration
```python
regime_method = "rolling_return_threshold"
regime_lookback_days = 20           # Window for computing regime
bull_threshold = 0.03               # Return > 3% = bull
bear_threshold = -0.03              # Return < -3% = bear
```

### Process

1. **Compute market regime** for each date:
   - Rolling 20-day return of VNINDEX or market proxy
   - If return > 0.03: BULL
   - If return < -0.03: BEAR
   - Else: SIDEWAY
2. **Load dual_task and combined_signal results**
3. **Attach regime label** to each prediction
4. **Partition by regime**: split into bull_results, bear_results, sideway_results
5. **Compute metrics** per regime:
   - Regression: MAE, RMSE, MAPE, directional accuracy
   - Classification: accuracy, precision, recall, F1
6. **Generate summary tables**

### Outputs

**Directory**: `artifacts/regime_aware_analysis/`

```
artifacts/regime_aware_analysis/
└── summary/
    ├── 3d/
    │   ├── bull.csv               # Metrics for bull regime
    │   ├── bear.csv               # Metrics for bear regime
    │   ├── sideway.csv            # Metrics for sideway regime
    │   ├── regime_distribution.json  # % time in each regime
    │   └── charts/
    ├── 5d/
    ├── 20d/
    └── cross_regime_comparison.json   # Which models win in which regimes?
```

**Example bull.csv**:
```
model_name,observations,mae,rmse,mape,directional_accuracy,accuracy,precision,recall,f1
cart,150,0.038,0.055,1.8,0.51,0.54,0.52,0.45,0.48
xgboost,150,0.035,0.051,1.6,0.53,0.56,0.55,0.48,0.51
...
```

### Metrics

Same as dual-task, but computed per regime:
- Regression: MAE, RMSE, MAPE, directional accuracy
- Classification: accuracy, precision, recall, F1, ROC-AUC

**Regime-specific statistics**:
- `regime_distribution.json`:
  ```json
  {
    "bull": {"observations": 400, "pct": 35.2},
    "bear": {"observations": 150, "pct": 13.1},
    "sideway": {"observations": 600, "pct": 51.7}
  }
  ```

### Key Insights
- **Model stability across regimes**: Does XGBoost win in ALL regimes, or only bull?
- **Regime-specific winners**: "SARIMAX wins in bear" vs. "CART wins in sideway"
- **Regime bias**: If training data was mostly bull, eval metrics may inflate

---

## 6. Walk-Forward Regime-Aware Robustness Workflow

### Purpose
Assess method STABILITY across multiple time periods (folds). Answers: "Does this work reliably across different market windows, or just in the sweet spot?"

### Configuration
```python
train_start = "2020-01-01"
first_eval_start = "2023-01-01"
last_eval_end = "2026-04-10"
eval_window_days = 60              # Per-fold eval window
step_size_days = 30                # Advance between folds
max_folds = 4                      # Number of temporal folds
training_window_mode = "expanding" # or "rolling"
```

### Process

1. **Generate folds** (see [ML Implementation Guide](./ML_IMPLEMENTATION_GUIDE.md#fold-generation))
2. **For each fold** (in sequential time order):
   - Train models on fold's training window
   - Evaluate on fold's eval window
   - Run full pipeline: dual_task → combined_signal → regime_aware_analysis
   - Save results to `fold_{N}/`
3. **Aggregate cross-fold**:
   - For each model-horizon pair, collect results from all folds
   - Compute win rates: % of folds where model beat baseline
   - Compute stability level (high/medium/low based on win rate)
4. **Generate summary**

### Outputs

**Directory**: `artifacts/walk_forward_regime_robustness/`

```
artifacts/walk_forward_regime_robustness/
├── fold_1/
│   ├── dual_task/
│   ├── combined_signal/
│   └── regime_aware_analysis/
├── fold_2/
├── fold_3/
├── fold_4/
└── summary/
    ├── cross_fold_stability.csv
    ├── fold_breakdown.csv
    ├── regime_stability_{regime}.json
    └── charts/
```

**cross_fold_stability.csv**:
```
model,horizon,fold_count,win_rate,stability_level,avg_mae,avg_acc
cart,3d,4,0.75,high,0.043,0.52
cart,5d,4,0.50,medium,0.048,0.51
xgboost,3d,4,0.25,low,0.045,0.50
...
```

**fold_breakdown.csv**:
```
fold,model,horizon,mae,acc,profit_rate
1,cart,5d,0.041,0.53,0.48
2,cart,5d,0.046,0.50,0.45
3,cart,5d,0.039,0.55,0.52
4,cart,5d,0.052,0.48,0.43
```

### Metrics

**Cross-fold aggregation**:
- **Win rate**: (# folds where model beat baseline) / (total folds)
  - **High stability**: win_rate >= 0.70
  - **Medium**: 0.45 ≤ win_rate < 0.70
  - **Low**: win_rate < 0.45
- **Average MAE**: mean of MAE across folds
- **Average accuracy**: mean of classification accuracy across folds
- **Std dev MAE**: temporal volatility (higher = less stable)

### Key Insights
- **Baseline: 50% win rate** = coin flip (no real edge)
- **70%+ win rate** = method is likely robust, not overfitted to one period
- **Stability variance**: "CART wins in folds 1&2 but loses in 3&4" = regime-dependent
- **Longer horizons**: Often have fewer wins (harder to predict)

---

## 7. Meta-Selector Workflow

### Purpose
From walk-forward results, automatically select the best model-horizon-threshold combination for EACH market regime. Operationalizes research findings into recommendations.

### Configuration
```python
minimum_prior_folds_per_regime = 2     # Need at least 2 folds for regime
minimum_samples_per_regime = 30         # And at least 30 observations
primary_top_k = 3                       # Rank metric: top-3 selection accuracy
selector_modes = [
    "simple_regime_lookup",             # If bull, use bull-best
    "regime_weighted_rank",             # Blend across regimes
    "fallback_global",                  # Use global winner if regime data sparse
]
```

### Process

1. **Load walk-forward results** for all folds
2. **For each regime** (bull, bear, sideway):
   - Filter to observations in this regime
   - Check if enough samples (>= min_samples) and folds (>= min_folds)
   - Rank candidates by utility score:
     ```
     utility = 0.40 * topk_avg_return 
             + 0.30 * topk_profit_rate
             + 0.20 * positive_class_precision
             + 0.10 * directional_accuracy
     ```
   - Select top 3 candidates
3. **Generate selector recommendations**:
   - Mode 1 (simple): Pick top candidate per regime
   - Mode 2 (weighted): Blend top 3 based on regime confidence
   - Mode 3 (fallback): Use global winner if regime sparse
4. **Save recommendations**

### Outputs

**Directory**: `artifacts/meta_selector/`

```
artifacts/meta_selector/
├── candidates.csv                      # Ranked candidates per regime
├── selector_modes/
│   ├── simple_regime_lookup.json
│   ├── regime_weighted_rank.json
│   └── fallback_global.json
└── selector_summary.json               # Recommendation summary
```

**candidates.csv**:
```
regime,rank,model,horizon,ranking_method,return_threshold,probability_threshold,sample_count,prior_fold_count,topk_average_return,topk_profit_rate,positive_class_precision,directional_accuracy,utility_score,recommendation_label
bull,1,xgboost,5d,combined_topk,0.005,0.60,180,3,0.0148,0.52,0.51,0.53,0.542,rank1
bull,2,lightgbm,5d,combined_topk,0.01,0.60,180,3,0.0142,0.50,0.49,0.51,0.519,rank2
bear,1,sarimax,3d,combined_topk,0.0,0.55,45,2,0.0102,0.48,0.45,0.48,0.468,rank1
...
```

**simple_regime_lookup.json**:
```json
{
  "bull": {
    "model": "xgboost",
    "horizon": "5d",
    "confidence": 0.85,
    "expected_accuracy": 0.52
  },
  "bear": {
    "model": "sarimax",
    "horizon": "3d",
    "confidence": 0.60,
    "expected_accuracy": 0.48
  },
  "sideway": {
    "model": "cart",
    "horizon": "5d",
    "confidence": 0.72,
    "expected_accuracy": 0.50
  }
}
```

### Metrics

**Utility score components**:
- topk_average_return: Expected return of top-K selected portfolio
- topk_profit_rate: % of top-K that were profitable
- positive_class_precision: % of predicted profits that were correct
- directional_accuracy: % of direction predictions correct

### Key Insights
- **Minimal improvement expected**: Walk-forward already filtered for good methods
- **Regime sparsity**: Bear regime likely has sparse data (rare crises) → fallback to global winner
- **Selector mode choice**: 
  - simple_regime_lookup: operationally clean but ignores regime uncertainty
  - regime_weighted_rank: more robust but requires inference-time regime probabilities
  - fallback_global: safest if live trading (avoid tiny-sample bets)

---

## Workflow Comparison

| Workflow | Focus | Key Metric | Typical Runtime |
|----------|-------|-----------|---|
| Real-data backtest | Baseline | Directional accuracy | ~10 min |
| Forward-return | Multi-horizon | MAE per horizon | ~10 min |
| Dual-task | Cost-aware | Profit accuracy | ~20 min |
| Combined signal | Consensus | Top-K hit rate | ~5 min |
| Regime-aware | Conditional | Win rate per regime | ~2 min |
| Walk-forward | Stability | Cross-fold win rate | ~hours |
| Meta-selector | Selection | Utility score ranking | ~1 min |

---

## Related Documentation

- [USAGE_GUIDE.md](./USAGE_GUIDE.md) — How to run each workflow
- [ML_IMPLEMENTATION_GUIDE.md](./ML_IMPLEMENTATION_GUIDE.md) — Internal architecture
- [RESEARCH_FINDINGS_AND_LIMITATIONS.md](./RESEARCH_FINDINGS_AND_LIMITATIONS.md) — Empirical results
