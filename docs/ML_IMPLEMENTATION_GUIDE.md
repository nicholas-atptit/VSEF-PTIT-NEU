# ML Implementation Guide: Internal Architecture and Data Flow

This document explains how the ML system is structured internally, including the data flow, feature engineering, model training, target generation, and key architectural decisions.

---

## System Architecture Overview

```
Data Fetch (vnstock OHLCV)
    ↓
Feature Engineering
    ├→ Base Technical Features
    ├→ Context Features (market, sector, sentiment)
    └→ Risk Features (VaR, CVaR, drawdown)
    ↓
Target Generation
    ├→ Regression branch: forward_return[horizon]
    └→ Classification branch: profit_label[horizon]
    ↓
Trainer (DualModelTrainer)
    ├→ Create sequence dataset (for LSTM)
    ├→ Split: train / val / test
    ├→ Train models: CART, XGBoost, LightGBM, SARIMAX, ETS, LSTM
    └→ Save manifests
    ↓
Inference (per-day evaluation)
    ├→ Load features
    ├→ Run trained models
    ├→ Collect predictions + probabilities
    ├→ Compute evaluation metrics
    └→ Save artifacts
    ↓
Post-processing
    ├→ Combined signal fusion
    ├→ Regime detection & conditioning
    ├→ Walk-forward aggregation
    └→ Meta-selector ranking
```

---

## Data Flow in Detail

### 1. Real-World Data Fetch

```python
# src/ml/backtest/real_data.py
adapter = VnstockAdapter(symbol_list=["ACB", "MWG"])
ohlcv_df = adapter.fetch(
    start_date="2020-01-01",
    end_date="2026-04-10",
    interval="1D"  # Daily
)
# Returns: DataFrame with date, ticker, open, high, low, close, volume
```

**Key properties**:
- Real Vietnamese market data from vnstock
- OHLCV (open, high, low, close, volume) columns
- Daily bars by default
- Handles ticker delisting, gaps, weekends

### 2. Feature Engineering

#### Base Technical Features

```python
from src.ml.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
features_df = engineer.compute_base_features(
    ohlcv_df,
    lookback_days=20,  # for RSI, SMA, etc.
)
```

**Generated features** (typical set):
- **Momentum**: close-to-close return, log return
- **Trend**: SMA(5), SMA(20), EMA(12), EMA(26)
- **Volatility**: rolling std, ATR, high-low range
- **Volume**: volume MA, volume ratio
- **Reversal indicators**: RSI, MACD
- **Mean reversion**: Bollinger bands

**Time-safety principle**: All features use **historical data only**. No future data leakage.

Example:
```python
# Safe: uses close[t-20:t]
df['sma_20'] = df['close'].rolling(20).mean()

# Unsafe (leakage): would use forward data
# df['future_return'] = df['close'].shift(-1)  ← NEVER do this in features
```

#### Context Features (Market, Sector, Sentiment)

```python
# Optional: market and sector context
market_proxy = load_market_proxy()  # VNINDEX-like proxy
sector_proxies = load_sector_proxies(tickers, market_data)
sentiment = load_sentiment(tickers, dates)

# Combine into features_df
features_df = apply_context_features(
    features_df,
    market_proxy=market_proxy,
    sector_proxies=sector_proxies,
    sentiment_features=sentiment,
)
```

**Context columns added**:
- `m_ret`: Market return (lag 1 day)
- `m_ret_5d`: Market return (lag 5 days)
- `rel_to_market`: Stock return relative to market
- `s_ret`: Sector return (lag 1 day)
- `s_ret_5d`: Sector return (lag 5 days)
- `rel_to_sector`: Stock return relative to sector

#### Risk Features (VaR, CVaR, Drawdown)

```python
risk_engine = RiskEngine()
risk_features = risk_engine.compute_rolling_risk(
    ohlcv_df,
    lookback_days=20,
    confidence_level=0.95
)
```

**Risk columns added**:
- `var_q`: Value-at-Risk at 95% confidence
- `cvar_q`: Conditional VaR (Expected Shortfall)
- `covar_q`: Covariance-based risk
- `delta_covar`: Change in covariance (crisis indicator)
- `rolling_drawdown`: Max underwater excursion in lookback window

**Time-safe**: Computed from historical returns only.

### 3. Target Generation (Label Engineering)

#### Regression Branch: Forward Return

```python
# src/ml/labels/regression.py

class RegNextCloseReturn(BaseLabelGenerator):
    """1-day forward close-to-close log return."""
    def _compute(self, df):
        close = df['close']
        df['target_reg_return'] = close.shift(-1) / close - 1
        return df

class RegMultiHorizonReturn(BaseLabelGenerator):
    """Multi-horizon forward return for 3d, 5d, 20d."""
    def _compute(self, df, horizon=5):
        close = df['close']
        df[f'target_reg_{horizon}d_return'] = close.shift(-horizon) / close - 1
        return df
```

**Time-safety**:
```python
# To avoid leakage: access ONLY future prices via shift(-horizon)
# close.shift(-1) looks one day FORWARD (safe)
# close.shift(1) looks one day BACKWARD (safe for training)
# close.iloc[i+1] would be tempting to use but LEAKS information
```

**Result**: Continuous target values (returns can be positive or negative).

**Last horizon rows are NaN**: Cannot predict forward return beyond last observation.

#### Classification Branch: Profit/Loss Label

```python
# src/ml/labels/classification.py

class ProfitLossLabel(BaseLabelGenerator):
    """Classify as profit (1) or loss (0) after transaction costs."""
    def _compute(self, df, horizon=5, transaction_fee_bps=15, slippage_bps=20):
        close = df['close']
        
        # Forward price after horizon days
        future_price = close.shift(-horizon)
        
        # Total transaction cost
        cost = (transaction_fee_bps + slippage_bps) / 10000  # Convert to percentage
        
        # Profitable if: future_price / current_price > (1 + cost)
        profit_label = (future_price / close) > (1 + cost)
        
        df['profit_label'] = profit_label.astype(int)
        return df
```

**Key insight**: Profitable means the stock move beats transaction costs.

**Imbalance**: Expect ~40-50% positive class (depends on volatility and cost assumptions).

**Time-safety**: Same `shift(-horizon)` principle; only uses forward price, not future profits.

### 4. Data Splitting (Time-Aware)

```python
# src/ml/trainer.py - DualModelTrainer

split = SplitDefinition(
    train_stop=1000,      # Train on rows 0:1000
    val_start=1000,       # Validation rows 1000:1200
    val_stop=1200,
    test_start=1200,      # Test rows 1200:end
    gap=5,                # Gap between train/val/test (avoid lookahead)
)

train_x = features[split.train_stop]
train_y = targets[:split.train_stop]

val_x = features[split.val_start:split.val_stop]
val_y = targets[split.val_start:split.val_stop]

test_x = features[split.test_start:]
test_y = targets[split.test_start:]
```

**Time-aware principle**:
- No shuffling: rows maintain temporal order
- Contiguous blocks: train → validation → test
- Gap: `gap=5` means skip 5 rows between train/val to avoid temporal correlation
- Last rows are NaN targets (horizon rows): automatically excluded from eval

### 5. Sequence Dataset (for LSTM/BiLSTM)

```python
# src/ml/sequence_dataset.py

sequence_dataset = create_sequence_dataset(
    features_df,
    targets,
    sequence_length=20,   # Look back 20 days
    horizon_days=5,       # Predict 5 days ahead
    stride=1,             # No overlap (conservative)
)
```

**Sequence structure**:
```
Sequence_i = (X[t-20:t], y[t+5])
                 20 days  target at day t+5
```

**Time-safe**:
- Lookback window [t-20, t) uses only past data
- Target y[t+5] is future return/label
- No overlap between train/val/test sequences

**Why sequences?**:
- LSTM needs temporal context
- Captures multi-day trends
- More data points per ticker

---

## Model Training: DualModelTrainer

### Overview

```python
class DualModelTrainer:
    """Train both regression and classification models in one orchestration."""
    
    def train(
        self,
        ticker: str,
        features_df: pd.DataFrame,
        config: TrainConfig,
    ) -> dict[str, Any]:
        # Compute targets for both branches
        regression_target = compute_regression_target(features_df, horizon)
        classification_target = compute_classification_target(features_df, horizon)
        
        # Split data into train/val/test
        split = self._compute_split(features_df)
        
        # Clone data for each algorithm
        for algorithm in ["cart", "xgboost", "lightgbm", "sarimax", "ets", "lstm"]:
            if algorithm in SEQUENCE_ALGORITHMS:
                # Create sequences for LSTM
                dataset = create_sequence_dataset(features, targets, seq_len=20)
                model = train_lstm(dataset, config)
            else:
                # Flat data for tree/statistical models
                model = create_model(algorithm)
                model.fit(features[split.train], targets[split.train])
            
            # Save trained model manifest
            save_artifact(model, ticker, algorithm)
```

### Algorithm-Specific Training

#### Tree Models (CART, XGBoost, LightGBM)

```python
from src.ml.models.factory import create_model

# Flat tabular data
model = create_model("xgboost", max_depth=5, learning_rate=0.1)
model.fit(
    X_train,  # shape: (n_samples, n_features)
    y_train,  # shape: (n_samples,)
)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]  # For classification
```

**Hyperparameters**:
- `max_depth`: tree depth (default: None arbitrary)
- `learning_rate`: shrinkage (default: 0.1 for boosting)
- `num_trees`: number of iterations (default: 100)

#### Statistical Models (SARIMAX, ETS)

```python
from src.ml.models.sarimax import SARIMAXModel
from src.ml.models.ets import ETSModel

# Time series input: 1D array of values
sarimax = SARIMAXModel(order=(1,1,1), seasonal_order=(1,1,1,20))
sarimax.fit(y_train)  # y_train: 1D array
forecast = sarimax.predict(steps=len(y_test))

ets = ETSModel()
ets.fit(y_train)
forecast = ets.predict(steps=len(y_test))
```

**Note**: SARIMAX/ETS ignore features (X); use only target series history.

#### Deep Learning (LSTM, BiLSTM)

```python
from src.ml.models.lstm import LSTMModel

lstm = LSTMModel(
    sequence_length=20,
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    learning_rate=1e-3,
    epochs=30,
    patience=5,  # Early stopping
)

# Sequences: shape (n_seqs, 20, n_features)
lstm.fit(X_train_sequences, y_train_sequences)
predictions = lstm.predict(X_test_sequences)
```

**Early stopping**: Stops training if val loss doesn't improve for `patience` epochs.

### Artifact Manifest

Each trained model gets a JSON manifest:

```json
{
  "schema_version": 1,
  "ticker": "ACB",
  "algorithm": "xgboost",
  "model_family": "boosting",
  "feature_columns": ["sma_20", "rsi", "atr", "m_ret", ...],
  "target_type": "forward_return",
  "horizon_days": 5,
  "train_period": "2020-01-01 to 2023-12-31",
  "validation_method": "time-aware_split",
  "training_statistics": {
    "n_train": 900,
    "n_val": 150,
    "n_test": 200
  },
  "hyperparameters": {
    "max_depth": 5,
    "learning_rate": 0.1
  },
  "model_path": "models/ACB/xgboost_5d.pkl"
}
```

**Purpose**: Enables reproducible inference without retraining.

---

## Inference & Evaluation

### Daily Inference Loop

```python
class RealDataBacktestRunner:
    def _run_daily_inference(self, eval_date, ticker):
        # Load features up to eval_date
        features = load_features(ticker, up_to=eval_date)
        
        # Load all trained models for this ticker
        models = load_ticker_models(ticker, self.model_dir)
        
        # For each model
        for model_name, model in models.items():
            # Predict forward return
            pred_return = model.predict(features.iloc[-1:])  # Latest row only
            
            # Get classification probability
            if hasattr(model, 'predict_proba'):
                pred_prob_profit = model.predict_proba(features.iloc[-1:])[:, 1]
            else:
                pred_prob_profit = np.nan
            
            # Append to result
            results.append({
                'date': eval_date,
                'ticker': ticker,
                'model': model_name,
                'predicted_return': pred_return,
                'predicted_profit_probability': pred_prob_profit,
            })
```

### Evaluation Metrics

#### Regression Metrics

```python
def _compute_error_metrics(actual, predicted):
    errors = predicted - actual
    abs_errors = errors.abs()
    
    return {
        'mae': abs_errors.mean(),                    # Mean absolute error
        'rmse': np.sqrt((errors ** 2).mean()),      # Root mean squared error
        'mape': (abs_errors / actual.abs()).mean(),  # Mean absolute percentage error
        'directional_accuracy': (sign(predicted) == sign(actual)).mean(),
    }
```

**Why directional accuracy?** Traders care more about "did the price go up?" than exact magnitude.

#### Classification Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'precision': precision_score(y_true, y_pred),  # % predicted positives that were correct
    'recall': recall_score(y_true, y_pred),        # % actual positives found
    'f1': f1_score(y_true, y_pred),                # Harmonic mean of precision/recall
    'roc_auc': roc_auc_score(y_true, y_prob),      # AUC-ROC curve
}
```

---

## Regime Detection Layer

### Regime Types

```python
# src/ml/regime/regime_detector.py

class RegimeDetector:
    """Rule-based regime detection: NORMAL, HIGH_VOL, CRISIS."""
    
    def detect(self, volatility, drawdown, delta_covar=None):
        # CRISIS: large drawdown OR spike in correlation
        crisis = (drawdown <= -0.12) | (delta_covar >= 0.015)
        
        # HIGH_VOL: elevated volatility but not crisis
        high_vol = (~crisis) & (volatility >= 0.03)
        
        # NORMAL: baseline
        normal = ~(crisis | high_vol)
        
        return {
            'labels': pd.Series(['NORMAL', 'HIGH_VOL', 'CRISIS']),
            'probabilities': pd.DataFrame({
                'NORMAL': [prob_n, ...],
                'HIGH_VOL': [prob_hv, ...],
                'CRISIS': [prob_c, ...],
            }),
        }
```

**Interpretation for strategy**:
- **NORMAL**: Sideways, mean-reverting behavior; shorting can work
- **HIGH_VOL**: Trending, momentum works; stationary models fail
- **CRISIS**: Panic, correlations spike; diversification breaks

### Regime-Conditioned Evaluation

```python
# Split results by regime
bull_results = results[results['regime'] == 'NORMAL']  # or detect via momentum threshold
bear_results = results[results['regime'] == 'CRISIS']
sideways_results = results[results['regime'] == 'HIGH_VOL']

# Compute metrics per regime
bull_metrics = compute_metrics(bull_results)
bear_metrics = compute_metrics(bear_results)
# ...

# Summary: which models work in which regime?
print(f"XGBoost wins in bull regime: {bull_metrics['xgboost']['mae']}")
print(f"SARIMAX wins in sideways: {sideways_metrics['sarimax']['mae']}")
```

---

## Combined Signal Fusion

### Two-Signal Consensus

```python
# src/ml/backtest/combined_signal.py

def _fused_score(regression_score, classification_prob, w_return=0.5, w_profit=0.5):
    """Combine return signal and profit signal into single score."""
    
    # Normalize both signals to [0, 1]
    norm_return = (regression_score - min_return) / (max_return - min_return)
    norm_profit = classification_prob  # Already in [0, 1]
    
    # Weighted average
    combined = w_return * norm_return + w_profit * norm_profit
    
    return combined
```

**Intuition**: A stock that BOTH has high expected return AND high profit probability is more credible.

### Threshold-Based Ranking

```python
# Apply thresholds
return_threshold = 0.01      # 1% expected return
probability_threshold = 0.60 # 60% profit probability

# Classify predictions
return_signal = 'strong' if pred_return > return_threshold else 'weak'
profit_signal = 'strong' if pred_prob > probability_threshold else 'weak'

# Joint signal
signal = f"{return_signal}_{profit_signal}"  # e.g., "strong_strong", "weak_strong"
```

---

## Walk-Forward Temporal Architecture

### Fold Generation

```python
# src/ml/backtest/walk_forward_regime_robustness.py

def _generate_folds(self):
    folds = []
    for fold_number in range(1, max_folds + 1):
        # Expanding window: all data from start to eval_end
        if mode == 'expanding':
            fold_train_start = train_start
        # Rolling window: last N days
        elif mode == 'rolling':
            fold_train_start = eval_start - timedelta(days=rolling_train_window_days)
        
        fold_train_end = eval_start - timedelta(days=1)
        fold_eval_end = eval_start + timedelta(days=eval_window_days - 1)
        
        folds.append({
            'fold': fold_number,
            'train_period': (fold_train_start, fold_train_end),
            'eval_period': (eval_start, fold_eval_end),
        })
        
        eval_start += timedelta(days=step_size_days)  # Advance to next fold
    
    return folds
```

**Example timeline** (4 folds, 60-day eval, 30-day step):
```
Fold 1: Train 2020-01-01 to 2023-01-01 | Eval 2023-01-01 to 2023-02-29
Fold 2: Train 2020-01-01 to 2023-01-31 | Eval 2023-01-31 to 2023-03-30
Fold 3: Train 2020-01-01 to 2023-02-28 | Eval 2023-02-28 to 2023-04-28
Fold 4: Train 2020-01-01 to 2023-03-30 | Eval 2023-03-30 to 2023-05-29
```

### Aggregation Across Folds

```python
# Combine fold results
cross_fold_results = pd.concat([
    fold_1_results,
    fold_2_results,
    fold_3_results,
    fold_4_results,
])

# Compute stability metrics
win_rate_per_model = cross_fold_results.groupby('model').apply(
    lambda df: (df['mae'] < baseline_mae).mean()  # Win if better than baseline
)

# Stability level
stability = 'high' if win_rate >= 0.70 else ('medium' if win_rate >= 0.45 else 'low')
```

---

## Meta-Selector Logic

### Candidate Scoring

```python
# src/ml/backtest/meta_selector.py

def _select_best_candidate(regime, walk_forward_results):
    """Pick best model-horizon-threshold for given regime."""
    
    # Filter to this regime
    regime_results = results[results['regime'] == regime]
    
    # Compute utility score
    candidates = []
    for candidate in regime_results.groupby(['model', 'horizon', 'thresholds']):
        utility = (
            0.40 * normalize(candidate['topk_avg_return']) +
            0.30 * normalize(candidate['topk_profit_rate']) +
            0.20 * normalize(candidate['positive_class_precision']) +
            0.10 * normalize(candidate['directional_accuracy'])
        )
        candidates.append({
            'label': candidate_id,
            'utility_score': utility,
            'recommendation_rank': rank,
        })
    
    return sorted(candidates, key=lambda x: x['utility_score'], reverse=True)
```

---

## Time-Safety Invariants

**Core principle**: Features and targets NEVER use future data.

### Safe Patterns
```python
# Computing features for date t:
sma_20[t] = mean(close[t-20:t])           # ✅ Safe: uses past data
return[t] = close[t] / close[t-1] - 1     # ✅ Safe: historical return
```

### Unsafe Patterns (Do NOT use)
```python
# Computing features for date t:
sma_20[t] = mean(close[t:t+20])           # ❌ Unsafe: uses future data (leakage!)
next_return[t] = close[t+1] / close[t]    # ❌ Unsafe: uses next day (leakage!)
```

### Targets (Forward-looking, but time-safe)
```python
# Target for date t:
forward_return_5d[t] = close[t+5] / close[t] - 1  # ✅ Safe: using shift(-5)
```

**Why safe?** Because `shift(-5)` in pandas correctly accesses rows AFTER row t, which is "future" from training perspective, but is known data when making predictions at time t.

---

## Key Dependencies

**Core**:
- `pandas`, `numpy`: data manipulation
- `scikit-learn`: tree models, metrics
- `xgboost`, `lightgbm`: boosting

**Optional**:
- `statsmodels`: SARIMAX, ETS
- `tensorflow`/`torch`: LSTM, BiLSTM
- `vnstock`: Vietnamese market data

**Graceful degradation**: If optional lib missing, that algorithm is skipped with warning.

---

## Performance Characteristics

| Aspect | Notes |
|--------|-------|
| Feature engineering | ~10ms per ticker per day |
| Model training (1 ticker, 5 algorithms) | ~5-30 seconds (depends on data size) |
| Daily inference (100 tickers) | ~100ms |
| Walk-forward (4 folds) | ~hours (full pipeline, all tickers, all algorithms) |
| Meta-selector (cross-fold aggregation) | ~seconds |

---

## Related Documentation

- [EVALUATION_WORKFLOWS.md](./EVALUATION_WORKFLOWS.md) — Each evaluation method
- [USAGE_GUIDE.md](./USAGE_GUIDE.md) — Running the workflows
- [RESEARCH_FINDINGS_AND_LIMITATIONS.md](./RESEARCH_FINDINGS_AND_LIMITATIONS.md) — Results
