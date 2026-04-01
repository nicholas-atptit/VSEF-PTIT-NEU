# Technical Prediction System Audit

## A. Feature Engineering Audit

### Inventory
- **File**: `src/ml/feature_engineering.py`
- **Indicator Count**: 80+
- **Key Factor Groups**:
    - **Volatility**: HV-20, Parkinson-20, Rogers-Satchell-20, Yang-Zhang-20.
    - **Momentum**: RSI, ROC, Stochastic, Williams %R.
    - **Structure**: Multi-horizon returns (5, 20, 60), Price Gaps, Volatility Ratios.
    - **Sentiment**: Integrated from `data/sentiment_features.csv`.
    - **Market/Sector**: Integrated from `data/market_proxy.csv`.

### Finding: Indicator Redundancy (Medium)
The repository implements four different volatility measures (HV, Parkinson, RS, YZ). While robust, these are highly collinear. Redundant indicators can increases training noise and reduce Optuna efficiency.
- **Recommendation**: Standardize on Yang-Zhang for open-gap sensitive volatility and HV for simpler drift.

### Finding: Alpha Factor Preprocessing (High)
The Kalman Smoothing applied to the `close` price (Line 310) is a powerful denoising technique, but it creates a secondary "synthetic close".
- **Risk**: If downstream features (e.g., RSI) are calculated on the smoothed `close` while the model is evaluated on raw prices, divergence may occur.

---

## B. Target / Label Audit

### Core Target Logic
- **File**: `scripts/train_ml_tickers.py` (Lines 96-100), `src/ml/labels/classification.py`.
- **Primary Target**: Binary "Price-at-T+H > Price-at-T".
- **Horizon**: 1w (5), 1m (20), 6m (120).

### Finding: Label Smoothing Risk (Critical)
In `train_ml_tickers.py`, the hardcoded targets use `c_raw` (raw price). However, `FeatureEngineer.transform()` overwrites `df['close']` with the Kalman-filtered price.
- **Problem**: If custom labels (via `--label-mode`) are used, they call `generator.generate(feat_df)`. Since `feat_df['close']` is the **filtered price**, the labels will reflect the movement of the **smoothed** price.
- **Impact**: Predictions become "artificially" accurate because smoothed paths are predictable. This accuracy will **fail in production** where entries and exits occur at raw market prices.
- **Recommendation**: Create a `labeling.py` or `feature_registry.py` that strictly isolates `close_raw` for all target/label generation.

---

## C. Training Pipeline Audit

### Observation: Advanced Anti-Overfitting
The v3 trainer in `scripts/train_ml_tickers.py` implements high-standard practices:
- **Purge Gap**: 3-day gap between train and test sets prevents leakage.
- **TimeSeriesSplit**: Optuna uses 3-fold cross-validation honoring time order.
- **Sample Weighting**: Exponential decay weights more recent data higher (Lines 309). This is excellent for non-stationary markets.
- **Stacking Ensemble**: Uses LGBM, XGB, RF, and ET. Meta-model used for "Elite" signal filtering.

### Finding: Hyperparameter Overhang (Low)
Optuna is set to `200` trials (Lines 52). While thorough, this is computationally expensive for 1600+ tickers and may lead to "p-hacking" the test set if the dataset is small (< 500 rows).

---

## D. Inference Pipeline Audit

### Implementation
- **File**: `scripts/per_session_predict.py`.
- **Logic**: Loads current prices via `VN100DataLoader`, computes features via `trainer.compute_features_for_ticker`, and runs inference on the last row.

### Finding: Train-Serving Consistency (High)
Inference uses `loader.build_inference_dataset` which might have different preprocessing than the training script.
- **Consistency Check**: `train_ml_tickers.py` uses `pd.read_csv`, while `per_session_predict.py` uses `VN100DataLoader`.
- **Recommendation**: Centralize data loading into a single robust class to ensure features and scaling are identical between `fit` and `predict`.
