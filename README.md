# AI_ML_LLM-in-Stock: Risk-Aware Quantitative Forecasting Pipeline (v1)

## Overview

This repository implements a manifest-driven quantitative forecasting pipeline for stock prediction.

The current version expands the original ML workflow into a broader architecture that supports:

- Statistical forecasting: **SARIMAX**, **ETS**
- Boosting models: **XGBoost**, **LightGBM**
- Optional hyperparameter tuning: **Optuna**
- Deep learning compatibility: **LSTM**, **BiLSTM**
- Conservative time-series-safe ensemble: **Stacking v1**
- Post-forecast risk evaluation: **Monte Carlo VaR / CVaR**

The system is designed to remain backward-compatible with the existing ML path while introducing structured risk-aware outputs and stronger model modularity.

---

## Key Features

### 1. Multi-branch forecasting architecture

The pipeline now supports multiple model families:

- **Tree / Tabular ML**: CART, XGBoost, LightGBM
- **Statistical models**: SARIMAX, ETS
- **Sequence models**: LSTM, BiLSTM
- **Ensemble model**: Stacking v1

### 2. Time-series-safe validation

All newly introduced tuning and ensemble logic use time-aware validation principles to avoid look-ahead bias.

### 3. Optional Optuna tuning

Boosting models can be tuned using Optuna without making Optuna a hard dependency for the whole project.

### 4. Risk-aware inference

A separate Monte Carlo risk module computes:

- **VaR (Value-at-Risk)**
- **CVaR / Expected Shortfall**
- Scenario summary statistics

This layer is intentionally kept outside the forecasting model factory.

### 5. Structured artifacts and manifests

Training and inference artifacts preserve:

- algorithm name
- model family
- tuning backend
- validation method
- stacking base learners
- risk configuration
- volatility proxy source
- risk assumptions

---

## Architecture Summary

### Forecasting branches

#### Statistical branch

- `sarimax`
- `ets`

#### Boosting branch

- `xgboost`
- `lightgbm`

#### Deep learning branch

- `lstm`
- `bilstm`

#### Ensemble branch

- `stacking`

### Risk layer

- `MonteCarloRiskSimulator`

The risk layer consumes forecast outputs and residual-based volatility proxies to generate VaR/CVaR without modifying the predictive registry.

---

## Project Structure

```text
src/ml/
├── models/
│   ├── sarimax.py
│   ├── ets.py
│   ├── xgboost_model.py
│   ├── lightgbm_model.py
│   ├── stacking.py
│   ├── base.py
│   └── factory.py
├── tuning.py
├── risk.py
├── trainer.py
└── artifacts.py

tests/ml/
├── test_statistical_models.py
├── test_boosting_models.py
├── test_tuning.py
├── test_deep_learning_models.py
├── test_stacking.py
├── test_risk.py
└── test_integration.py
```

---

## Supported Algorithms

Current supported algorithm names include:

- `cart`
- `sarimax`
- `ets`
- `xgboost`
- `lightgbm`
- `lstm`
- `bilstm`
- `stacking`

---

## Command-Line Usage

### Basic training

```bash
python scripts/train_ml_tickers.py --tickers FPT --algorithms "xgboost"
```

### Train boosting models with optional tuning

```bash
python scripts/train_ml_tickers.py --tickers FPT --algorithms "xgboost,lightgbm" --tune-boosters
```

### Enable stacking

```bash
python scripts/train_ml_tickers.py --tickers FPT --algorithms "cart,xgboost,lightgbm,sarimax,ets" --enable-stacking
```

### Enable risk-aware output

```bash
python scripts/train_ml_tickers.py \
  --tickers FPT \
  --algorithms "xgboost" \
  --enable-risk \
  --risk-simulations 5000 \
  --risk-confidence-levels 0.95,0.99 \
  --risk-seed 42
```

---

## Example Risk-Aware Output

```json
{
  "algorithm": "xgboost",
  "predicted_return": 0.0606,
  "trend_probabilities": {
    "up": 0.8166,
    "sideways": 0.0611,
    "down": 0.1223
  },
  "risk_assessment": {
    "var": {
      "95.0": -0.0154,
      "99.0": -0.0450
    },
    "cvar": {
      "95.0": -0.0336,
      "99.0": -0.0598
    },
    "summary": {
      "mean": 0.0597,
      "std": 0.0450
    },
    "metadata": {
      "risk_simulations": 5000,
      "volatility_proxy_source": "test_residuals_std",
      "assumptions": "Normal distribution of residuals around forecast mean"
    }
  }
}
```

---

## Testing

Run the ML test suite:

```bash
python -m pytest tests/ml/ -v
```

This suite covers:

- registry and capability contracts
- statistical wrappers
- boosting wrappers
- optional tuning behavior
- deep learning compatibility
- stacking stability
- Monte Carlo risk reproducibility
- integration behavior

---

## Current Limitations

- The risk layer assumes approximately normal residual behavior.
- The volatility proxy is derived from validation residuals and should be treated as an empirical fallback, not a perfect forward volatility estimate.
- Deep learning models are intentionally excluded from Stacking v1.
- CLI hyperparameter exposure is still intentionally conservative.

---

## Roadmap

Future versions may include:

- modularizing `src/ml/risk.py` into a dedicated risk package
- Student-t or historical simulation risk distributions
- GARCH-like volatility modeling
- portfolio sizing driven by VaR/CVaR
- deeper ensemble integration for sequence models

---

## Status

This repository currently provides a **risk-aware quantitative forecasting architecture v1** suitable for research, experimentation, and supervised technical evaluation.
