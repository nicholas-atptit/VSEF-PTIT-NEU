# AI_ML_LLM-in-Stock: Regime-Aware Quantitative Stock Forecasting Framework

## Overview

This repository implements a comprehensive quantitative stock forecasting and evaluation framework for Vietnamese equities. It evolved from a basic statistical pipeline into a research-grade system supporting multi-horizon predictions, dual-task learning, regime-aware analysis, and walk-forward robustness testing.

**Current version**: Multi-model, multi-horizon, regime-conditioned framework with forward-return regression (3d/5d/20d) and profit/loss classification.

---

## Main Capabilities

### Forecasting Models
- **Tree-based**: CART, XGBoost, LightGBM
- **Statistical**: SARIMAX, ETS
- **Deep learning**: LSTM, BiLSTM (optional)
- **Ensemble**: Stacking

### Evaluation Architecture
- **Real-data backtesting** using vnstock market data (OHLCV)
- **Fixed-window backtest**: Train once, evaluate daily on holdout window
- **Multi-horizon predictions**: 3-day, 5-day, 20-day ahead returns
- **Dual-task learning**: 
  - Regression branch for forward return prediction
  - Classification branch for profit/loss after transaction costs
- **Combined signal analysis**: Fuses predicted return and profit probability
- **Regime detection**: Identifies bull/bear/sideway market conditions (volatility + drawdown)
- **Walk-forward robustness**: Expanding/rolling folds to assess method stability
- **Regime-conditioned meta-selection**: Picks best model-horizon-threshold combinations by market regime

### Time-Series Safety
All workflows use time-aware validation to avoid look-ahead bias. Future prices are accessed via `shift(-horizon)`.

---

## Quick Start

### Prerequisites
- Python 3.8+
- Dependencies: `pip install -r requirements.txt`
- Optional: vnstock data adapter (pre-configured)

### Basic Real-Data Backtest
```bash
cd h:\AI-ML-LLM in Stock_march26_PTIT_NEU
python -m scripts.run_backtest_real_data
```
Outputs to `artifacts/backtest/`

### Dual-Task Backtest (Regression + Classification)
```bash
python -m scripts.run_dual_task_backtest
```
Outputs to `artifacts/dual_task/`

### Full Workflow (Multi-Horizon + Combined Signal + Regime Analysis)
```bash
python -m scripts.run_backtest_forward_return          # 3d/5d/20d regression
python -m scripts.run_dual_task_backtest               # Add classification
python -m scripts.run_combined_signal_analysis         # Merge signals
python -m scripts.run_regime_aware_analysis            # Attach market regimes
python -m scripts.run_walk_forward_regime_robustness   # Stability assessment
python -m scripts.run_meta_selector                    # Best-by-regime selection
```

All artifacts saved in `artifacts/` directory with subdirectories per workflow.

---

## Main Workflows

| Workflow | Script | Purpose | Artifacts |
|----------|--------|---------|-----------|
| Real data backtest | `run_backtest_real_data.py` | Train on fixed window, evaluate daily | `backtest/models`, `backtest/predicted_vs_actual.csv` |
| Forward return | `run_backtest_forward_return.py` | Multi-horizon regression (3d/5d/20d) | `backtest_forward_return/{horizon}/` |
| Dual task | `run_dual_task_backtest.py` | Parallel regression + profit classification | `dual_task/{regression,classification}/{horizon}/` |
| Combined signal | `run_combined_signal_analysis.py` | Fuse return + profit probability | `combined_signal/{horizon}/combined_signal_table.csv` |
| Regime analysis | `run_regime_aware_analysis.py` | Attach bull/bear/sideway labels, summarize by regime | `regime_aware_analysis/summary/{horizon}/` |
| Walk-forward robustness | `run_walk_forward_regime_robustness.py` | Expanding/rolling folds, measure stability | `walk_forward_regime_robustness/summary/` |
| Meta selector | `run_meta_selector.py` | Pick best model+horizon per regime | `meta_selector/candidates.csv`, selector modes |

---

## Key Files & Organization

```text
src/ml/
├── backtest/                  # Evaluation workflows
│   ├── real_data.py          # Fixed-window Real Data Backtest
│   ├── forward_return.py      # Multi-horizon forward return
│   ├── dual_task.py           # Regression + classification together
│   ├── combined_signal.py     # Fuse return & profit signals
│   ├── regime_aware_analysis.py
│   ├── walk_forward_regime_robustness.py
│   ├── meta_selector.py       # Regime-conditioned candidate selection
│   └── model_comparison.py    # Compare model families
├── regime/
│   └── regime_detector.py     # Bull/bear/sideway detection
├── labels/
│   ├── regression.py          # Forward return targets
│   ├── classification.py       # Profit/loss targets
│   └── volatility.py          # Transaction cost helpers
├── trainer.py                 # Main training/inference facade
├── models/factory.py          # Model instantiation
├── data_loader.py             # Feature & data loading
├── features/                  # Feature engineering
├── accuracy/                  # Evaluation metrics
└── ...
```

---

## Documentation

For detailed guidance, see:

- **[CHANGELOG_SUMMARY.md](docs/CHANGELOG_SUMMARY.md)** — System evolution and major changes
- **[USAGE_GUIDE.md](docs/USAGE_GUIDE.md)** — How to run each workflow with examples
- **[ML_IMPLEMENTATION_GUIDE.md](docs/ML_IMPLEMENTATION_GUIDE.md)** — Internals: architecture, data flow, targets
- **[EVALUATION_WORKFLOWS.md](docs/EVALUATION_WORKFLOWS.md)** — Each evaluation method in detail
- **[RESEARCH_FINDINGS_AND_LIMITATIONS.md](docs/RESEARCH_FINDINGS_AND_LIMITATIONS.md)** — Empirical results and next steps

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

### Evaluation stack

1. **Real Data Backtest** → Fixed-window train/eval with vnstock OHLCV
2. **Forward Return Backtest** → Multi-horizon regression (3d/5d/20d)
3. **Dual Task** → Add classification branch for profit/loss
4. **Combined Signal** → Merge predicted return + profit probability
5. **Regime Analysis** → Attach market regime labels (bull/bear/sideway)
6. **Walk-Forward Robustness** → Expanding/rolling folds for stability
7. **Meta Selection** → Pick best candidates by regime

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
