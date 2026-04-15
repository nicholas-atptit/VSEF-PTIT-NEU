# Changelog Summary: Evolution of the Stock Forecasting Framework

This document chronicles the major changes that transformed the system from a basic statistical forecasting pipeline into a comprehensive research-grade evaluation framework.

---

## Phase 1: Real-Data Integration (vnstock Adapter)

**When**: Foundation layer  
**What changed**: 
- Integrated vnstock library to fetch Vietnamese market OHLCV data directly
- Implemented `VnstockAdapter` for transparent symbol and date range handling
- Moved away from synthetic/sample data to real VN100 universe

**Why it mattered**:
- Enabled real-world backtesting with actual execution costs and market conditions
- Supported daily incremental data syncing for production-like workflows
- Allowed evaluation on true Vietnamese equity market behavior

**Key files**:
- `src/data/adapters/vnstock_adapter.py`
- `src/ml/data_loader.py`

---

## Phase 2: Fixed-Window Real-Data Backtest

**When**: Early evaluation framework  
**What changed**:
- Introduced `FixedWindowBacktestConfig` and `RealDataBacktestRunner`
- Train once on a fixed historical window (e.g., 2020-2023)
- Evaluate daily on a holdout window (e.g., 2023-2026)
- Supports multiple tickers simultaneously

**Why it mattered**:
- Baseline evaluation method: simple, interpretable, avoids train leakage
- Single train phase reduces computational cost
- Clear separation between model selection (train) and deployment (eval)

**Key files**:
- `src/ml/backtest/real_data.py`
- `scripts/run_backtest_real_data.py`

**Artifacts**:
- `artifacts/backtest/models/` — trained model manifests per ticker
- `artifacts/backtest/predicted_vs_actual.csv` — daily predictions

---

## Phase 3: Multi-Model Comparison

**When**: Model family evaluation  
**What changed**:
- Added support for multiple model families: CART, XGBoost, LightGBM, SARIMAX, ETS
- Implemented configurable model factory (`src/ml/models/factory.py`)
- Automatic algorithm switching and graceful degradation if optional libs unavailable

**Why it mattered**:
- Enables systematic model comparison without code duplication
- Supports research question: which model family is most stable?
- Foundational for ensemble and selection logic downstream

**Key files**:
- `src/ml/models/factory.py`
- `src/ml/backtest/model_comparison.py`

---

## Phase 4: Forward-Return Multi-Horizon Regression

**When**: Extended target definitions  
**What changed**:
- Introduced `ForwardReturnBacktestConfig` with configurable prediction horizons
- Support for 3-day, 5-day, and 20-day forward returns
- Time-safe targets: use `shift(-horizon)` to avoid leakage
- Error metrics: MAE, RMSE, MAPE, directional accuracy

**Why it mattered**:
- Different investors care about different time horizons
- Enables assessment of "can we predict short-term moves?" vs. "longer-term trends?"
- Directional accuracy metric captures what traders care about most

**Key files**:
- `src/ml/labels/regression.py` — label generators with time-safe horizon shifts
- `src/ml/backtest/forward_return.py` — multi-horizon backtest infrastructure
- `scripts/run_backtest_forward_return.py`

**Artifacts**:
- `artifacts/backtest_forward_return/{3d,5d,20d}/` — per-horizon results

---

## Phase 5: Dual-Task Architecture (Regression + Classification)

**When**: Dual objective learning  
**What changed**:
- Added classification branch: predict profit/loss after transaction costs
- `DualModelTrainer` trains both regression (forward return) and classification (profit yes/no) heads
- `DualTaskBacktestRunner` evaluates both tasks in parallel
- Configurable transaction fee and slippage simulation

**Why it mattered**:
- Return prediction ≠ Profitable trade (accounting for costs)
- Profit classification directly answers: "Will this position be profitable?"
- Allows separate evaluation of predictions vs. real-world tradability

**Key files**:
- `src/ml/labels/classification.py` — profit/loss target generators
- `src/ml/labels/volatility.py` — transaction cost helpers
- `src/ml/backtest/dual_task.py` — parallel evaluation
- `src/ml/trainer.py` — `DualModelTrainer` orchestrates both tasks
- `scripts/run_dual_task_backtest.py`

**Artifacts**:
- `artifacts/dual_task/regression/{horizon}/` — regression metrics per horizon
- `artifacts/dual_task/classification/{horizon}/` — classification metrics per horizon

---

## Phase 6: Combined Decision-Support Signal

**When**: Multi-signal fusion  
**What changed**:
- Introduced `CombinedSignalAnalysisRunner` to merge regression and classification outputs
- Combines `predicted_return` and `predicted_profit_probability` via configurable weights
- Applies return/probability thresholds to generate rank scores
- Different ranking methods: simple, weighted linear, gated thresholds

**Why it mattered**:
- Single predictions can be noisy; combining reduce false positives
- Consensus between "will it go up?" and "will it be profitable?" increases confidence
- Thresholding converts soft predictions into alpha-seeking signals

**Key files**:
- `src/ml/backtest/combined_signal.py`
- `scripts/run_combined_signal_analysis.py`

**Artifacts**:
- `artifacts/combined_signal/{horizon}/combined_signal_table.csv`

---

## Phase 7: Market Regime Detection

**When**: Conditional forecasting  
**What changed**:
- Added `RegimeDetector`: rule-based bull/bear/sideway classification
- Uses rolling volatility and drawdown thresholds
- Outputs probability scores in addition to hard labels

**Why it mattered**:
- Market behavior differs across regimes (trending vs. mean-reverting)
- Different models may excel in different regimes
- Enables regime-conditioned strategy and evaluation

**Key files**:
- `src/ml/regime/regime_detector.py`

**Regime definitions**:
- **NORMAL** (Sideway): volatility < threshold, no recent heavy drawdown
- **HIGH_VOL** (Bull/Bear trending): elevated volatility but no crisis peak
- **CRISIS**: large drawdown OR spike in covariance

---

## Phase 8: Regime-Aware Evaluation

**When**: Conditional performance analysis  
**What changed**:
- `RegimeAwareAnalysisRunner` attaches market regime labels to evaluation outputs
- Summarizes metrics per regime: bull, bear, sideway
- Highlights which models/horizons work best in which market conditions

**Why it mattered**:
- Global metrics mask regime-dependent behavior
- Separates "method is always good" from "method is good in trending markets"
- Informs strategy risk management

**Key files**:
- `src/ml/backtest/regime_aware_analysis.py`
- `scripts/run_regime_aware_analysis.py`

**Artifacts**:
- `artifacts/regime_aware_analysis/summary/{horizon}/` — per-regime breakdowns

---

## Phase 9: Walk-Forward Regime-Aware Robustness

**When**: Temporal stability assessment  
**What changed**:
- Introduced `WalkForwardRegimeRobustnessRunner` with expanding and rolling train windows
- Multiple folds (e.g., 4 folds over 3+ years)
- Each fold: train on prior data, eval on subsequent window
- Aggregates results across folds to measure method stability

**Why it mattered**:
- Fixed-window backtest is susceptible to specific historical period bias
- Walk-forward insulates against "overfitting" the test window
- Rolling windows show if method degrades with time
- Stability (win rate across folds) is more honest than peak performance

**Key files**:
- `src/ml/backtest/walk_forward_regime_robustness.py`
- `scripts/run_walk_forward_regime_robustness.py`

**Configuration options**:
- Expanding mode: all prior data vs. recent window
- Rolling mode: fixed-size sliding train window
- Configurable fold size, step size, max folds

**Artifacts**:
- `artifacts/walk_forward_regime_robustness/summary/` — cross-fold aggregates

---

## Phase 10: Regime-Conditioned Meta-Selector

**When**: Automatic best-candidate selection  
**What changed**:
- `MetaSelectorConfig` and `MetaSelectorRunner` consolidate walk-forward results
- Per regime (bull/bear/sideway), select best model-horizon-threshold combination
- Multiple selector modes:
  - `simple_regime_lookup`: pick top candidate per regime
  - `regime_weighted_rank`: blend across regime-confidence scores
  - `fallback_global`: use global winner if regime-specific data sparse

**Why it mattered**:
- Answers: "Given the walk-forward results, what should I actually deploy?"
- Accommodates regimes with sparse training data (fallback logic)
- Enables data-driven strategy selection without manual tuning

**Key files**:
- `src/ml/backtest/meta_selector.py`
- `scripts/run_meta_selector.py`

**Artifacts**:
- `artifacts/meta_selector/candidates.csv` — ranked candidates per regime
- Selector mode configurations and recommendations

---

## Evolution Summary

| Phase | Component | Purpose | Status |
|-------|-----------|---------|--------|
| 1 | vnstock adapter | Real data source | ✅ Stable |
| 2 | Fixed-window backtest | Baseline evaluation | ✅ Stable |
| 3 | Multi-model comparison | Algorithm selection | ✅ Stable |
| 4 | Multi-horizon regression | Temporal flexibility | ✅ Stable |
| 5 | Dual-task architecture | Cost-aware prediction | ✅ Stable |
| 6 | Combined signal | Signal fusion | ✅ Stable |
| 7 | Regime detection | Market condition sensing | ✅ Stable |
| 8 | Regime-aware analysis | Conditional evaluation | ✅ Stable |
| 9 | Walk-forward robustness | Temporal stability | ✅ Stable |
| 10 | Meta-selector | Automatic selection | ✅ Stable |

---

## Key Insights from Evolution

### Architectural Lessons
1. **Time-safety is non-negotiable**: Every phase enforces `shift(-horizon)` discipline
2. **Dual tasks are more informative than single task**: Knowing "will I profit?" beats "will it go up?"
3. **Regime conditioning exposes instability**: Many models thrive in trending markets, fail sideways
4. **Walk-forward is humbling**: Methods that win in one period often break in the next
5. **Stability beats peak performance**: A method that wins 3 of 4 folds is more useful than one that crushes fold 2

### Research Redirection
- Initial focus: "Which algorithm is best?" → Found: families matter more than exact config
- Mid focus: "Can we predict returns?" → Found: Yes, but profit margin is small
- Current focus: "Can we be stable across regimes?" → Found: Stability is low; need more data or better features

### Future Work
- Benchmark audit: need gold-standard labels for model calibration
- Context-conditioned meta-selector: incorporate market microstructure indicators
- Ensemble stacking: combine regime predictions for higher confidence
- Feature engineering: expand beyond simple technical indicators

---

## Related Documentation

- [USAGE_GUIDE.md](./USAGE_GUIDE.md) — How to run each workflow
- [ML_IMPLEMENTATION_GUIDE.md](./ML_IMPLEMENTATION_GUIDE.md) — Internal architecture
- [EVALUATION_WORKFLOWS.md](./EVALUATION_WORKFLOWS.md) — Each workflow in detail
