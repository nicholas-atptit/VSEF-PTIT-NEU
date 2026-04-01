# 📈 Technical Prediction System Audit Report (Master)

## 1. EXECUTIVE SUMMARY

The current repository represents a highly sophisticated, multi-agent hybrid stock prediction system (v5.3.1). It successfully integrates quantitative alpha factors (technical) with qualitative sentiment analysis (LLM).

**Strengths:**
- **Robust Feature Set**: Includes state-of-the-art volatility estimators (Yang-Zhang).
- **Advanced Training**: Implements purging, embargoing, and exponential time-weighting.
- **Event-Driven Simulation**: Reality-based paper trading with execution costs.

**Weaknesses:**
- **Label Leakage (Custom Mode)**: Risk of predicting smoothed prices instead of raw market movements.
- **Artifact Bloat**: 1600+ ticker directories in `models/` degrade maintainability.
- **Coordination Overlap**: Multiple ingestion scripts and hardcoded configurations.

**Top 5 Risks:**
1. **Critical**: Time-serving skew if Kalman smoothing is inconsistent.
2. **High**: Inflated accuracy in custom label modes due to price smoothing.
3. **High**: Disk I/O bottlenecks and data integrity risks with massive ticker-split CSVs.
4. **Medium**: High multi-collinearity in features (redundant volatility/momentum factors).
5. **Medium**: Lack of centralized data management (hardcoded paths).

---

## 2. CURRENT PROJECT MAP

### Folder Structure
- `scripts/`: Training, Inference, Ingestion, and Sentiment crawlers.
- `src/ml/`: Core logic (Features, Labels, Training, Backtest).
- `src/data/`: Adapters and Universe management.
- `data/`: Ingested prices (daily, hourly, raw, processed).
- `models/`: Saved model artifacts (.joblib).

### Main Entry Points
- Ingestion: `scripts/sync_all_data.py`
- Training: `scripts/train_ml_tickers.py`
- Inference: `scripts/per_session_predict.py`
- Sentiment: `scripts/run_news_crawler.py`

---

## 3. AUDIT FINDINGS BY CATEGORY

| ID | Severity | Area | File | Issue | Proposed Fix |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **F-01** | **Critical** | Features | `src/ml/feature_engineering.py` | Kalman filter overwrites `close` | Preserve `close_raw` and ensure it's used for all label generation. |
| **L-01** | **High** | Labels | `src/ml/labels/classification.py` | Potential smoothed price target | Explicitly pass `close_raw` to label generators. |
| **T-01** | **Medium** | Training | `scripts/train_ml_tickers.py` | Hardcoded parameters (Purge Gap, Horizons) | Move parameters to `config/settings.py`. |
| **B-01** | **Medium** | Backtest | `src/ml/backtest/paper.py` | Lack of benchmark comparison | Integrate `VNINDEX` relative performance tracking. |
| **D-01** | **High** | Data | `data/daily_market_split_data/` | Massive ticker-split CSVs (1600+) | Move to partitioned Parquet for efficiency. |
| **S-01** | **High** | Structure | `models/` | 1600+ ticker directories | Archive non-VN100 models and focus on a curated universe. |

---

## 4. TECHNICAL PREDICTION GAP ANALYSIS

**What is missing:**
- **Standardized Feature Registry**: To manage factor dependencies and avoid redundancy.
- **Walk-Forward Performance Monitoring**: Real-time evaluation of model drift.
- **Walk-Forward Retraining Simulation**: To capture regime shifts in backtests.

**What is redundant:**
- Overlapping volatility measures (HV, Parkinson, RS, YZ). 
- Multiple ingestion scripts (`import_historical_csv.py`, `fetch_fundamentals.py`).

---

## 5. REFACTOR PLAN THEO ƯU TIÊN

### Phase 1: Essential Validity (P0)
- **Fix Label Smoothing**: Ensure all targets use `close_raw`.
- **Sanitize Environment**: Centralize hardcoded paths and constants.
- **Archive Stale Models**: Move non-VN100 models to `archive/`.

### Phase 2: Maintainability (P1)
- **Implement Feature Registry**: Standardize indicator logic.
- **Data Infrastructure Upgrage**: Move from CSV split-files to Parquet.
- **Consolidate Ingestion**: Unified `sync_all_data.py` as the only source of truth.

### Phase 3: Performance & Scale (P2)
- **Enhanced Backtest**: Add benchmark-relative metrics.
- **Experiment Tracking**: Full integration of `ExperimentTracker` into all pipelines.

---

## 6. PROPOSED TARGET STRUCTURE

```
├── configs/                       # Centralized YAML/Python configs
├── data/                          
│   ├── raw/                       # Original downloads (unprocessed)
│   ├── staging/                   # Cleaned, standardized OHLCV (Parquet)
│   ├── features/                  # Computed feature matrices (Parquet)
├── src/
│   ├── data/                      # Ingestion, Adapters, Universe
│   ├── features/                  # Feature Registry, Technical Indicators
│   ├── labels/                    # Specialized Label Generators
│   ├── models/                    # Model Definitions & Training Logic
│   ├── training/                  # Orchestrated Training Pipelines
│   ├── inference/                 # Prediction Logic
│   └── backtest/                  # Event-driven and Vectorized Testing
├── artifacts/
│   ├── models/                    # Versions of trained .joblib files
│   ├── predictions/               # Prediction results (snapshot historical)
│   └── backtests/                 # Detailed performance artifacts
└── reports/                       
    └── audits/                    # Historical audit logs
```

---

## 7. QUICK WINS (10 Actions)

1. Centralize `PURGE_GAP` and `WINDOWS` in `settings.py`.
2. Move `close_raw` preservation to the top of `FeatureEngineer.transform`.
3. Create a `archive_models.py` script to prune the `models/` directory.
4. Add `is_trading_day` check to `per_session_predict.py` to prevent useless cycles.
5. Standardize on `VN100_PLUS_VIETTEL` universe across all scripts.
6. Unified log format for training and inference summaries.
7. Integrate `ExperimentTracker` into the custom label training loop.
8. Add a `--force-train` flag to overwrite specific models easily.
9. Export a monthly `universe_health.csv` showing data coverage per ticker.
10. Add `VNINDEX` as a mandatory comparison in `get_portfolio_summary`.

---

## 8. FILES TO CREATE

- `src/features/registry.py`: Manage factor computation order.
- `src/data/partitions.py`: Logic for Parquet-based partitioned data storage.
- `configs/ml_params.yaml`: Externalize all magic numbers from scripts.
- `scripts/maintenance_cleanup.py`: Automate pruning of stale models and data.

---

## 9. FILES TO MODIFY FIRST

1. `src/ml/feature_engineering.py`: Critical fix for Kalman/Close preservation.
2. `scripts/train_ml_tickers.py`: Consolidate settings and use unified universe.
3. `config/settings.py`: Add all ML and backtest constants.
4. `scripts/per_session_predict.py`: Align with `train_ml_tickers` data loading.
5. `src/ml/backtest/paper.py`: Add benchmark comparison and better cost modeling.

---

## 10. FINAL VERDICT

The current system has an **excellent foundation** but suffers from "Phase-by-Phase" complexity accumulation.
- **Current Maturity**: High (Technically), Medium (Maintainability).
- **Inference Readiness**: High.
- **Backtest Credibility**: Medium (Requires better cost/benchmark modeling).
- **Maintainability**: Low (Requires structural refactor).

**Next Step Recommendation**: Perform **Phase 1 (Essential Validity)** immediately to ensure prediction accuracy before further model tuning.
