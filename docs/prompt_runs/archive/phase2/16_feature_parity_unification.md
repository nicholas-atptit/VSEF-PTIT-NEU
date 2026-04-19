# 16. Feature Parity Unification for VN100 Prediction System

### Prompt Intent
Achieve absolute parity between the training and inference feature engineering pipelines. Resolve critical drifts in technical indicators (Kalman filter, RSI, ROC) and unify the order of operations for context features (Market/Sector).

### Actual Outcome
1.  **Unified Pipeline**: Consolidated all technical indicator logic, Kalman filtering, and `d_` prefix delta transformations into a single `FeatureEngineer.transform` method.
2.  **Synchronized Trainings**: Updated `scripts/train_ml_tickers.py` to use the unified `FeatureEngineer` and ensured feature calculation occurs on **FULL history** before any date-based slicing.
3.  **Synchronized Inference**: Updated `DualModelTrainer.compute_features_for_ticker` in `src/ml/trainer.py` to use the identical `FeatureEngineer` logic without redundant/incorrect raw-copy logic.
4.  **Stable Lookback**: Refactored `VN100DataLoader.build_inference_dataset` in `src/ml/data_loader.py` to load full historical data from CSV, providing a sufficient buffer for long-window rolling features.
5.  **Verified Success**: Enhanced `scripts/debug_compare_train_infer.py` to perform date-aligned, type-normalized numeric comparisons. Final verification on ticker `FPT` confirmed **100% agreement** on all contract-required features.

### Files Created
- `docs/prompt_runs/16_feature_parity_unification.md` (This file)

### Files Modified
- `src/ml/feature_engineering.py`: Complete unification of features, Kalman filtering, and deltas.
- `src/ml/trainer.py`: Simplified inference path to match training feature logic.
- `src/ml/data_loader.py`: Enabled full history loading for inference stability.
- `scripts/train_ml_tickers.py`: Rewritten to simplify feature generation and stabilize date filtering.
- `scripts/debug_compare_train_infer.py`: Refactored into a high-fidelity parity validation tool.

### Key Code Changes
- **FeatureEngineer**: Integrated `_add_delta_features` which calculates `val - val.shift(1)` for all technical columns.
- **Data Stability**: Moved date filtering in training script to occur **after** `build_daily_features` to prevent Kalman initialization drift.
- **Normalization**: Enforced `pd.to_datetime().dt.date` consistency across all data ingestion and comparison points.

### Algorithms / Methods / Logic Introduced
- Implemented: Unified Kalman filter initialization on full historical data.
- Implemented: Canonical `d_` prefix delta logic (session-to-session change) for all non-metadata features.
- Implemented: Inference-safe `drop_na=False` mode for feature transformation that utilizes forward-filling instead of row dropping.

### Config / CLI / Environment Changes
- No new environment variables.
- `scripts/train_ml_tickers.py` now produces more stable models by calculating features on full history even when `--start-date` is provided.

### Storage / Persistence Impact
- Models trained with the updated pipeline will have a consistent feature contract encompassing both raw and delta (`d_`) features.

### Compatibility Notes
- Fully backward compatible with the previous `models/` structure (as long as `feature_cols.joblib` exists).

### Tests / Validation
- Run parity validation: `python scripts/debug_compare_train_infer.py --ticker FPT --join-market --join-sectors`.
- Confirmed zero numeric drift for FPT on contract features (2025-09-24 alignment).

### Remaining Gaps
- None identified for feature parity. Pipeline is now technicaly synchronized.
