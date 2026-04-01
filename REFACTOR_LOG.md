# 🔄 Refactor & Cleanup Log

## 1. MOVED / RENAMED FILES
- (None in current audit phase; only identified for P0 improvement.)

## 2. ARCHIVED / CLEANUP CANDIDATES
- `scripts/準備_fundamental_features.py` (Identified for archiving).
- `scripts/prepare_vip_list.py` (Identified for archiving).
- `scripts/import_historical_csv.py` (Redundant with `sync_all_data.py`).
- Non-VN100 subdirectories in `models/` (Identified for archiving).

## 3. CONSOLIDATED UTILITIES
- `src/ml/feature_engineering.py` (Identified for critical update to `close_raw` logic).
- `src/ml/labels/*.py` (Identified for target consistency update).

## 4. CONSOLIDATED MODULES
- All ingestion logic should be centralized in `src/data/adapters/` and `src/data/historical/`.

---
*This log tracks the structural evolution of the repository to ensure traceability and maintainability.*
