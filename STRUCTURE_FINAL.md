# 📂 Repository Structure (Final Design)

## 🏗️ SYSTEM ARCHITECTURE
This repository follows a Domain-Driven Design (DDD) for Quantitative ML, separated into clear layers: Data -> ML -> Engine -> API.

## 📁 CORE DIRECTORIES

| Folder | Responsibility |
| :--- | :--- |
| `src/data/` | Ingestion, Database Adapters, VN100 Universe management. |
| `src/ml/` | **Core Brain**. Features, Labels, Training, Prediction, and RAG. |
| `src/engine/` | Decision Logic (Matrix), Risk Management, Execution Rules. |
| `src/api/` | FastAPI routes, schemas, and health checks. |
| `scripts/` | **CLI Entry Points**. Operational scripts for sync, train, and inference. |
| `data/` | **Artifacts Layer**. Raw/Processed prices and temporary CSV splits. |
| `models/` | **Serialized Weights**. `.joblib` files organized by Ticker. |
| `reports/` | **Analytics**. Audit logs, daily briefs, and performance reports. |
| `docs/` | **Documentation**. Prompt runs, feature engineering notes, and design specs. |

## 🛠️ GUIDANCE FOR FUTURE CODE
1. **New Indicators**: Must be added to `src/ml/feature_engineering.py` (Unified Feature Layer).
2. **New Model Types**: Place definitions in `src/ml/training_pipeline/` and training logic in `scripts/train_ml_tickers.py`.
3. **Execution Rules**: Modify `src/engine/matrix.py` or `src/engine/risk.py`.
4. **Data Sync**: Always extend `scripts/sync_all_data.py` instead of creating new sync scripts.

---
*Created by Antigravity Audit Agent*
