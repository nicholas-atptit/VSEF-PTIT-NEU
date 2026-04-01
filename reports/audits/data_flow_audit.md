# Data Flow & Infrastructure Audit

## A. Data Ingestion Audit

### Source Integration
- **Source**: `vnstock`.
- **Adapter**: `src/data/adapters/vnstock_adapter.py`.
- **Script**: `scripts/sync_all_data.py`.
- **Logic**: Uses `BackdateIngestor` to pull OHLCV data from the `VCI` source.

### Finding: vnstock Dependency (Medium)
The repo uses `vnstock` (Pro/v5). Reliance on `source="VCI"` inside `Vnstock().stock()` is a brittle dependency if the provider API changes.
- **Recommendation**: Standardize all ingestion through a central `DataLoader` that handles version tracking.

---

## B. Data Processing Lifecycle

### Stages
1. **Raw**: `data/raw/` (CSV copies if requested).
2. **Database**: TimescaleDB `raw_prices`.
3. **Split Data**: `data/daily_market_split_data/` (Ticker-specific CSVs).
4. **Intermediate**: `data/processed/` (Merged datasets).
5. **Report**: `models/evaluation_report.csv`.

### Finding: Ticker-Based Splitting (Medium)
The repository splits data into 1600+ CSV files in `data/daily_market_split_data/`. 
- **Efficiency**: Opening 1600 files for batch training is disk-I/O intensive.
- **Risk**: File corruptions or partial writes during sync can lead to inconsistent training sets.
- **Recommendation**: Transition to a partitioned Parquet structure for faster I/O and better compression.

---

## C. Storage & Persistence Audit

### Convention
- **Format**: CSV for features, JSON for reports, Joblib for models.
- **Naming**: `models/<TICKER>/trend_classifier.joblib`.
- **Snapshotting**: The system saves results to `latest_predictions.json`.

### Finding: Artifact Bloat (High)
The `models/` directory contains 1,600+ ticker folders. This significantly increases repository size and makes versioning difficult.
- **Cleanup**: Archiving model folders for tickers not in the VN100 universe is recommended.
- **Recommendation**: Use an artifact registry or a compressed model bundle for daily releases.

### Finding: Configuration Scatter (High)
Hardcoded paths like `"data/market_proxy.csv"`, `"data/fundamentals_clean.csv"`, `"data/sentiment_features.csv"` are scattered throughout `scripts/train_ml_tickers.py`.
- **Maintainability**: Moving data indices or renaming files requires mass logic updates.
- **Recommendation**: Centralize all data paths in `src/data/universe.py` or a dedicated `data_registry.py`.
