# Repository Structure & Hygiene Audit

## A. Codebase Map

### Folder Responsibilities
- `src/ml/`: Core logic, feature computation, and specialized model trainers.
- `src/data/`: Data adapters and universe definitions.
- `src/engine/`: Consensus/Multi-Agent Debate logic.
- `src/api/`: Frontend/V2 API using FastAPI.
- `src/ui/`: Terminal dashboard.
- `scripts/`: Operational entry points (sync, train, predict).

### Finding: Conflicting Scripts (Medium)
There are multiple "sync/load" scripts: `scripts/sync_all_data.py`, `scripts/import_historical_csv.py`, `scripts/fetch_fundamentals.py`.
- **Logic Overlap**: `sync_all_data.py` should be the master orchestrator. 
- **Risk**: Overlapping functionality can lead to inconsistent database states.

---

## B. Modularity & Dependencies

### Cross-Module Dependencies
- The `src/ml/trainer.py` (DualModelTrainer) is centralized and well-structured.
- `src/ml/signal_generator.py` (SignalGenerator) depends on `trainer.py` outputs. This follows a clean DAG.

### Finding: Hardcoded Magic Numbers (Medium)
Constants such as `MIN_ROWS = 40`, `WINDOWS = [5, 20, 60]`, `PURGE_GAP = 3` are hardcoded in the scripts.
- **Config Scatter**: Adjusting these requires code changes.
- **Recommendation**: Move all ML parameters to `config/settings.py` or a dedicated `ml_params.yaml`.

---

## C. Folder & File Hygiene

### Finding: Massive Model Directory (Critical)
The `models/` directory has 1600+ folders. 
- **Bloat**: Large repo size and hard to manage meta-information about models.
- **Inconsistency**: Some tickers may be stale or have insufficient liquidity.
- **Recommendation**: Restrict the training universe to VN100 by default and move other models to a `legacy/` or `archive/` folder.

### Dead/Archive Candidate Code
- `scripts/prepare_vip_list.py` (Likely from an older phase).
- `scripts/準備_fundamental_features.py` (Need to verify usage).
- `scripts/import_historical_csv.py` (Redundant if `sync_all` is the master).

---

## D. Architecture Summary

### Status
- **Pros**: Clear domain separation (Technical vs Sentiment), robust trainer, event-driven paper engine.
- **Cons**: High storage bloat, config scattering, and redundant ingestion scripts.
- **Verdict**: The architecture is sound but requires "Sanitization" for production scaling.
