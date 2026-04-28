# Codebase Cleanup Report (Three-Phase Pass)
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Historical archive |
| Created / authored | Monday, 2026-04-20 00:28:39 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-04-28 22:28:23 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `8800ce6e3780c7978856737e70cb5e3b999eacee` |
| Timestamp source | Git history |
| Status | Historical reference |

This report documents the systematic cleanup and refactor conducted to improve repository maintainability and architectural clarity.

## Cleanup Scope
- **Safe Cleanup**: Root clutter removal and absolute path normalization.
- **Structural Refactor**: Organization of legacy runners and research scripts.
- **Archive Migration**: Clustering of historical documentation.

## Actions Taken

### Deleted Items
- `k-aware benchmark, stress testing, tuning, and legacy compatibility fixes?git push origin HEAD` (Corrupted junk file).
- `tmpl*` and `tmpoctg4a3n` (Temporary directories).

### Archived / Moved Items
- **Root History**: Moved `k-regime-upgrade` and `sponsor_after_upgrade.txt` to `archive/root_history/`.
- **Legacy Runners**: Moved `run_phase*.py` and `benchmark_local_baseline.py` to `scripts/legacy/`.
- **Research**: Moved `scripts/research/` to `scripts/legacy/research/`.
- **Tests**: Moved root-level `test_smoke_inference.py` and `test_vnstock.py` to `tests/smoke/` and `tests/tools/` respectively.
- **Prompt Runs**: Clustered 20+ historical reports into `docs/prompt_runs/archive/` (categorized by Phase 1, Phase 2, Quant Core, Analysis Feed).

### Refactored Components
- **Documentation Paths**: Normalized absolute `H:\` and `file:///` paths in `docs/archive/root/REFACTOR_LOG.md`, `docs/archive/root/STRUCTURE_FINAL.md`, and `reports/stress_test_report.md` to relative repo paths.
- **Registry & Links**: Updated `docs/prompt_runs/INDEX.md` to reflect the new hierarchical structure of the archive.

## Preserved Intentionally
- `src/ml` and `src/forecast` were kept as separate layers to avoid deep interface breaking in this pass.
- Full model zoo support in `src/forecast/registry.py` remains intact.
- Validated runtime artifacts in `artifacts/` were preserved.

## Validation Evidence
- **Quant Core**: `python scripts/run_quant_core.py --preset smoke` completed successfully (924 forecast rows).
- **Analysis Feed**: `python scripts/run_analysis_feed.py --mode smoke` completed successfully (generated manifest in `artifacts/analysis_feed/`).
- **Unit Tests**: `pytest tests/quant_core tests/retrieval` passed fully (23 passed).

## Portability Summary
The repository no longer contains machine-specific absolute paths in its core documentation, significantly improving portability and "clean-clone" behavior.
