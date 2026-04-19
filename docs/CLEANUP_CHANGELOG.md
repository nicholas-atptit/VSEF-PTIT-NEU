# Cleanup Changelog (Post-Refactor)

Detailed summary of file-level changes during the three-phase cleanup and refactor.

## Phase 1: Safe Cleanup

| File / Path | Action | Reason |
| :--- | :--- | :--- |
| `k-aware benchmark...` | DELETE | Corrupted/dead junk file. |
| `k-regime-upgrade` | MOVE | Archived root history. |
| `sponsor_after_upgrade.txt` | MOVE | Archived root history. |
| `reports/stress_test_report.md` | MODIFY | Cleaned absolute `H:\` paths. |
| `REFACTOR_LOG.md` | MODIFY | Cleaned absolute `H:\` path from runtime trace. |
| `STRUCTURE_FINAL.md` | MODIFY | Cleaned absolute root path in structure tree. |

## Phase 2: Structural Refactor

| File / Path | Action | New Location |
| :--- | :--- | :--- |
| `test_smoke_inference.py` | MOVE | `tests/smoke/test_smoke_inference.py` |
| `test_vnstock.py` | MOVE | `tests/tools/test_vnstock.py` |
| `scripts/run_phase*.py` | MOVE | `scripts/legacy/` |
| `scripts/research/` | MOVE | `scripts/legacy/research/` |
| `docs/prompt_runs/*.md` | MOVE | `docs/prompt_runs/archive/{phase}/` |
| `docs/prompt_runs/INDEX.md` | MODIFY | Updated hierarchical links for archive. |

## Phase 3: Documentation

| File | Role |
| :--- | :--- |
| `docs/CODEBASE_CLEANUP_REPORT.md` | Comprehensive audit report. |
| `docs/CODEBASE_STRUCTURE_REF.md` | Architecture and structure map. |
| `docs/CLEANUP_CHANGELOG.md` | This file. |
| `docs/TECH_SUMMARY_CLEANUP_AND_REFACTOR.md` | Professional summary for supervisor review. |
