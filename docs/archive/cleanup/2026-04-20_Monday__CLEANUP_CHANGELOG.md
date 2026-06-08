# Cleanup Changelog (Post-Refactor)
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

Detailed summary of file-level changes during the three-phase cleanup and refactor.

## Phase 1: Safe Cleanup

| File / Path | Action | Reason |
| :--- | :--- | :--- |
| `k-aware benchmark...` | DELETE | Corrupted/dead junk file. |
| `k-regime-upgrade` | MOVE | Archived root history. |
| `sponsor_after_upgrade.txt` | MOVE | Archived root history. |
| `reports/superseded/stress_test_report.md` | MODIFY | Cleaned absolute `H:\` paths. |
| `docs/archive/root/REFACTOR_LOG.md` | MODIFY | Cleaned absolute `H:\` path from runtime trace. |
| `docs/archive/root/STRUCTURE_FINAL.md` | MODIFY | Cleaned absolute root path in structure tree. |

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
