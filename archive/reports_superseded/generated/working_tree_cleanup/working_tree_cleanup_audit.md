# Working Tree Cleanup Audit

## Summary

- Branch: `research/vn100-evidence-hardening-v1`
- Cleanup purpose: remove generated data/archive/cache clutter from normal Source Control before continuing VN30 hourly 2015 data-readiness work.
- Changed/untracked paths before cleanup, using `git status --porcelain -uall`: 10,139.
- Changed/untracked paths after cleanup, before metadata commit: 14.
- Massive generated flood removed from normal status: yes.
- Benchmark run: no.
- Data fetch run: no.
- Model training run: no.
- Paper/DOCX generation run: no.

## Classification

### A. Source/config/docs that may need to be committed

- `.gitignore`: cleanup-specific ignore rules added narrowly.
- `reports/generated/working_tree_cleanup/working_tree_cleanup_audit.md`: this audit report.

### B. Generated reports that are small and intentionally tracked

- Existing tracked generated markdown/CSV reports under `reports/generated/` remain tracked where already committed.
- No generated report CSVs were newly staged for this cleanup commit.

### C. Raw data/cache/output/archive artifacts that must not be committed

Removed or hidden from normal status:

- `archive/generated_data_snapshots/`
- `data/raw/vnstock_fetch/`
- `data/market_cache/`
- `outputs/`
- untracked generated fetch/probe folders under:
  - `reports/generated/index_hourly_fetch/fetch/`
  - `reports/generated/index_hourly_fetch/provider_probe/`
  - `reports/generated/index_hourly_gateway/fetch/`
  - `reports/generated/vn30_hourly_gateway/fetch/`

These are reproducible generated artifacts and should not be staged.

### D. Accidental local changes requiring user decision

README.md is modified. The diff adds a short Vietnamese note attributing Vnstock:

```text
Dữ liệu được kết nối và truy xuất thông qua Vnstock - gói phần mềm Python phân tích thị trường chứng khoán Việt Nam. 
+(thinh-vu @ Github, Copyright (c) 2022-2026).
```

This looks like a meaningful manual documentation edit, not generated noise. It was left unstaged and was not reverted.

Other pre-existing modified source/archive/report files remain unstaged and were not touched by this cleanup.

## Top-Level Directories Affected

Before cleanup, the changed/untracked flood was concentrated in:

- `data/`: 9,653 paths
- `archive/`: 478 paths
- `src/`: 6 modified tracked files
- `README.md`: 1 modified tracked file
- `reports/`: 1 modified tracked file

After cleanup, normal status contains only tracked modifications and no untracked generated flood.

## Ignore Rules

Confirmed existing ignores:

- `data/market_cache/`
- `/outputs/`

Added narrow ignores:

- `data/raw/`
- `archive/generated_data_snapshots/`
- `reports/generated/**/fetch/*.csv`
- `reports/generated/**/fetch/*chunk_log*.csv`
- `reports/generated/**/fetch/*failures*.csv`
- `reports/generated/**/validation/*.csv`
- `reports/generated/**/debug/cache_roundtrip/`
- `reports/generated/index_hourly_2015/`
- `reports/generated/vn30_hourly_2015/`

These rules target generated data products and do not ignore source files, scripts, configs, or top-level report plans.

## Tracked Generated Files

Tracked generated files were found under `reports/generated/` and one tracked raw file exists under `data/raw/`. They were not removed or untracked in this cleanup because they are already part of repository history and require a separate explicit policy decision before any `git rm --cached` action.

## Cleanup Notes

Scoped cleanup commands were used rather than broad `git clean -fd` or `git clean -fdx`.

Some ignored archive/output files reported Windows delete errors during scoped cleanup, but the remaining paths are ignored and no longer appear in normal Source Control status.
