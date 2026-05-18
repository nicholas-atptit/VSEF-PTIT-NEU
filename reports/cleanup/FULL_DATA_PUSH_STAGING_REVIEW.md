# Full Data Push Staging Review

Created at local time: 2026-05-17 20:36:05 +07:00

## Summary
- Staged file count: 5254
- Staged size estimate from working tree files: 499.44 MB
- LFS tracked file count: 4906
- Inventory coverage missing count: 0
- Forbidden paths staged: False
- Forbidden paths removed before commit: not needed

## Scope Notes
- Staged `.gitattributes` with Git LFS rules for data, outputs, reports/generated, archive snapshots, superseded reports, and heavy file extensions.
- Force-added ignored data/output/archive artifact scopes requested for the reproducibility package.
- Did not stage node_modules, virtual environments, Python caches, pytest caches, .env files, or secret-like filenames.
