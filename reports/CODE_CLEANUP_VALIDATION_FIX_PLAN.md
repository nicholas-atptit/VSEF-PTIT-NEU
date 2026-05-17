# Code Cleanup Validation Fix Plan

## Scope

This pass fixes validation hygiene only. It does not change provider behavior, model behavior, label logic, metric logic, empirical benchmark results, data-fetch behavior, research claims, paper artifacts, or DOCX output.

## Protected Paths

The following paths remain protected from deletion and empirical changes:

- `data/`
- `outputs/`
- `archive/generated_data_snapshots/`
- `reports/generated/`
- `archive/reports_superseded/`
- `src/data/providers/`
- `src/data/adapters/`
- `tests/data/`
- `tests/ml/`

## Hygiene Failures Captured

The pre-fix hygiene run reported 6,242 total violations:

- `tracked_generated_artifact`: 869 failures, all under `outputs/`.
- `local_absolute_path`: 5,373 failures.

The exact raw failure output was captured locally in `tmp/repo_hygiene_failures_before.txt`; this temporary file is not staged.

## Classification

### A. Policy Needs Update

Tracked `outputs/` files are intentional in this full-data-backup repository and are covered by Git LFS policy in `.gitattributes`.

Intended fix:

- Update `scripts/check_repo_hygiene.py` so `data/`, `outputs/`, `archive/generated_data_snapshots/`, `archive/reports_superseded/`, and `reports/generated/` are treated as approved full-data-backup paths only when Git attributes apply the `lfs` filter.
- Keep failures for unapproved generated roots such as `artifacts/`, `models/`, `tmp/`, bytecode, `__pycache__`, `node_modules`, egg-info metadata, malformed filenames, local file URIs, and local absolute paths.

### B. Redaction Needed

Local absolute paths were reported in tracked text metadata and inventory files:

- `reports/full_data_push_inventory.csv`: 5,134 lines.
- `reports/full_data_push_largest_files.csv`: 100 lines.
- Experiment manifests and logs under `outputs/experiments/`.
- Internal walk-forward model manifests under `outputs/walkforward_governance_audit*`.

Intended fix:

- Replace machine-local repository prefixes with `<repo>`.
- Replace machine-local virtual-environment prefixes with `<repo-approved-venv>` if found.
- Replace any remaining Windows absolute path prefix with `<local path redacted>`.
- Preserve filenames, relative paths, row counts, metric values, benchmark values, and claim text.

### C. Real Junk

The active compile check failed because Windows denied replacement of bytecode under `scripts/research/__pycache__`.

Intended fix:

- Remove only `scripts/research/__pycache__`.
- Run `py_compile` with `PYTHONPYCACHEPREFIX` set under `%TEMP%`.
- Do not commit bytecode cache files.

## Explicit Non-Actions

- Benchmark run: no.
- Data fetch: no.
- Model training: no.
- Paper/DOCX generation: no.
- Tags: no create, no push.
- Main branch: not touched.
