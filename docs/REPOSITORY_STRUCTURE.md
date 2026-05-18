# Repository Structure

This repository is the VN Market Directional Benchmark Lab. Its active identity is VN30 directional forecasting research, provider-governed Vietnamese OHLCV access, evidence tracking, and claim-boundary control.

## Top-Level Folders

| Path | Meaning |
| --- | --- |
| `src/` | Importable source code, including provider contracts, adapters, metrics, and reusable platform modules. |
| `scripts/` | Runnable entry points for validation, audits, fetch/readiness work, benchmarks, and research diagnostics. |
| `scripts/research/` | Active research scripts. Some are safe validators; benchmark/fetch/train scripts require a written protocol before use. |
| `scripts/legacy/` | Preserved superseded scripts, failed experiments, old paper builders, and compatibility material. Do not delete for cleanup convenience. |
| `tests/` | Automated tests. `tests/data/` and `tests/ml/` are protected because they enforce provider policy and metric contracts. |
| `docs/` | Human-facing usage, workflow, repository structure, governance, runbooks, and archived documentation. |
| `reports/` | Active evidence summaries, claim registers, cleanup reports, and source artifact indexes. |
| `reports/generated/` | Generated reports, generated figures/tables, and diagnostic output summaries. Treat as preserved evidence. |
| `data/` | Raw, cached, or curated market data and local data workspaces. |
| `outputs/` | Benchmark, prediction, model-run, and diagnostic output artifacts. |
| `archive/` | Historical snapshots and superseded reports retained for provenance. |
| `artifacts/`, `models/`, `tmp/`, `tmp_reports/` | Runtime or local artifact areas. Avoid adding new tracked files unless a protocol requires it. |
| `config/`, `configs/`, `infra/`, `alembic/`, `tools/` | Configuration, infrastructure, migrations, and supporting utilities. |

## Protected Paths

These paths may be read for inventory and validation, but they must not be reorganized, deleted, or renamed during cleanup unless the change is a small report/index document and the reason is documented:

- `data/`
- `outputs/`
- `archive/generated_data_snapshots/`
- `archive/reports_superseded/`
- `reports/generated/`
- `src/data/providers/`
- `src/data/adapters/`
- `tests/data/`
- `tests/ml/`

## Active Evidence

Start with these files when checking current evidence and allowed claims:

- `reports/ACTIVE_EVIDENCE_INDEX.md`
- `reports/ACTIVE_CODE_MAP.md`
- `reports/VN30_HOURLY_SELECTED_CANDIDATE_ROLLING_STABILITY_RESULT.md`
- `reports/VN30_HOURLY_SELECTED_CANDIDATE_CLAIM_BOUNDARY.md`
- `reports/VN30_RESEARCH_CLAIM_REGISTER.md`
- `reports/VN30_DAILY_2015_RESULT_SUMMARY.md`
- `reports/VN30_INDEX_BENCHMARK_RESULT_SUMMARY.md`
- `reports/VN30_INDEX_BENCHMARK_CLAIM_REGISTER.md`

## Generated Artifacts

Generated folders are evidence outputs, not a naming playground. Current generated evidence includes:

- `reports/generated/vn30_hourly_selected_candidate_rolling/`
- `reports/generated/paper_tables_current/`
- `reports/generated/paper_figures_current/`
- `reports/generated/vn30_daily_2015/`
- `reports/generated/index_benchmark/`
- `outputs/`

Where folder names are historically useful but not ideal, prefer a small index document over folder renames.

## Safe Validation Scripts

These are safe validation commands and do not fetch data, run benchmarks, train models, or generate paper/DOCX artifacts:

- `python scripts/check_repo_hygiene.py`
- `python scripts/check_runtime_preflight.py`
- `<repo-approved-venv>\Scripts\python.exe scripts\check_runtime_preflight.py`
- `<repo-approved-venv>\Scripts\python.exe scripts\check_provider_usage_policy.py`
- `<repo-approved-venv>\Scripts\python.exe -m pytest tests/data/test_provider_usage_policy.py -q`
- `<repo-approved-venv>\Scripts\python.exe -m pytest tests/data/test_vn_price_gateway_contract.py -q`
- `<repo-approved-venv>\Scripts\python.exe -m pytest tests/ml/test_directional_accuracy_metrics.py -q`
- `powershell -ExecutionPolicy Bypass -File scripts/dev_tasks.ps1 -Task validate-all`

## Research And Benchmark Scripts

Do not run these casually. They can fetch data, run benchmarks, train/refit models, or generate result artifacts depending on the script:

- `scripts/research/fetch_*.py`
- `scripts/research/refetch_*.py`
- `scripts/research/run_*benchmark*.py`
- `scripts/research/run_vn30_*target*.py`
- `scripts/research/run_vn30_hourly_*comparison*.py`
- `scripts/research/run_vn30_hourly_*stacking*.py`
- `scripts/research/build_vn30_hourly_paper_empirical_tables.py`
- `scripts/research/build_vn30_hourly_paper_empirical_figures.py`

Audit scripts under `scripts/research/audit_*.py` are usually read-only over existing artifacts, but check the protocol and output path before running them.

## Paper And Source Artifacts

Paper/source index files use explicit VN30 hourly names:

- `reports/VN30_HOURLY_PAPER_FIGURE_DATA_SOURCE_INVENTORY.md`
- `reports/VN30_HOURLY_PAPER_TABLE_FIGURE_CAPTIONS_EN.md`
- `reports/VN30_HOURLY_PAPER_TABLE_FIGURE_CAPTIONS_VI.md`
- `reports/VN30_HOURLY_PAPER_MISSING_METRICS_TODO.md`
- `reports/VN30_HOURLY_PAPER_ROW_LEVEL_FIGURE_TODO.md`
- `reports/VN30_HOURLY_PAPER_LITERATURE_DATA_TODO.md`
- `reports/VN30_HOURLY_PAPER_DOCX_MISSING_FOR_FIGURE_INSERTION.md`
- `reports/VN30_HOURLY_PAPER_WITH_FIGURES_LAYOUT_QA.md`

Paper/DOCX drafts are not generated during cleanup. Builders must read existing repository artifacts only.

## Naming Convention

Future active files should use stable, descriptive names:

- Reports: `reports/VN30_HOURLY_SELECTED_CANDIDATE_<SUBJECT>.md`, `reports/VN30_DAILY_2015_<SUBJECT>.md`, or `reports/VN30_INDEX_BENCHMARK_<SUBJECT>.md`.
- Claim registers: end with `_CLAIM_REGISTER.md` or `_CLAIM_BOUNDARY.md`.
- Protocols: end with `_PROTOCOL.md`.
- Results: end with `_RESULT.md` or `_RESULT_SUMMARY.md`.
- Scripts: use action names such as `audit_...`, `rerun_...`, `build_...`, `validate_...`, `fetch_...`, or `run_...`.
- Avoid vague active names such as `CURRENT`, `PAPER_READY`, `RESULT_V1`, `TARGET62`, or `FINAL65` unless that phrase is the actual protocol scope being preserved.

## Do Not Rename Data/Output/Archive Artifacts Casually

Data, output, generated, and archive paths are often referenced by manifests, reports, tests, Git LFS tracking, and external review notes. Renaming them without a protocol can break provenance even when metric values are unchanged.
