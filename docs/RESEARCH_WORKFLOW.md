# Research Workflow

This workflow keeps benchmark results, data provenance, and claim language aligned.

## 1. Before Any Experiment

- Write a protocol before running code.
- Confirm the data scope, universe, frequency, and date windows.
- Confirm there is no target leakage or proxy leakage.
- Confirm metrics and thresholds before seeing final results.
- State the claim boundary before running the benchmark.

Do not run fetches, benchmarks, model training, paper generation, or DOCX generation as part of repository cleanup.

## 2. Data Workflow

- Use the provider gateway first.
- Keep raw provider calls limited to approved probes, diagnostics, adapters, and tests.
- Validate cache and coverage before benchmark execution.
- Never resample daily data to hourly.
- Preserve raw fetch data, market cache, outputs, generated reports, and archive snapshots.

Relevant entry points:

- `src/data/providers/vn_price_gateway.py`
- `src/data/providers/vn_provider_contract.py`
- `src/data/adapters/vnstock_adapter.py`
- `scripts/check_provider_usage_policy.py`

## 3. Benchmark Workflow

1. Run readiness checks.
2. Run the benchmark only if readiness passes and the protocol allows it.
3. Run the audit script for the benchmark family.
4. Write or update the result summary.
5. Update the relevant claim register.

Benchmark outputs are evidence artifacts. Do not rewrite metric values, labels, or empirical results during cleanup.

## 4. Claim Workflow

Safe claims:

- A claim tied directly to a named artifact, summary, audit, and date window.
- A provider or data-forensics claim supported by diagnostic outputs.
- A benchmark pass/fail statement with exact scope and metric definition.

Conditional claims:

- Claims that depend on coverage, data availability, provider support, or frequency.
- Claims that compare separate metric families, such as top-k ranking versus directional accuracy.

Unsafe claims:

- Trading readiness.
- Profitability.
- Investment recommendation.
- Live deployment.
- Full 2015-start hourly stock benchmark.
- Final65 success for stock hourly available-window.
- Any claim without artifact support.

## 5. Cleanup Workflow

- Archive or move preserved scripts; do not delete evidence or data.
- Delete only obvious junk such as bytecode cache when it is not tracked evidence.
- Keep active evidence and protected paths intact.
- Use `git mv` for tracked file renames.
- Update references after renaming docs, reports, or scripts.
- Prefer an index document over renaming protected generated folders.
- Do not create tags by default.
- Do not touch or merge `main` during branch cleanup.

Protected paths include:

- `data/`
- `outputs/`
- `reports/generated/`
- `archive/generated_data_snapshots/`
- `archive/reports_superseded/`
- `src/data/providers/`
- `src/data/adapters/`
- `tests/data/`
- `tests/ml/`

## 6. Current Active Evidence Entry Points

- `reports/_index/ACTIVE_EVIDENCE_INDEX.md`
- `reports/_index/ACTIVE_CODE_MAP.md`
- `docs/REPOSITORY_STRUCTURE.md`
- `reports/cleanup/REPO_RENAME_CLEANUP_INVENTORY.md`
- `reports/_index/REPO_CLEANUP_INVENTORY.md`
- `reports/cleanup/CODE_CLEANUP_CHANGES.md`

Use these before changing script layout, README language, benchmark claim text, or report indexes.
