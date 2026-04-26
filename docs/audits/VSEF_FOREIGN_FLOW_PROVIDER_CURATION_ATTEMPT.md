# VSEF Foreign-Flow Provider Curation Attempt

Date: 2026-04-26

Branch: `vsef-foreign-flow-provider-curation-attempt`

## Purpose

This note documents the provider-backed foreign-flow curation path added after the fixture and artifact-validation work. The goal is to make a real provider attempt possible when `vnstock_data` is available, while failing clearly when provider support is unavailable.

This is a data-governance workflow only. It does not add model families, change model governance status, change training behavior, or claim trading-performance improvement.

## Provider Capability Findings

`VnstockAdapter.get_foreign_flow(symbol, start_date, end_date)` exists in `src/data/adapters/vnstock_adapter.py`.

The adapter path:

- requests `Trading.foreign_trade` through `vnstock_data`
- tries VCI first and CAFEF as fallback
- accepts one ticker symbol plus start and end dates
- normalizes returned dates and ticker values
- standardizes common `fr_*` fields into `foreign_*` columns such as `foreign_buy_volume`, `foreign_sell_volume`, `foreign_net_volume`, `foreign_buy_value`, `foreign_sell_value`, and `foreign_net_value`

The local runtime did not expose `vnstock_data` during this task. A direct import probe returned no module specification, so provider-backed curation could not fetch real rows in this environment.

## Script Added

```text
scripts/curate_foreign_flow_provider_artifact.py
```

The script attempts provider-backed curation only when the active runtime exposes the `vnstock_data` `Trading` interface through `VnstockAdapter`.

Suggested command when provider access is available:

```bash
python scripts/curate_foreign_flow_provider_artifact.py --tickers SSI,FPT,ACB,HPG --start-date 2025-01-02 --end-date 2025-01-31 --output-path data/foreign_flow_curated.csv
```

The default output path is:

```text
data/foreign_flow_curated.csv
```

The script refuses to overwrite:

```text
data/foreign_flow.csv
```

## Output Path Policy

`data/foreign_flow_curated.csv` is ignored by `.gitignore` so local provider pulls are not committed by accident.

Generated provider artifacts should be committed only after an explicit curation decision confirms that the file is small, governed, documented, and appropriate to retain. This task did not commit generated provider data.

## Provenance Schema

When provider rows are returned, the script adds:

- `source`
- `source_date`
- `retrieved_at`
- `provider`
- `coverage_note`

Rows fetched through the adapter are labeled with:

- `source = vnstock_data.Trading.foreign_trade`
- `provider = vnstock_data`

If a mocked or supplied provider response carries fixture/sample source hints, the script preserves that status and labels the output as non-real fixture evidence instead of marking it as real provider data.

## Validation Behavior

The script validates the resulting frame with:

```text
src/ml/backtest/foreign_flow_validation.py
```

The report includes:

- artifact classification
- requested ticker/date coverage rate
- provenance completeness
- fixture/sample source status
- whether the artifact is real provider evidence

If provider support is unavailable, the script returns `provider_unavailable`, writes no output file, and validates an empty frame as `empty_or_missing`.

## Local Attempt Result

Provider-backed fetch was not attempted locally because `vnstock_data` was unavailable in the active runtime.

No real foreign-flow rows were fetched.

No generated provider data was committed.

## Limitations

- This task validates the curation workflow and unavailable-provider path; it does not prove live provider availability.
- The script does not validate official provider release timestamps.
- Complete provider ticker/date coverage would still be governance metadata, not evidence of causality or trading performance.
- Fixture/sample rows remain suitable only for workflow tests and must not be treated as market evidence.

## Next Recommended Task

Run the provider curation command in an environment where `vnstock_data` is installed and authorized, then audit the generated `data/foreign_flow_curated.csv` with:

```bash
python scripts/audit_foreign_flow_coverage.py --tickers SSI,FPT,ACB,HPG --start-date 2025-01-02 --end-date 2025-01-31 --foreign-flow-path data/foreign_flow_curated.csv
```

Review whether coverage is sufficient before interpreting any `foreign_*` feature diagnostics.
