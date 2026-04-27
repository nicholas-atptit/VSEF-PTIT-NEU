# VSEF Foreign-Flow Coverage Investigation

Date: 2026-04-26

Branch: `vsef-foreign-flow-coverage-investigation`

## Purpose

This investigation reviews why cached walk-forward governance audits reported absent foreign-flow context coverage. It is a data-governance review only. It does not add model families, change model governance status, remove features, or claim improved trading performance.

## Why Foreign-Flow Coverage Matters

Foreign-flow features such as `foreign_net_value`, `foreign_net_value_ratio`, `foreign_participation_20`, and `abnormal_foreign_participation_20` depend on joined ticker/date source rows. If source coverage is absent, those features may be structurally unavailable or weakly supported in a run. Any interpretation of their importance or coefficient behavior should remain conservative until source coverage is confirmed.

## Source Path And Provider Findings

The local default foreign-flow path is:

```text
data/foreign_flow.csv
```

The loader path is:

- `load_foreign_flow()` reads `data/foreign_flow.csv` when present.
- If the file is missing and explicit tickers are supplied, `build_foreign_flow_incremental()` can attempt a live `vnstock_data` fetch through `VnstockAdapter.get_foreign_flow()`.
- If the file is missing and no ticker list is supplied, the loader returns a stub frame.

The local artifact exists, but the investigation found it contains only one row:

| property | value |
| --- | --- |
| source status | `loaded_local_artifact` |
| row count | 1 |
| ticker coverage | `TEST` |
| date range | 2026-04-24 to 2026-04-24 |

The broader audit used `SSI`, `FPT`, `ACB`, and `HPG` for January 2025. The local foreign-flow artifact has no rows for those tickers and no rows in that date window.

## Diagnostic Command

```bash
python scripts/audit_foreign_flow_coverage.py --tickers SSI,FPT,ACB,HPG --start-date 2025-01-02 --end-date 2025-01-31
```

Result summary:

| ticker | requested OHLCV dates | foreign-flow rows for ticker | exact join matches | exact join missing ratio |
| --- | ---: | ---: | ---: | ---: |
| `SSI` | 17 | 0 | 0 | 1.0000 |
| `FPT` | 17 | 0 | 0 | 1.0000 |
| `ACB` | 17 | 0 | 0 | 1.0000 |
| `HPG` | 17 | 0 | 0 | 1.0000 |

The probe is read-only and does not fetch provider data.

## Ticker And Date Alignment Findings

The join behavior is exact ticker/date alignment:

- ticker values are normalized to uppercase
- dates are normalized to daily timestamps
- future-dated rows are not pulled backward
- ticker mismatches produce missing availability
- date mismatches produce missing availability

No join-code bug was identified in this task. The absent coverage is explained by source coverage: the only local row is for `TEST` on 2026-04-24, while the audit requested `SSI`, `FPT`, `ACB`, and `HPG` in January 2025.

## Local Fallback And Provider Availability

The broader audit used local CSV fallback OHLCV data rather than live `vnstock_data`. A local foreign-flow fallback exists as `data/foreign_flow.csv`, but it does not cover the audited tickers or date range.

Provider availability would be required to build a usable foreign-flow artifact for the audited tickers if no curated local artifact is supplied. This investigation did not fetch live provider data and did not fabricate replacement foreign-flow rows.

## Why Broader Audit Coverage Was 100% Missing

`context_coverage_summary.csv` reported `mean_foreign_flow_missing_rate = 1.0000` and `weak_coverage` because every requested OHLCV date for each audited ticker lacked a matching foreign-flow ticker/date source row.

That result is expected under the current local data state.

## Tests Added

`tests/ml/test_foreign_flow_coverage.py` covers:

- full missing foreign-flow coverage when no source rows exist
- exact ticker/date joins setting availability metadata
- ticker mismatch producing missing coverage
- date mismatch producing missing coverage
- matching rows producing non-missing coverage
- missing local source reporting without crashing
- the read-only diagnostic probe running against synthetic local paths without provider access

## Changes Made

Added a read-only diagnostic script:

```text
scripts/audit_foreign_flow_coverage.py
```

No model code, feature selection behavior, API routes, or training behavior was changed. No data was fabricated or fetched.

## Limitations

- This investigation explains the cached local state; it does not validate live `vnstock_data` availability.
- The local artifact may be a test fixture or prior scratch artifact and should not be treated as broad foreign-flow coverage.
- Coverage diagnostics measure exact source-row availability, not official release timestamps.
- Absence of matching rows does not prove foreign-flow data is unavailable from all sources.
- This report does not establish leakage absence, causality, or trading performance.

## Next Recommended Task

Create a governed foreign-flow artifact refresh task:

- explicitly fetch or curate `SSI`, `FPT`, `ACB`, and `HPG` foreign-flow rows for the intended audit window
- record source provenance, date range, ticker coverage, and provider status
- rerun context coverage diagnostics
- require minimum foreign-flow coverage before interpreting `foreign_*` feature diagnostics

## Follow-Up: Artifact Curation Policy

`docs/governance/VSEF_FOREIGN_FLOW_ARTIFACT_POLICY.md` defines the governed schema and validation expectations for foreign-flow artifacts. The follow-up validator classifies the current local artifact as fixture-like when it only contains `TEST` rows and prevents that file from being mistaken for real ticker/date coverage.

`docs/audits/VSEF_FOREIGN_FLOW_CURATED_SAMPLE.md` documents the non-real curated sample fixture used to validate foreign-flow coverage mechanics without claiming provider data was fetched.

`docs/audits/VSEF_FOREIGN_FLOW_PROVIDER_CURATION_ATTEMPT.md` documents the provider-backed curation script and the current local limitation: `vnstock_data` was unavailable, so no real provider rows were fetched.
