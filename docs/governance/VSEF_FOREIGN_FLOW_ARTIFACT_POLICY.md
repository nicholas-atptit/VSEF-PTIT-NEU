# VSEF Foreign-Flow Artifact Policy

Date: 2026-04-26

Branch: `vsef-foreign-flow-artifact-curation`

## Purpose

This policy defines how foreign-flow artifacts should be curated and validated before they are used as context evidence in VSEF audits. It is a data-governance policy only. It does not change model governance status, feature selection, training behavior, or trading-performance claims.

## Why Provenance Matters

Foreign-flow features depend on ticker/date source rows. Missing, fixture-like, stale, or partially covered artifacts can make `foreign_*` features appear unavailable or weakly supported. They can also create misleading confidence if a row with measured zero is confused with a missing-context fallback.

Every foreign-flow artifact used for governance interpretation should identify where it came from, when it was retrieved, and which ticker/date window it covers.

## Expected Schema

Required columns:

- `ticker`
- `date`
- at least one numeric foreign-flow measure

Accepted numeric foreign-flow measures include columns such as:

- `foreign_buy_volume`
- `foreign_sell_volume`
- `foreign_net_volume`
- `foreign_buy_value`
- `foreign_sell_value`
- `foreign_net_value`
- `foreign_room_pct`
- `foreign_owned_pct`
- `foreign_available_pct`

Recommended provenance columns:

- `source`
- `source_date`
- `retrieved_at`
- `provider`
- `coverage_note`

## Normalization Rules

Ticker values should be uppercase, stripped strings matching the ticker convention used by OHLCV files.

Date values should be normalized to daily dates before coverage checks. Foreign-flow joins remain exact ticker/date joins. Future-dated rows must not be pulled backward into earlier prediction rows.

## Artifact Classification

The validation utility classifies artifacts as:

| Classification | Interpretation |
| --- | --- |
| `usable_for_requested_window` | Required schema is present, at least one numeric flow measure exists, and requested ticker/date coverage is complete. |
| `partial_coverage` | Schema is valid, but requested ticker/date coverage is incomplete or absent for real tickers. |
| `fixture_only` | The artifact contains only fixture-like tickers such as `TEST`, `DUMMY`, or `SAMPLE`. |
| `schema_invalid` | Required columns or numeric flow measures are missing. |
| `empty_or_missing` | No artifact rows are available. |

Only `usable_for_requested_window` should be treated as suitable for interpreting `foreign_*` feature diagnostics for a requested audit window.

## Minimum Coverage Interpretation

Coverage should be evaluated against the requested ticker/date window. A row outside the requested date range or for a different ticker does not support interpretation for that audit.

Partial coverage can still be useful for diagnosing source availability, but it should not be used as evidence that foreign-flow features are broadly governed or reliable.

## Fixture And Cache Policy

Fixture rows such as ticker `TEST` must not be interpreted as real market coverage. They may be useful for local tests, but they should be clearly separated from curated real artifacts.

Local raw provider pulls and scratch caches should not be broadly tracked. Curated sample data may be tracked only when it is intentionally documented, small, and required for reproducible tests or examples.

The current `data/foreign_flow.csv` file is ignored by `.gitignore` and is not tracked. In the local workspace it contains only a `TEST` row dated 2026-04-24, so it is fixture-like and unsuitable for interpreting `SSI`, `FPT`, `ACB`, or `HPG` foreign-flow coverage.

The curated sample fixture at `tests/fixtures/foreign_flow_sample.csv` is tracked for validator and coverage workflow tests only. It uses `source = fixture_sample` and `provider = non_real_fixture`, so it can validate exact ticker/date mechanics without being treated as real provider evidence.

## Validation Utility

The read-only validator is implemented in:

```text
src/ml/backtest/foreign_flow_validation.py
```

The diagnostic probe uses the validator:

```text
scripts/audit_foreign_flow_coverage.py
```

The validator does not fetch provider data, fabricate rows, change loader behavior, or remove features.

## Provider Curation Attempt

`docs/audits/VSEF_FOREIGN_FLOW_PROVIDER_CURATION_ATTEMPT.md` documents the follow-up provider-backed curation script:

```text
scripts/curate_foreign_flow_provider_artifact.py
```

The script writes provenance columns when `vnstock_data` provider access is available, refuses to overwrite `data/foreign_flow.csv`, and treats fixture/sample-labeled rows as non-real evidence.

`docs/audits/VSEF_FOREIGN_FLOW_REAL_COVERAGE_AUDIT.md` documents a real provider-artifact coverage audit. The artifact was provider-backed but still partial for the requested January 2025 business-date window, so coverage interpretation remains conservative.

## Private Repository Reminder

VSEF is private and proprietary. Foreign-flow artifacts, validation logic, and documentation in this repository are not licensed for public reuse, redistribution, deployment, or publication without written permission from the repository owner.

## Limitations

- Validation checks source-row coverage, not official provider release timestamps.
- Complete local coverage does not prove feature causality or trading value.
- Provider-side availability can change and must be audited separately.
- Coverage classification does not automatically include or exclude features from model training.
