# VSEF Foreign-Flow Curated Sample
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Audit |
| Created / authored | Sunday, 2026-04-26 00:00:00 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-04-28 22:34:05 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `8800ce6e3780c7978856737e70cb5e3b999eacee` |
| Timestamp source | existing document date |
| Status | Active |

## Purpose

This note documents a small curated foreign-flow sample artifact used only for validator and coverage workflow checks. The sample is not live provider data, not market evidence, and not suitable for trading-performance interpretation.

## Sample File

```text
tests/fixtures/foreign_flow_sample.csv
```

The file is intentionally stored under `tests/fixtures/` instead of `data/foreign_flow.csv` so it is not confused with local provider cache data or production-style artifacts.

## Data Status

Real provider data was not fetched in this task.

The sample values are explicitly non-real fixture values. They are labeled with:

- `source = fixture_sample`
- `provider = non_real_fixture`
- `coverage_note = Non-real fixture sample for validator and coverage workflow tests only.`

These rows must not be interpreted as real foreign-flow activity for `SSI`, `FPT`, or any other ticker.

## Schema

The sample includes:

- `ticker`
- `date`
- `foreign_net_volume`
- `foreign_net_value`
- `foreign_buy_volume`
- `foreign_buy_value`
- `foreign_sell_volume`
- `foreign_sell_value`
- `source`
- `source_date`
- `retrieved_at`
- `provider`
- `coverage_note`

The controlled validation window is:

- tickers: `SSI`, `FPT`
- dates: 2025-01-02 through 2025-01-10, business days only

## Validation Result

For the controlled sample window, the validator reports:

- `artifact_classification = usable_for_requested_window`
- `fixture_or_sample_source = true`
- `real_provider_evidence = false`
- `suitable_for_performance_interpretation = false`

This means the artifact is complete enough to test exact ticker/date coverage mechanics, but it is not real provider evidence.

## Diagnostic Command

```bash
python scripts/audit_foreign_flow_coverage.py --tickers SSI,FPT --start-date 2025-01-02 --end-date 2025-01-10 --foreign-flow-path tests/fixtures/foreign_flow_sample.csv
```

When paired with matching OHLCV rows, the diagnostic script should report full ticker/date coverage and preserve the fixture/sample warning in `artifact_validation`.

## Limitations

- The sample uses non-real values.
- It validates workflow mechanics only.
- It does not validate `vnstock_data` provider availability.
- It does not prove source release timing, leakage absence, feature causality, or trading performance.
- It does not change model governance status or feature-selection behavior.

## Next Recommended Task

Create a real foreign-flow curation run only if provider access is available and appropriate:

- fetch or curate real rows for the target tickers and dates
- record provider, source date, retrieval timestamp, and coverage notes
- validate the artifact with `src/ml/backtest/foreign_flow_validation.py`
- rerun context coverage diagnostics before interpreting any `foreign_*` feature behavior

The follow-up provider-attempt workflow is documented in `docs/audits/VSEF_FOREIGN_FLOW_PROVIDER_CURATION_ATTEMPT.md`. It adds a provider-backed curation script but does not treat fixture/sample rows as real provider evidence.
