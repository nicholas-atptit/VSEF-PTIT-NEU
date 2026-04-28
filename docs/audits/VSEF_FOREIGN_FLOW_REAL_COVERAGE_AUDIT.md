# VSEF Foreign-Flow Real Coverage Audit
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

This audit verifies whether `data/foreign_flow_curated.csv` is a real provider-backed foreign-flow artifact, audits exact ticker/date coverage, and reruns the broader walk-forward governance audit against that artifact.

This is a data-governance audit only. It does not add model families, change model governance status, or claim trading-performance improvement.

## Artifact Inspection

Artifact path:

```text
data/foreign_flow_curated.csv
```

The file exists locally and was not committed.

Inspection summary:

| item | value |
| --- | --- |
| row count | 68 |
| tickers | `ACB`, `FPT`, `HPG`, `SSI` |
| date range | 2025-01-02 through 2025-01-24 |
| provider | `vnstock_data` |
| source | `vnstock_data.Trading.foreign_trade` |
| fixture/sample source flag | false |
| provenance columns present | yes |

Provenance columns present:

- `source`
- `source_date`
- `retrieved_at`
- `provider`
- `coverage_note`

## Validator Result

The artifact was validated for:

- tickers: `SSI`, `FPT`, `ACB`, `HPG`
- date range: 2025-01-02 through 2025-01-31

Validation result:

| field | value |
| --- | --- |
| `artifact_classification` | `partial_coverage` |
| `row_count` | 68 |
| requested ticker/date pairs | 88 |
| matched ticker/date pairs | 68 |
| requested ticker/date coverage rate | 0.7727 |
| `fixture_or_sample_source` | false |
| `real_provider_evidence` | true |
| `provenance_complete` | true |

The artifact is real provider evidence for the rows it contains, but it is partial for the full 2025-01-02 through 2025-01-31 requested business-date window.

## Coverage Audit Command

```bash
python scripts/audit_foreign_flow_coverage.py --tickers SSI,FPT,ACB,HPG --start-date 2025-01-02 --end-date 2025-01-31 --foreign-flow-path data/foreign_flow_curated.csv
```

Coverage audit result:

| ticker | requested OHLCV dates | exact join matches | exact join missing ratio |
| --- | ---: | ---: | ---: |
| `SSI` | 17 | 17 | 0.0000 |
| `FPT` | 17 | 17 | 0.0000 |
| `ACB` | 17 | 17 | 0.0000 |
| `HPG` | 17 | 17 | 0.0000 |

The exact OHLCV-date coverage is complete for the local cached OHLCV dates available in the requested window. The validator still reports partial coverage against the full business-date calendar through 2025-01-31 because the artifact ends on 2025-01-24.

## Walk-Forward Rerun

The broader governance audit was rerun because the artifact is provider-backed and has partial coverage.

Command:

```bash
python scripts/run_walkforward_all_models_stacking_eval.py --tickers SSI FPT ACB HPG --history-start 2020-12-21 --history-end 2025-02-28 --initial-train-start 2020-12-21 --initial-train-end 2024-12-31 --forecast-start 2025-01-02 --forecast-end 2025-01-24 --horizons short_5d --step-sizes 1 --algorithms cart,xgboost,lightgbm --output-dir outputs/walkforward_governance_audit_foreign_flow_real --max-workers 1 --max-depth 3 --meta-min-samples 1 --epochs 1
```

The runner currently loads foreign-flow context from `data/foreign_flow.csv`. For this local rerun only, the ignored default file was temporarily replaced with `data/foreign_flow_curated.csv` and then restored to its original ignored `TEST` fixture content after the run. No generated provider data or output folder was committed.

Generated diagnostic folder:

```text
outputs/walkforward_governance_audit_foreign_flow_real/ssi_fpt_acb_hpg/_combined_internal/csv/
```

The folder contained the expected diagnostics, including:

- `context_coverage_diagnostics.csv`
- `context_coverage_summary.csv`
- `feature_governance_review.csv`
- `feature_importance_diagnostics.csv`
- `feature_importance_stability_summary.csv`
- `linear_coefficient_diagnostics.csv`
- `linear_coefficient_stability_summary.csv`
- `linear_vs_importance_feature_comparison.csv`

## Old Versus New Context Coverage

Previous broader audit:

| ticker | mean foreign-flow missing rate | max foreign-flow missing rate | warning |
| --- | ---: | ---: | --- |
| `ACB` | 1.0000 | 1.0000 | `weak_coverage` |
| `FPT` | 1.0000 | 1.0000 | `weak_coverage` |
| `HPG` | 1.0000 | 1.0000 | `weak_coverage` |
| `SSI` | 1.0000 | 1.0000 | `weak_coverage` |

Provider-artifact rerun:

| ticker | mean foreign-flow missing rate | max foreign-flow missing rate | warning |
| --- | ---: | ---: | --- |
| `ACB` | 0.9842 | 0.9982 | `weak_coverage` |
| `FPT` | 0.9842 | 0.9982 | `weak_coverage` |
| `HPG` | 0.9842 | 0.9982 | `weak_coverage` |
| `SSI` | 0.9842 | 0.9982 | `weak_coverage` |

The provider artifact improves coverage from complete absence to partial source-row availability, but fold-level coverage remains weak because each fold-level feature frame spans the training history while the provider artifact only covers January 2025.

## Loader Safeguard Added

The first walk-forward rerun exposed that provider-artifact provenance columns and sparse raw foreign-flow columns could be merged into the feature frame in a way that dropped historical training rows.

The loader now:

- excludes foreign-flow provenance columns from ticker feature joins
- fills missing numeric `foreign_*` source columns with `0.0`
- preserves `foreign_flow_context_available` and `foreign_flow_context_missing` so missing fallback zeros remain distinguishable from measured zeros

This is a data-governance safeguard. It does not mark missing context as available and does not remove any features automatically.

## Limitations

- `data/foreign_flow_curated.csv` is ignored and was not committed.
- Provider rows cover 2025-01-02 through 2025-01-24, not the full January 2025 requested business-date calendar.
- Local OHLCV coverage in the audit window has 17 dates per ticker.
- Context coverage diagnostics measure feature-frame rows across training and evaluation windows, not only final forecast dates.
- Provider coverage does not prove release-time availability, leakage absence, causality, or trading performance.
- The local runtime still logged `vnstock_data_not_installed` for OHLCV fallback calls, even though the curated foreign-flow artifact exists.

## Next Recommended Task

Add a first-class walk-forward CLI option or configuration field for the foreign-flow artifact path. That would avoid temporary default-path substitution and make future provider-backed context audits reproducible without touching ignored local cache files.

## Follow-Up: Foreign-Flow Path Option

The follow-up branch `vsef-walkforward-foreign-flow-path-option` adds `--foreign-flow-path` to the walk-forward all-model evaluation script. Future provider-backed audits can pass `data/foreign_flow_curated.csv` or another governed artifact directly, and run metadata records the supplied path and validation summary.

## Follow-Up: 10-Year Walk-Forward Audit

`docs/audits/VSEF_10Y_WALKFORWARD_AUDIT.md` rechecked the same local `data/foreign_flow_curated.csv` artifact against the requested 2015-01-01 through 2025-12-31 walk-forward window. The artifact remained `partial_coverage` with 68 matched ticker/date rows out of 11480 requested ticker/date pairs, so the optional long-window foreign-flow walk-forward was skipped and `foreign_*` interpretation remains conservative.
