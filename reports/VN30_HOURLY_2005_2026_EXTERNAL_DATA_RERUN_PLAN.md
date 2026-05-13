# VN30 Hourly 2005-2026 External Data Rerun Plan

## Place External Data

Place the vendor-provided hourly dataset in a new external-data directory, for example:

- `data/external/vn30_hourly_2005_2026/vn30_hourly_2005_2026.csv`
- `data/external/vn30_hourly_2005_2026/vn30_hourly_2005_2026.parquet`

The file must contain hourly rows for all 30 frozen VN30 tickers from `2005-01-01 00:00:00` through `2026-05-31 23:59:59`.

## Run Validator

```powershell
python scripts/research/validate_external_vn30_hourly_dataset.py data/external/vn30_hourly_2005_2026/vn30_hourly_2005_2026.csv
```

Expected outputs:

- `reports/generated/vn30_hourly_external_data_validation/vn30_external_hourly_validation.csv`
- `reports/generated/vn30_hourly_external_data_validation/vn30_external_hourly_validation.md`

Do not proceed if the validator reports fewer than 30 benchmark-usable tickers.

## Import Validated Data Into Project Cache

After validation passes, split the validated external file by ticker and write hourly cache files to:

`data/market_cache/vnstock_data/vn30/hourly/{TICKER}.csv`

Each ticker file should contain:

`datetime,ticker,open,high,low,close,volume`

Use the external vendor timestamp as `datetime` after conversion to Vietnam exchange time. Keep a separate copy of the original vendor export under `data/external/`; do not overwrite it.

## Rerun Audit

```powershell
python scripts/research/audit_vn30_hourly_coverage_2005_2026.py
```

The audit must show `benchmark_usable=true` for all 30 frozen VN30 tickers.

## Rerun Benchmark

```powershell
python scripts/research/run_vn30_hourly_benchmark_2005_2026.py
```

The benchmark may proceed only after the audit confirms all 30 tickers are benchmark-usable.

## Rerun Confidence Sweep

```powershell
python scripts/research/run_vn30_hourly_confidence_sweep_2005_2026.py
```

This reads official VN30 hourly predictions from `outputs/vn30_hourly_official_2005_2026_traincutoff/`.

## Rerun Regime Diagnostics

```powershell
python scripts/research/run_vn30_hourly_exante_regime_validation_2005_2026.py
```

Use only hourly prediction artifacts and ex-ante labels based on prior information.

## Rerun Cost/Slippage Proxy Diagnostics

```powershell
python scripts/research/run_vn30_hourly_cost_slippage_validation_2005_2026.py
```

These diagnostics remain proxy evidence and do not establish execution-ready trading readiness.

## Final Paper and DOCX Timing

Generate the final paper and DOCX only after:

- external hourly data validation passes for all 30 tickers;
- the 2005-2026 hourly audit passes all 30 benchmark-usable tickers;
- the official benchmark produces non-empty hourly predictions;
- confidence, regime, significance, and cost/slippage diagnostics have been regenerated from that benchmark;
- claim boundaries are updated to match the actual evidence.

## Blocking Warning

Do not write a full-history VN30 paper, DOCX, or benchmark claim if the validator fails all-30 usability. Missing external data must result in a missing-evidence report, not a fabricated paper.
