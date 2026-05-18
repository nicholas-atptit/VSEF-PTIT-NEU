# VN30 Daily 2015 BCM/VIB Recovery Report

- Created at UTC: 2026-05-17
- Track: VN30 Daily 2015
- Branch: research/vn100-evidence-hardening-v1
- Tag: nckh/vn30-daily-2015-bcm-vib-recovery-v1

## Problem

BCM and VIB were missing from the daily cache after the initial fetch run (28/30 usable). Both tickers returned raw data from the provider (vnstock_data/KBS) but the canonical gateway's strict OHLCV validation rejected all data due to a small number of rows with inconsistent OHLCV geometry.

## Diagnosis

- **BCM**: 2033 raw rows, 7 problematic (0.3%). Issues: high < max(open/close) on 5 rows, low > min(open/close) on 2 rows.
- **VIB**: 2324 raw rows, 2 problematic (0.1%). Issues: low > min(open/close) on 2 rows.

## Recovery

Updated `fetch_vn30_daily_gateway_2015.py` with fallback path that:
1. Tries canonical gateway first (strict validation)
2. On OHLCV geometry failure, fetches raw data directly
3. Filters out bad rows, retains valid ones
4. Saves if >= 100 rows remain

Results:
- BCM: 2026 rows retained (2018-02-21 to 2026-05-15)
- VIB: 2322 rows retained (2017-01-09 to 2026-05-15)

## Post-Recovery Readiness

- Daily universe: **30/30 tickers usable**
- Validation: passed
- Readiness manifest: rebuilt

## Benchmark Rerun

Benchmark rerun with full 30-ticker universe:
- Best model/horizon: **XGBoost h=1**
- Final accuracy: **55.84%** (9,604 rows, 30 tickers)
- Baseline: 50.00%
- Lift: +5.84%
- 60% threshold passed: **no** (all experiments claim_level=failed)

## Audit

- Audit: **passed**
- All 30 tickers represented in predicted_vs_actual
- No hourly resampling used
- No trading/profitability claims

## Constraints Maintained

- No new branch created
- main not touched
- No paper/DOCX generated
- Daily track only, no hourly mixing
- No daily-to-hourly resampling
