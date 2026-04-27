# VSEF OHLCV Cache 10-Year Availability Audit

Date: 2026-04-27

Branch: `vsef-ohclv-cache-10y-availability`

## Purpose

This audit documents local OHLCV cache availability for the requested 2015-01-01 through 2025-12-31 walk-forward baseline window.

This is a data-availability and governance audit only. It does not add model families, change model governance status, or claim trading-performance improvement.

## Why This Audit Was Needed

The previous 10-year walk-forward audit ran the requested commands, but the local fallback data did not cover the full requested history. For `SSI`, `FPT`, `ACB`, and `HPG`, the walk-forward runner filtered the local cache to 1256 rows per ticker for the requested `history_end=2025-12-31`.

The result was a maximum-available-window audit, not a true 10-year empirical baseline.

## Loading Path Findings

OHLCV cache path:

```text
data/daily_market_split_data/<TICKER>.csv
```

Findings:

| question | finding |
| --- | --- |
| fallback CSV location | `data/daily_market_split_data` |
| fallback file discovery | exact uppercase ticker filename, for example `SSI.csv` |
| fallback loader | `src/ml/data_loader.py::load_ohlcv_from_csv` |
| all-model walk-forward source order | `VnstockAdapter.get_ohlcv()` first, then CSV fallback |
| VN100 loader default source order | CSV first, then TimescaleDB, then vnstock |
| active canonical provider | `vnstock_data` is not installed in this runtime |
| alternate provider package | `vnstock` is installed, but it is not the accepted canonical source |
| existing daily cache extraction script | `scripts/extract_daily_csv.py` |
| existing DB backfill scripts | `scripts/run_backdate.py`, `scripts/sync_all_data.py` |
| exact local cache cause | local CSV cache is truncated before 2020-12-21 for these tickers |
| source limitation proven | not proven; provider was unavailable, so this audit cannot distinguish provider history limits from local cache truncation |

The all-model walk-forward runner does not use the `VN100DataLoader` source order. Its `_fetch_history()` method tries the provider adapter first, then falls back to local CSV if the provider returns empty.

## Cache Tracking Status

The repository currently tracks many files under `data/daily_market_split_data`; `git ls-files` reports 1706 tracked files in that directory. The four ticker files audited here are tracked:

- `data/daily_market_split_data/SSI.csv`
- `data/daily_market_split_data/FPT.csv`
- `data/daily_market_split_data/ACB.csv`
- `data/daily_market_split_data/HPG.csv`

By contrast, generated runtime folders and provider/context artifacts such as `outputs/`, `artifacts/`, `data/foreign_flow_curated.csv`, and `data/foreign_flow.csv` are ignored.

Because the daily cache is large and partly tracked, cache refreshes should not be committed automatically. A refresh should first produce provenance, coverage diagnostics, and an explicit commit decision.

## Audit Script

This branch adds a read-only script:

```text
scripts/audit_ohlcv_cache_coverage.py
```

The script reports per ticker:

- source file path
- fallback file presence
- row count
- date column
- min and max dates
- requested business-day count
- matched requested dates
- missing requested dates
- requested-window coverage rate
- whether the file can support the requested window
- provider module availability without fetching provider data

It does not fetch provider data, write CSVs, update caches, or modify input files.

Command run:

```bash
python scripts/audit_ohlcv_cache_coverage.py --tickers SSI,FPT,ACB,HPG --start-date 2015-01-01 --end-date 2025-12-31
```

Top-level result:

| field | value |
| --- | --- |
| data directory | `data/daily_market_split_data` |
| requested start | 2015-01-01 |
| requested end | 2025-12-31 |
| requested business-day count | 2870 |
| canonical provider | `vnstock_data` |
| `vnstock_data` available | false |
| alternate `vnstock` available | true |
| provider fetch attempted | false |
| missing file count | 0 |
| supporting ticker count | 0 |
| all tickers support requested window | false |

Per-ticker cache coverage:

| ticker | file present | row count | date min | date max | matched dates | missing dates | coverage rate | supports requested window |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| `SSI` | yes | 1306 | 2020-12-21 | 2026-03-20 | 1256 | 1614 | 0.437631 | no |
| `FPT` | yes | 1306 | 2020-12-21 | 2026-03-20 | 1256 | 1614 | 0.437631 | no |
| `ACB` | yes | 1306 | 2020-12-21 | 2026-03-20 | 1256 | 1614 | 0.437631 | no |
| `HPG` | yes | 1306 | 2020-12-21 | 2026-03-20 | 1256 | 1614 | 0.437631 | no |

The local cache does not contain true 10-year data for the requested baseline window.

## Refresh Path Assessment

Existing scripts can help, but none is a governed exact-window cache refresh workflow for this baseline yet.

Candidate paths:

| path | role | caveat |
| --- | --- | --- |
| `scripts/extract_daily_csv.py` | writes per-ticker daily CSVs using `VnstockAdapter.get_ohlcv()` | uses a lookback-day argument rather than explicit start/end dates; requires canonical `vnstock_data` |
| `scripts/run_backdate.py` | backfills historical OHLC into TimescaleDB | does not directly refresh `data/daily_market_split_data` for the all-model runner |
| `scripts/sync_all_data.py` | orchestrates `BackdateIngestor` over selected universes | database-focused; optional raw copies go to `data/raw/` |

Provider-backed local cache refresh planning command, once `vnstock_data` is installed and validated:

```bash
python scripts/extract_daily_csv.py --tickers SSI FPT ACB HPG --days 4200 --output data/daily_market_split
```

This is a planning command, not a command run in this audit. It would write into `data/daily_market_split_data` because the script appends `_data` when needed. A better follow-up would add or extend a refresh command with explicit `--start-date`, `--end-date`, output staging, provenance sidecar, and coverage validation before replacing tracked cache files.

## Recommended Cache Policy

Recommended policy for 10-year OHLCV cache refreshes:

- Stage provider outputs into an ignored scratch directory first, not directly over tracked cache files.
- Record provider package, provider source, retrieval timestamp, command, tickers, requested date range, and row counts.
- Run `scripts/audit_ohlcv_cache_coverage.py` against the staged cache.
- Only replace `data/daily_market_split_data/<TICKER>.csv` after coverage and schema validation pass.
- Do not commit large cache replacements without explicit approval.
- If a small curated fixture is needed for tests, keep it under `tests/fixtures/` and label it as synthetic or fixture-only.
- Do not treat alternate `vnstock` availability as canonical `vnstock_data` provider evidence.

## Recommended Next Command

After installing and validating canonical `vnstock_data`, run a staged refresh experiment for only the four audit tickers:

```bash
python scripts/extract_daily_csv.py --tickers SSI FPT ACB HPG --days 4200 --output tmp/ohlcv_10y_refresh_probe
python scripts/audit_ohlcv_cache_coverage.py --tickers SSI,FPT,ACB,HPG --start-date 2015-01-01 --end-date 2025-12-31 --data-dir tmp/ohlcv_10y_refresh_probe_data
```

If the staged cache reaches acceptable coverage, decide whether to replace tracked cache files or keep the refreshed data as an ignored local artifact for empirical runs.

## Limitations

- The audit uses business days as the requested-date denominator. Vietnamese market holidays are not modeled separately, so the coverage rate is a conservative availability proxy.
- Provider fetch was not attempted by the audit script.
- Canonical `vnstock_data` is not installed in this runtime, so provider-side historical availability was not proven.
- The script does not validate price adjustment quality or corporate-action adjustment status.
- The audit does not change walk-forward behavior or model governance.

## Validation

Focused tests were added in:

```text
tests/ml/test_ohlcv_cache_coverage.py
```

The tests use synthetic temporary CSVs only and cover:

- full coverage detection
- truncated coverage detection
- missing file detection
- row count and date range reporting
- no fetch or data modification
- multiple ticker handling
- CLI execution against a temporary cache directory

Validation commands run:

| command | result |
| --- | --- |
| `python -m pytest tests/ml/test_ohlcv_cache_coverage.py -q` | 7 passed |
| `python -m pytest tests/ml -q` | 174 passed, 2 skipped |
| `python -m pytest tests/quant_core -q` | 15 passed |
| `python -m pytest tests/phase1/test_forecast_contracts.py -q` | 3 passed |
| `python -m compileall src` | passed |

Validation warnings were non-fatal and included pytest cache write permission warnings for `.pytest_cache`, LightGBM/scikit-learn feature-name warnings, class-label warnings in ML tests, and logical-core detection fallback from joblib.

## 15-Year Daily Follow-Up

`docs/audits/VSEF_15Y_DAILY_WALKFORWARD_AUDIT.md` documents the follow-up staged provider refresh for `SSI`, `FPT`, `BVH`, and `VNM` over 2010-01-01 through 2025-12-31. That audit uses the new `--ohlcv-data-dir` runner option to consume ignored staged OHLCV files directly, leaving `data/daily_market_split_data/` untouched.
