# VN30 Hourly 2005-2026 External Data Requirements

## Required Universe

Use the frozen VN30 universe in `configs/universes/vn30_constituents_frozen.csv`.

Required tickers:

`ACB, BID, CTG, DGC, FPT, GAS, GVR, HDB, HPG, LPB, MBB, MSN, MWG, PLX, SAB, SHB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VPL, VRE`

## Required Design

- Universe: frozen VN30, exactly 30 tickers.
- Frequency: hourly only.
- Required time range: `2005-01-01 00:00:00` to `2026-05-31 23:59:59`.
- Training period: `2005-01-01 00:00:00` to `2024-12-31 23:59:59`.
- Evaluation/comparison period: `2025-01-01 00:00:00` to `2026-05-31 23:59:59`.
- Market index role: market context features, benchmark comparison, and regime/market-state diagnostics; market indices are not stock target labels.
- Daily data is not accepted.
- Daily-to-hourly resampling is not accepted.
- Old VN100 evidence is not accepted.

## Stock Schema

Every stock record must contain:

| column | requirement |
| --- | --- |
| timestamp | Hourly bar timestamp parseable as a datetime. |
| ticker | Frozen VN30 ticker. |
| index_code | Empty for stock rows in the combined external file. |
| open | Numeric hourly open price. |
| high | Numeric hourly high price. |
| low | Numeric hourly low price. |
| close | Numeric hourly close price. |
| volume | Numeric hourly traded volume. |

Optional columns:

| column | expectation |
| --- | --- |
| adjusted_close | Split/dividend-adjusted close if supplied by the vendor. |
| source | Data vendor or feed identifier. |
| exchange | HOSE or source exchange label. |
| session | Trading session or auction/session marker. |
| corporate_action_flag | Split, dividend, listing, suspension, or adjustment marker. |

## Market Index Coverage

The market-index coverage rule is intentionally not the same for all indices:

- `VNINDEX` must cover the full stock research range from `2005-01-01 00:00:00` to `2026-05-31 23:59:59`.
- `VN30INDEX` must cover its official-use range from `2012-02-06 00:00:00` to `2026-05-31 23:59:59`.
- `VNXALL` must cover its official-use range from `2016-10-24 00:00:00` to `2026-05-31 23:59:59`.
- The common comparison/evaluation window remains `2025-01-01 00:00:00` to `2026-05-31 23:59:59` for stocks and all three indices.
- Missing pre-start `VN30INDEX` or `VNXALL` rows must not fail validation.
- Pre-start vendor-backfilled or vendor-reconstructed `VN30INDEX`/`VNXALL` rows are optional and must be labeled if supplied.

## Market Index Schema

Every index record must contain:

| column | requirement |
| --- | --- |
| timestamp | Hourly index bar timestamp parseable as a datetime. |
| index_code | One of `VNINDEX`, `VN30INDEX`, `VNXALL`. |
| open | Numeric hourly open index level. |
| high | Numeric hourly high index level. |
| low | Numeric hourly low index level. |
| close | Numeric hourly close index level. |
| volume | Numeric hourly traded volume or vendor-provided index volume. |

The combined external file must still include the `ticker` column; it should be empty for index rows.

Optional column:

| column | allowed values |
| --- | --- |
| source_status | `official_live`, `vendor_backfilled`, `vendor_reconstructed`, `unknown` |

For `VN30INDEX` and `VNXALL`, rows before their official start dates are optional. If present, pre-start rows must use `source_status=vendor_backfilled` or `source_status=vendor_reconstructed`. If absent, that is acceptable and must not fail readiness.

## Timezone Assumption

Timestamps are assumed to be in Vietnam local exchange time, Asia/Ho_Chi_Minh, with no daylight-saving shift. If the external vendor exports UTC or another timezone, convert to Vietnam exchange time before validation.

## Trading-Session Calendar

The dataset must cover every expected HOSE hourly trading bar for the requested period, excluding official non-trading days, holidays, and exchange closures. Auction-only or partial sessions must be documented through `session` or `corporate_action_flag` if available.

## Duplicate Timestamp Handling

There must be at most one row per `(ticker, timestamp)`. If duplicate vendor rows exist, resolve them before import and document the rule. The preferred rule is to keep the final corrected vendor bar, not an arbitrary duplicate.

## Missing-Hour Handling

Missing bars must not be silently filled. The validator reports coverage gaps by ticker. Gaps caused by official market closures should be documented by the vendor calendar; gaps during expected trading hours are blocking for benchmark usability unless they are rare and explicitly justified.

## OHLCV Numeric Validation

- `open`, `high`, `low`, and `close` must be finite numeric values.
- Prices must be strictly positive.
- `high >= open`, `high >= close`, `low <= open`, `low <= close`, and `high >= low`.
- `volume` must be finite and non-negative.
- Zero volume bars may be accepted only when the exchange/vendor records a valid no-trade hourly bar.

## Corporate Action Adjustment

The benchmark should use a consistent price basis across the full period. If adjusted OHLCV is available, use it consistently and document the vendor adjustment methodology. If only raw OHLCV is available, the split/dividend adjustment limitation must be disclosed before benchmark claims are made.

## Split/Dividend Adjustment Note

Large discontinuities caused by splits, stock dividends, bonus issues, or symbol changes must be identified. A price series that mixes adjusted and unadjusted bars is unacceptable.

## Minimum Benchmark-Usable Criteria

A ticker is benchmark-usable only if:

- It is one of the 30 frozen VN30 tickers.
- It has hourly bars covering the required 2005-2026 design.
- It has usable training rows through `2024-12-31 23:59:59`.
- It has usable evaluation rows from `2025-01-01 00:00:00` through `2026-05-31 23:59:59`.
- It has no duplicate `(ticker, timestamp)` rows.
- It has valid, positive OHLC prices and valid non-negative volume.
- It has no unexplained missing expected trading-hour gaps that would invalidate the benchmark.

The full VN30 2005-2026 design may proceed only when all 30 frozen tickers are benchmark-usable.

Index readiness passes only if:

- `VNINDEX` has hourly coverage from `2005-01-01 00:00:00` to `2026-05-31 23:59:59`.
- `VN30INDEX` has hourly coverage from `2012-02-06 00:00:00` to `2026-05-31 23:59:59`.
- `VNXALL` has hourly coverage from `2016-10-24 00:00:00` to `2026-05-31 23:59:59`.
- All three indices cover the common comparison/evaluation window from `2025-01-01 00:00:00` to `2026-05-31 23:59:59`.
- Missing pre-start `VN30INDEX`/`VNXALL` rows are not treated as failures.

Combined readiness passes only when stock readiness and corrected index readiness both pass.

## Validation Checklist

- Confirm the file is CSV or Parquet.
- Confirm all required columns exist.
- Confirm exactly the frozen 30 VN30 tickers are present.
- Confirm `VNINDEX`, `VN30INDEX`, and `VNXALL` are present with their corrected required start dates.
- Confirm timestamp parsing succeeds.
- Confirm timestamps are hourly-aligned in Vietnam exchange time.
- Confirm per-ticker rows are sorted by timestamp.
- Confirm per-index rows are sorted by timestamp.
- Confirm no duplicate `(ticker, timestamp)` rows exist.
- Confirm no duplicate `(index_code, timestamp)` rows exist.
- Confirm OHLCV values are finite and valid.
- Confirm training coverage is present for every ticker.
- Confirm evaluation coverage is present for every ticker and every required market index.
- Confirm optional pre-start `VN30INDEX`/`VNXALL` rows are labeled `vendor_backfilled` or `vendor_reconstructed`.
- Confirm missing trading-hour gaps are explainable.
- Confirm corporate action adjustment policy is documented.

## Acceptable CSV Schema Example

```csv
timestamp,ticker,index_code,open,high,low,close,volume,adjusted_close,source,exchange,session,corporate_action_flag,source_status
2005-01-03 09:00:00,ACB,,12.10,12.25,12.05,12.20,125000,12.20,vendor_name,HOSE,continuous,,official_live
2005-01-03 09:00:00,,VNINDEX,244.50,245.20,243.90,244.80,0,,vendor_name,HOSE,continuous,,official_live
2012-02-06 09:00:00,,VN30INDEX,470.10,471.00,468.90,470.60,0,,vendor_name,HOSE,continuous,,official_live
2016-10-24 09:00:00,,VNXALL,1000.10,1002.00,998.40,1001.20,0,,vendor_name,HOSE,continuous,,official_live
```

## Unacceptable Cases

- Missing any frozen VN30 ticker.
- Missing `VNINDEX` from the full 2005-2026 range.
- Missing `VN30INDEX` from `2012-02-06` onward.
- Missing `VNXALL` from `2016-10-24` onward.
- Treating absent pre-start `VN30INDEX`/`VNXALL` rows as a validation failure.
- Supplying pre-start `VN30INDEX`/`VNXALL` rows with `source_status=official_live` or `source_status=unknown`.
- Using daily bars or daily-to-hourly synthetic bars.
- Ticker history starts in 2024, 2025, or 2026 for the full-history benchmark.
- Evaluation ends before `2026-05-31 23:59:59` without an exchange-calendar explanation.
- Duplicate `(ticker, timestamp)` rows remain unresolved.
- Zero, negative, missing, or non-numeric OHLC prices.
- Negative or missing volume.
- Mixed adjusted and unadjusted price series.
- Reusing VN100 evidence or local available-window evidence as a substitute for the external 2005-2026 hourly dataset.
