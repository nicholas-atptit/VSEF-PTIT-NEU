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
- Daily data is not accepted.
- Daily-to-hourly resampling is not accepted.
- Old VN100 evidence is not accepted.

## Required Schema

Every record must contain:

| column | requirement |
| --- | --- |
| timestamp | Hourly bar timestamp parseable as a datetime. |
| ticker | Frozen VN30 ticker. |
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

## Validation Checklist

- Confirm the file is CSV or Parquet.
- Confirm all required columns exist.
- Confirm exactly the frozen 30 VN30 tickers are present.
- Confirm timestamp parsing succeeds.
- Confirm timestamps are hourly-aligned in Vietnam exchange time.
- Confirm per-ticker rows are sorted by timestamp.
- Confirm no duplicate `(ticker, timestamp)` rows exist.
- Confirm OHLCV values are finite and valid.
- Confirm training coverage is present for every ticker.
- Confirm evaluation coverage is present for every ticker.
- Confirm missing trading-hour gaps are explainable.
- Confirm corporate action adjustment policy is documented.

## Acceptable CSV Schema Example

```csv
timestamp,ticker,open,high,low,close,volume,adjusted_close,source,exchange,session,corporate_action_flag
2005-01-03 09:00:00,ACB,12.10,12.25,12.05,12.20,125000,12.20,vendor_name,HOSE,continuous,
2005-01-03 10:00:00,ACB,12.20,12.30,12.15,12.25,98000,12.25,vendor_name,HOSE,continuous,
```

## Unacceptable Cases

- Missing any frozen VN30 ticker.
- Using daily bars or daily-to-hourly synthetic bars.
- Ticker history starts in 2024, 2025, or 2026 for the full-history benchmark.
- Evaluation ends before `2026-05-31 23:59:59` without an exchange-calendar explanation.
- Duplicate `(ticker, timestamp)` rows remain unresolved.
- Zero, negative, missing, or non-numeric OHLC prices.
- Negative or missing volume.
- Mixed adjusted and unadjusted price series.
- Reusing VN100 evidence or local available-window evidence as a substitute for the external 2005-2026 hourly dataset.
