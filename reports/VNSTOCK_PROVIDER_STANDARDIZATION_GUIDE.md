# VNStock Provider Standardization Guide

## Why This Exists

The VN30 hourly research path previously allowed agents to bypass the
repository adapter, call random provider APIs directly, try unsupported index
codes, and confuse small sample support with full-history support. This guide
defines the canonical route for stock/index OHLCV data so benchmarks cannot run
on wrong, daily, resampled, or fabricated hourly data.

## Correct Venv

Run provider and validation commands with:

```powershell
C:\Users\luong\.venv\Scripts\python.exe
```

Do not rely on bare `python` unless `sys.executable` confirms the same venv.

## Provider Priority

Normal OHLCV fetches must use:

```python
from src.data.providers.vn_price_gateway import fetch_price_history
```

The gateway enforces this priority:

1. Repository adapter / `vnstock_data` path first.
2. Direct `vnstock_data` fallback second if the adapter cannot perform the operation.
3. Legacy `vnstock` fallback third only when `allow_legacy_fallback=True`.

Every attempted provider/source is recorded in response metadata or exception
attempts.

## Index Code Map

Allowed current provider codes:

```text
VNINDEX, HNXINDEX, UPCOMINDEX, VN30, HNX30, VN100
```

Do not use `VN30INDEX`; use `VN30`. Do not use `VNXALL` unless a future provider
probe proves support.

## Allowed And Disallowed APIs

Allowed:

- `src.data.providers.vn_price_gateway.fetch_price_history`
- `src.data.adapters.vnstock_adapter.py` as the low-level adapter
- Raw provider APIs only in provider probes and provider tests

Disallowed in normal fetch scripts:

- direct `vnstock_data` imports
- direct legacy `vnstock` imports
- direct `Quote` construction
- direct provider history calls
- daily fetches for hourly research
- daily-to-hourly resampling

## Correct OHLCV Fetch

Use `FetchRequest` with `frequency=Frequency.HOURLY`, explicit `start` and
`end`, and supported sources only:

```python
from src.data.providers.vn_provider_contract import AssetType, FetchRequest, Frequency, SourceName
from src.data.providers.vn_price_gateway import fetch_price_history

response = fetch_price_history(
    FetchRequest(
        symbol="FPT",
        asset_type=AssetType.STOCK,
        start="2026-01-01",
        end="2026-01-31",
        frequency=Frequency.HOURLY,
        preferred_sources=(SourceName.KBS, SourceName.VCI),
        allow_legacy_fallback=True,
        allow_daily=False,
        allow_resample=False,
    )
)
df = response.data
```

Normalized stock schema:

```text
datetime,ticker,open,high,low,close,volume,provider,source,frequency
```

Normalized index schema:

```text
datetime,index_code,open,high,low,close,volume,provider,source,frequency
```

## Validate Before Benchmark

Before any benchmark:

- Use the intended venv.
- Fetch through the gateway.
- Confirm all required symbols have rows.
- Confirm `frequency == "1H"`.
- Confirm no duplicate timestamp per symbol.
- Confirm numeric OHLCV, positive prices, non-negative volume, and high/low consistency.
- Confirm no forward fill, synthetic bars, daily data, resampling, or reused VN100 evidence.
- Confirm output directories are empty or explicitly versioned.

## Future-Agent Checklist

- Provider path is `src.data.providers.vn_price_gateway`.
- Index codes are from the supported set.
- `VN30INDEX` and `VNXALL` are not used.
- `allow_daily=False` and `allow_resample=False` for hourly research.
- Validation gate passes before benchmark.
- `scripts/check_provider_usage_policy.py` passes.
