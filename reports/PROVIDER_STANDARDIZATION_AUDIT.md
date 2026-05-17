# Provider Standardization Audit

## Scope

Branch: `research/vn100-evidence-hardening-v1`

Starting commit: `94da12860f483545865ef98717057193418c32a9`

This audit covers direct `vnstock_data` imports, direct `vnstock` imports,
`Quote.history` style calls, research scripts that fetch data, the current
adapter surface, provider tests, and runtime preflight provider checks.

## Classification

| File | Classification | Notes |
| --- | --- | --- |
| `src/data/adapters/vnstock_adapter.py` | canonical provider layer | Existing low-level adapter centralizes lazy `vnstock_data` access and already wraps `Quote.history` / `QuoteHistory.history`. |
| `src/data/providers/vn_provider_contract.py` | canonical provider layer | Added contract types and supported-code constants. |
| `src/data/providers/vn_price_gateway.py` | canonical provider layer | Added approved stock/index OHLCV gateway. |
| `scripts/research/probe_vnstock_supported_indices.py` | allowed provider probe | Raw provider calls are permitted because this script probes supported index codes. |
| `scripts/research/probe_vnstock_hourly_provider_capability.py` | allowed provider probe | Raw provider calls are permitted because this script probes provider capability. |
| `scripts/research/verify_vnstock_data_environment.py` | allowed provider probe | Environment/import verification only. |
| `scripts/research/verify_repo_vnstock_provider_paths.py` | allowed provider probe | Provider-path verification only. |
| `scripts/research/diagnose_vn30_vnstock_hourly_fetch_failures.py` | allowed provider probe | Diagnostic raw-provider failure investigation. |
| `tests/test_vnstock_adapter.py` | allowed provider test | Adapter behavior tests. |
| `tests/test_vnstock_news.py` | allowed provider test | Live/best-effort provider contract probe test. |
| `tests/data/test_vn_price_gateway_contract.py` | allowed provider test | New offline gateway contract tests. |
| `tests/data/test_provider_usage_policy.py` | allowed provider test | New static policy tests. |
| `scripts/research/vn30_hourly_vnstock_common.py` | violation: direct provider bypass | Shared VN30 fetch helper directly enumerated raw providers before this task. |
| `scripts/research/fetch_vnstock_supported_indices_hourly.py` | violation: direct provider bypass | Index fetch script directly called raw `Quote.history` before this task. |
| `scripts/research/fetch_vn30_hourly_from_vnstock_2005_2026.py` | violation: direct provider bypass by dependency | The script fetched through `vn30_hourly_vnstock_common.py`, which bypassed the adapter before this task. |
| `scripts/research/fetch_vn30_hourly_listing_aware_from_vnstock.py` | violation: direct provider bypass by dependency | The script fetched through `vn30_hourly_vnstock_common.py`, which bypassed the adapter before this task. |
| `scripts/research/validate_fetched_vn30_hourly_2005_2026.py` | documentation/validation only | Validates cache and benchmark gate; no raw provider calls. Needed frequency metadata validation. |
| `scripts/research/validate_vn30_hourly_listing_aware_dataset.py` | documentation/validation only | Validates listing-aware cache and benchmark gate; no raw provider calls. Needed frequency metadata validation. |
| `src/data/historical/hourly_service.py` | violation: direct provider bypass | Direct hourly OHLCV call outside provider gateway before this task. |
| `scripts/check_rest_price.py` | violation: direct provider bypass | Direct raw provider smoke check before this task. |
| `src/data/historical/backdate.py` | remaining legacy direct provider path | Daily historical backdate path; not migrated in this VN30 hourly task. |
| `src/api/streaming/fallback.py` | remaining legacy direct provider path | Streaming fallback; outside VN30 hourly standardization scope. |
| `src/api/streaming/producers/market_data_producer.py` | remaining legacy direct provider path | Streaming producer; outside VN30 hourly standardization scope. |
| `scripts/run_vn100_hybrid_frequency_accuracy_benchmark.py` | remaining legacy direct provider path | Benchmark path; not run or migrated in this task. |
| `scripts/discover_vn100.py` | remaining legacy direct provider path | Listing discovery, not OHLCV price history. |
| `scripts/extract_market_csv_single.py` | remaining legacy direct provider path | Listing extraction, not OHLCV price history. |
| `scripts/extract_market_csv_per_ticker.py` | remaining legacy direct provider path | Listing extraction, not OHLCV price history. |
| `scripts/extract_llm_jsonl.py` | remaining legacy direct provider path | Listing extraction, not OHLCV price history. |
| `scripts/check_runtime_preflight.py` | runtime preflight provider check | Checks importability of `vnstock_data` and legacy `vnstock`. |

## Findings

- VN30 hourly research fetches shared a direct-provider helper instead of using
  the repository adapter/provider path.
- Index fetch logic used raw package imports and ad hoc normalization.
- Optional index codes still referred to `VN30INDEX` and `VNXALL`, both unsafe
  for current provider assumptions.
- Normalized hourly caches did not require explicit `frequency` metadata.
- Benchmark scripts already consume validation outputs, but the provider layer
  did not provide a static guard against future raw-provider bypasses.

## Decision

Normal stock/index OHLCV fetching must use
`src.data.providers.vn_price_gateway.fetch_price_history`. Raw provider calls
remain allowed only in the low-level adapter/gateway, provider probes, provider
tests, and documented legacy paths that were not in the VN30 hourly migration
scope.
