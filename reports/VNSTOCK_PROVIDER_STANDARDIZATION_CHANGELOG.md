# VNStock Provider Standardization Changelog

## Files Created

- `src/data/providers/__init__.py`
- `src/data/providers/vn_provider_contract.py`
- `src/data/providers/vn_price_gateway.py`
- `scripts/check_provider_usage_policy.py`
- `tests/data/test_vn_price_gateway_contract.py`
- `tests/data/test_provider_usage_policy.py`
- `reports/PROVIDER_STANDARDIZATION_AUDIT.md`
- `reports/VNSTOCK_PROVIDER_STANDARDIZATION_GUIDE.md`
- `reports/VNSTOCK_PROVIDER_STANDARDIZATION_CHANGELOG.md`

## Files Modified

- `scripts/research/vn30_hourly_vnstock_common.py`
- `scripts/research/fetch_vnstock_supported_indices_hourly.py`
- `scripts/research/fetch_vn30_hourly_listing_aware_from_vnstock.py`
- `scripts/research/validate_fetched_vn30_hourly_2005_2026.py`
- `scripts/research/validate_vn30_hourly_listing_aware_dataset.py`
- `scripts/research/vn30_hourly_listing_aware_common.py`
- `src/data/historical/hourly_service.py`
- `scripts/check_rest_price.py`
- `scripts/check_repo_hygiene.py`
- `src/data/adapters/vnstock_adapter.py`

## Scripts Migrated

- `scripts/research/fetch_vnstock_supported_indices_hourly.py`
- `scripts/research/fetch_vn30_hourly_from_vnstock_2005_2026.py` through `vn30_hourly_vnstock_common.py`
- `scripts/research/fetch_vn30_hourly_listing_aware_from_vnstock.py` through `vn30_hourly_vnstock_common.py`
- `src/data/historical/hourly_service.py`
- `scripts/check_rest_price.py`

## Guardrails Added

- Canonical request/response contract with provider/source/frequency enums.
- Gateway validation for unsupported indices, unsupported aliases, source names,
  hourly frequency, daily fallback, and resampling.
- Normalized stock/index schemas with explicit `frequency`.
- OHLCV validation for datetime, duplicates, numeric values, price positivity,
  non-negative volume, and high/low/open/close consistency.
- Static provider usage policy check with line-level violations and replacement guidance.

## Raw-Provider Scripts Still Allowed

- `scripts/research/probe_vnstock_supported_indices.py`
- `scripts/research/probe_vnstock_hourly_provider_capability.py`
- `scripts/research/verify_vnstock_data_environment.py`
- `scripts/research/verify_repo_vnstock_provider_paths.py`
- `scripts/research/diagnose_vn30_vnstock_hourly_fetch_failures.py`

## Remaining Limitations

- Some legacy non-VN30-hourly paths still have direct provider calls and are
  explicitly allowlisted in `scripts/check_provider_usage_policy.py`.
- The repository hygiene check still fails on pre-existing local absolute path
  evidence files unrelated to this standardization.
- No live provider data was fetched in this task, so full-history provider
  availability remains a future data-gate concern.
