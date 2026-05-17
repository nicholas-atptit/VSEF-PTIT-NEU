# VN Market Directional Benchmark Lab

This repository preserves a VN market directional-benchmark research workspace. It focuses on reproducible evidence for directional classification and validation tracks, with explicit provider governance and archived generated artifacts.

It does not make a trading, profitability, execution, or investment-recommendation claim.

## Current Structure

- `src/data/providers/` - canonical provider/API gateway and request contract.
- `src/data/adapters/` - repository adapters around provider behavior and provenance.
- `scripts/research/` - active benchmark, validation, audit, and provider-diagnostic scripts.
- `scripts/legacy/` - preserved superseded scripts and older research utilities.
- `tests/data/` - provider-policy and provider-contract checks.
- `tests/ml/` - metric and benchmark helper checks.
- `reports/` - active evidence indexes, cleanup reports, claim registers, and audit summaries.
- `reports/generated/` - generated evidence outputs preserved after backup.
- `data/`, `outputs/`, and `archive/generated_data_snapshots/` - tracked backup data/artifact paths managed with Git LFS where applicable.

## Active Evidence Tracks

- Supported-index directional benchmark.
- VN30 daily 2015 benchmark and target-60 audit trail.
- VN30 hourly available-window benchmark and data forensics.
- VN30 hourly 2015 top-k ranking as a separate metric family.
- Provider standardization and provider/API adapter enforcement.

Start with:

- `reports/ACTIVE_EVIDENCE_INDEX.md`
- `reports/ACTIVE_CODE_MAP.md`
- `reports/RESEARCH_SCRIPT_STATUS.md`
- `reports/REPO_CLEANUP_INVENTORY.md`

## Provider/API Boundary

Normal fetch and benchmark code should use:

- `src.data.providers.vn_price_gateway.fetch_price_history`
- `src.data.providers.vn_provider_contract`

Raw `vnstock` or `vnstock_data` imports are allowed only in approved adapter, probe, diagnostic, or test locations. The policy check is:

```powershell
python scripts/check_provider_usage_policy.py
```

## Validation

Quick repository checks:

```powershell
python scripts/check_repo_hygiene.py
python scripts/check_runtime_preflight.py
.venv\Scripts\python.exe scripts\check_runtime_preflight.py
.venv\Scripts\python.exe scripts\check_provider_usage_policy.py
```

Targeted tests:

```powershell
.venv\Scripts\python.exe -m pytest tests\data\test_provider_usage_policy.py -q
.venv\Scripts\python.exe -m pytest tests\data\test_vn_price_gateway_contract.py -q
.venv\Scripts\python.exe -m pytest tests\ml\test_directional_accuracy_metrics.py -q
```

These commands are validation only. They do not run benchmarks, fetch market data, train models, or generate paper/DOCX artifacts.

## Git Notes

- Branch work should stay on the active research branch unless explicitly directed otherwise.
- Tags are not created by default.
- Do not use `git push --mirror` for normal branch publication.
- Preserve `data/`, `outputs/`, `reports/generated/`, and `archive/generated_data_snapshots/`.
