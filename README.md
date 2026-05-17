# VN Market Directional Benchmark Lab

A research repository for Vietnamese market directional forecasting benchmarks, provider-standardized OHLCV data access, evidence tracking, and claim governance.

## What This Repo Is

- A benchmark lab for stock and index directional forecasting experiments.
- A provider gateway/API adapter layer for governed VN OHLCV access.
- A data forensics and reproducibility workspace.
- A stock/index directional research workspace with preserved evidence artifacts.

## What This Repo Is Not

- Not a live trading system.
- Not an investment recommendation engine.
- Not a profitability guarantee.
- Not a full 2015-start hourly stock benchmark.
- Not the old VSEF project identity anymore.

## Active Tracks

- Stock hourly available-window benchmark.
- Stock daily 2015 benchmark.
- Index directional benchmark.
- Data forensics and provider diagnostics.
- Top-k ranking as a separate metric family.

Start with:

- `reports/ACTIVE_EVIDENCE_INDEX.md`
- `reports/ACTIVE_CODE_MAP.md`
- `reports/REPO_CLEANUP_INVENTORY.md`
- `reports/CODE_CLEANUP_CHANGES.md`

## Claim Boundary

- Stock hourly available-window has baseline60 evidence, but final65 is not established.
- Stock daily 2015 is 30/30 usable but below 60.
- Index benchmark has exact pass60 results, separate from stock.
- Top-k ranking is not overall directional accuracy.
- No trading, profitability, live-deployment, or investment-recommendation claim is made.

## Provider/API Adapter

Normal fetch and benchmark code should go through:

- `src/data/providers/vn_price_gateway.py`
- `src/data/providers/vn_provider_contract.py`
- `src/data/adapters/vnstock_adapter.py`
- `scripts/check_provider_usage_policy.py`

Raw `vnstock` or `vnstock_data` imports are allowed only in approved adapter, probe, diagnostic, or test locations.

## Data And Artifact Policy

- `data/`, `outputs/`, `reports/generated/`, and `archive/generated_data_snapshots/` are preserved.
- Data and generated artifacts are tracked with Git LFS where applicable after the repository backup.
- Do not delete data, output artifacts, market cache, raw fetch data, or archive snapshots.
- Do not force-add secrets, raw `.env` files, credentials, tokens, virtual environments, or local caches.

## Validation

Preferred task runner:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/dev_tasks.ps1 -Task validate-all
```

If `make` is available:

```powershell
make validate-all
```

Raw commands:

```powershell
python scripts/check_repo_hygiene.py
python scripts/check_runtime_preflight.py
<repo-approved-venv>\Scripts\python.exe scripts\check_runtime_preflight.py
<repo-approved-venv>\Scripts\python.exe scripts\check_provider_usage_policy.py
<repo-approved-venv>\Scripts\python.exe -m pytest tests\data\test_provider_usage_policy.py -q
<repo-approved-venv>\Scripts\python.exe -m pytest tests\data\test_vn_price_gateway_contract.py -q
<repo-approved-venv>\Scripts\python.exe -m pytest tests\ml\test_directional_accuracy_metrics.py -q
```

These commands are validation only. They do not run benchmarks, fetch market data, train models, or generate paper/DOCX artifacts.

## Development Rules

- No tags by default.
- No `git push --mirror` for normal branch work.
- No benchmark rerun without a written protocol.
- No data fetch without a written protocol.
- No final-label tuning.
- No paper/DOCX generation unless explicitly requested.
- No claim without artifact support.
- Keep active research work on the current research branch unless explicitly directed otherwise.

See `docs/USAGE.md` and `docs/RESEARCH_WORKFLOW.md` for operating details.
