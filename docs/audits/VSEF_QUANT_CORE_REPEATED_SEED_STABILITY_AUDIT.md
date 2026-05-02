# VSEF Quant Core Repeated-Seed Stability Audit
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Audit report |
| Created / authored | Thursday, 2026-04-30 15:55:58 ICT (UTC+07:00) |
| Last updated | Thursday, 2026-04-30 16:38:10 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `7b44ab41d3c23122ba52bb08a5e95c2bdfbfd647` |
| Timestamp source | Local repeated-seed stability runner validation |
| Status | Active |

## Purpose

This audit documents the governed repeated-seed Quant Core stability runner. The runner is intended to execute the existing Quant Core orchestration path repeatedly with different random seeds, preserve each seed's local outputs in a seed-specific directory, and aggregate stability diagnostics from the real generated CSV files.

Repeated-seed training is a stability diagnostic. It must not be used to select the single best random seed based on test-period performance.

The audit does not claim improved accuracy, production trading readiness, robust edge, or final buy recommendations.

## Why Repeated-Seed Training Is Useful

Repeated-seed execution helps identify whether model, policy, consensus, and decision-lane outputs are materially sensitive to random initialization or randomized training steps. Stable results across seeds are more defensible than a single lucky run, while unstable results should be treated as a governance warning.

The stability runner should support future 1000-seed smoke diagnostics, but 1000 seeds are never the default and should only run when explicitly requested with `--seed-count 1000`.

## Scope

In scope:

- wrapper CLI for repeated-seed Quant Core execution
- real per-seed calls to `scripts/run_quant_core.py`
- seed-specific output directories such as `seed_000001`
- resume detection based on minimum completion files
- failure logging without deleting failed seed folders
- aggregation from available real seed output CSV files
- dry-run command visibility

Out of scope:

- changing model families
- changing model governance status
- selecting or promoting a best seed
- running heavy 15-year or full-market jobs
- committing generated artifacts under `artifacts/`, `outputs/`, or `tmp/`

## How the Runner Works

The repeated-seed CLI is:

```powershell
scripts/run_quant_core_repeated_seeds.py
```

For each seed, it builds a real Quant Core command against:

```powershell
scripts/run_quant_core.py
```

Each seed writes into a deterministic directory:

```text
<output-dir>/seed_000001/
<output-dir>/seed_000002/
<output-dir>/seed_000003/
```

The per-seed command includes `--seed`, `--output-dir`, `--preset`, `--run-mode`, and `--max-workers`. It forwards optional `--models`, `--model-roles`, `--horizons`, `--target-types`, and `--no-ensemble` when provided.

The runner does not synthesize training metrics or write fake seed CSVs to pretend training succeeded. Root-level aggregate files are built only from actual seed output files that exist after completed or resumed-complete runs.

## Resume Behavior

A seed is considered complete only when all minimum completion files exist in the seed directory:

```text
run_manifest.json
forecast_summary.csv
model_execution_log.csv
```

With resume enabled, completed seeds are skipped. Incomplete seeds are rerun. Failed seed folders are not deleted automatically.

By default, execution continues after a failed seed and records the failure in `seed_execution_log.csv` and `failure_summary.csv`. `--stop-on-failure` stops after the first failed seed. `--dry-run` prints commands and does not execute Quant Core or create seed output directories.

## Output Files

The repeated-seed root output directory supports:

| File | Purpose |
| --- | --- |
| `repeated_seed_manifest.json` | Run configuration, status counts, and root output paths. |
| `seed_execution_log.csv` | One row per attempted, skipped, failed, or dry-run seed. |
| `seed_artifact_inventory.csv` | Expected per-seed artifact existence and row-count inventory. |
| `seed_forecast_summary.csv` | Concatenated per-seed `forecast_summary.csv` rows. |
| `model_seed_stability.csv` | Seed stability by `model_name` and `target_type`. |
| `model_horizon_seed_stability.csv` | Seed stability by `model_name`, `horizon`, and `target_type`. |
| `strategy_seed_stability.csv` | Strategy stability by available policy dimensions. |
| `model_health_seed_stability.csv` | Model-health stability by `model_name`. |
| `consensus_seed_stability.csv` | Consensus stability by available horizon, target, and run-mode dimensions. |
| `decision_candidate_seed_stability.csv` | Decision-candidate count and optional numeric metric stability. |
| `failure_summary.csv` | Failure counts and short failure-message summaries. |
| `summary.md` | Human-readable repeated-seed run summary. |

Missing optional seed CSV files do not crash aggregation. The affected aggregate file is emitted empty or with only available columns.

## Metrics Aggregated

For numeric metrics that are present, the runner computes:

```text
seed_count
mean
std
min
max
p05
p25
p50
p75
p95
```

Relevant metrics include:

```text
rmse
mae
directional_accuracy
f1
observations
sharpe
cagr
max_drawdown
total_return
win_rate
trade_count
agreement_score
disagreement_score
dispersion_score
sign_conflict_rate
active_signal_share
```

## Dry-Run Command

```powershell
C:\Users\luong\.venv\Scripts\python.exe scripts/run_quant_core_repeated_seeds.py ^
  --seed-start 1 ^
  --seed-count 3 ^
  --preset smoke ^
  --run-mode baseline_only ^
  --dry-run
```

Expected behavior:

- prints output directory
- prints seed count
- prints one real Quant Core command per seed
- does not execute training
- does not create seed output folders
- does not stage generated artifacts

## Local Validation Results

The 3-seed dry-run completed successfully and printed three real Quant Core commands. It did not create `artifacts/quant_core_repeated_seed`.

The tiny two-seed real validation command was run with:

```powershell
C:\Users\luong\.venv\Scripts\python.exe scripts/run_quant_core_repeated_seeds.py ^
  --seed-start 1 ^
  --seed-count 2 ^
  --preset smoke ^
  --run-mode baseline_only ^
  --output-dir artifacts\quant_core_repeated_seed_smoke_test ^
  --max-workers 1 ^
  --resume
```

The first sandboxed attempt recorded both seeds as failed because provider license verification was blocked by local sandbox network permissions. The runner preserved failure records in `seed_execution_log.csv` and `failure_summary.csv`.

The same command completed with normal local access. A second run with `--resume` skipped both completed seeds, confirming that minimum completion files are used for resume detection.

Verified local smoke outputs:

```text
artifacts\quant_core_repeated_seed_smoke_test\seed_execution_log.csv
artifacts\quant_core_repeated_seed_smoke_test\model_seed_stability.csv
artifacts\quant_core_repeated_seed_smoke_test\summary.md
artifacts\quant_core_repeated_seed_smoke_test\seed_000001\run_manifest.json
artifacts\quant_core_repeated_seed_smoke_test\seed_000001\forecast_summary.csv
artifacts\quant_core_repeated_seed_smoke_test\seed_000001\model_execution_log.csv
```

The local generated smoke outputs remain under ignored `artifacts/` paths and are not intended for commit.

## Recommended 1000-Seed Smoke Command

Use this only when explicitly ready to run 1000 seeds:

```powershell
C:\Users\luong\.venv\Scripts\python.exe scripts/run_quant_core_repeated_seeds.py ^
  --seed-start 1 ^
  --seed-count 1000 ^
  --preset smoke ^
  --run-mode research_core ^
  --output-dir artifacts\quant_core_repeated_seed_1000_smoke ^
  --max-workers 1 ^
  --resume
```

## Later Heavier 1000-Seed Option

This is a later option, not a default smoke command:

```powershell
C:\Users\luong\.venv\Scripts\python.exe scripts/run_quant_core_repeated_seeds.py ^
  --seed-start 1 ^
  --seed-count 1000 ^
  --preset medium ^
  --run-mode decision_core ^
  --output-dir artifacts\quant_core_repeated_seed_1000_medium_decision_core ^
  --max-workers 1 ^
  --resume
```

## Interpretation Guide

Use repeated-seed outputs to assess stability bands, not to choose the best-performing random seed. A defensible interpretation focuses on whether method-family patterns and policy behavior remain broadly consistent across seeds. Large seed-to-seed dispersion should be treated as uncertainty that requires additional investigation before making stronger claims.

Decision-lane candidates, if present, remain diagnostic outputs. They are not final buy recommendations.

## Limitations

- Smoke and medium presets are bounded orchestration checks, not proof of robust market edge.
- Local provider availability, cached data, and environment state can affect whether seed runs complete.
- Empty optional artifacts can be valid in narrow smoke modes.
- Aggregates only summarize numeric columns available in saved seed CSVs.
- Parallel seed execution can increase resource contention; the conservative default is `--max-workers 1`.

## Recommended Next Task

After the dry-run and tiny two-seed validation pass, run a governed 1000-seed smoke diagnostic with `--preset smoke`, `--run-mode research_core`, `--max-workers 1`, and `--resume`. Review stability aggregates before considering any broader medium or decision-core run.
