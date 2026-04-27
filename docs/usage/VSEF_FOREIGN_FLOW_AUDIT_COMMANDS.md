# VSEF Foreign-Flow Audit Commands

Date: 2026-04-27

Branch: `vsef-walkforward-foreign-flow-path-option`

## Purpose

This note records the supported command pattern for running walk-forward governance audits with an explicit curated foreign-flow artifact.

The command is for data-governance review only. It does not add model families, change model governance status, or claim trading-performance improvement.

## Walk-Forward Audit With Curated Foreign Flow

Use `--foreign-flow-path` to pass a governed artifact without replacing the ignored default `data/foreign_flow.csv`:

```bash
python scripts/run_walkforward_all_models_stacking_eval.py --tickers SSI FPT ACB HPG --history-start 2020-12-21 --history-end 2025-02-28 --initial-train-start 2020-12-21 --initial-train-end 2024-12-31 --forecast-start 2025-01-02 --forecast-end 2025-01-24 --horizons short_5d --step-sizes 1 --algorithms cart,xgboost,lightgbm --foreign-flow-path data/foreign_flow_curated.csv --output-dir outputs/walkforward_governance_audit_foreign_flow_real --max-workers 1 --max-depth 3 --meta-min-samples 1 --epochs 1
```

If `--foreign-flow-path` is omitted, the runner keeps the existing default foreign-flow loader behavior.

If a supplied path is missing, the run should fail clearly instead of silently falling back to the default fixture/cache path.

## Metadata

The run metadata records:

- supplied foreign-flow path
- whether the path was explicit
- loaded row count
- loader source name and provenance
- artifact validation result for the forecast window

## Interpretation

Provider-backed rows are evidence only for the ticker/date pairs they cover. A `partial_coverage` artifact can be useful for source-row governance checks, but it should not be treated as broad foreign-flow coverage or trading-performance evidence.

Generated provider CSVs and walk-forward output folders should remain untracked unless explicitly approved for repository inclusion.
