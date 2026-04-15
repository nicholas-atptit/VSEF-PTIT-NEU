# AI_ML_LLM-in-Stock

Research-oriented stock forecasting and evaluation framework for Vietnamese equities.

The repository now goes well beyond a single forecasting script. It supports real daily OHLCV fetching through `vnstock`, fixed-window and walk-forward evaluation, multi-horizon forward-return targets, dual-task regression and profit/loss classification, combined-signal analysis, regime-aware slicing, and selector experiments built on saved artifacts.

## Main Capabilities

- Real-data ingestion for Vietnam equities through `src/data/adapters/vnstock_adapter.py`
- Fixed-window backtest on daily OHLCV data with artifact persistence under `artifacts/backtest/`
- Multi-model comparison across `cart`, `xgboost`, `lightgbm`, and optional statistical models
- Forward-return forecasting for `3d`, `5d`, and `20d` trading-day horizons
- Dual-task evaluation:
  - regression for forward return
  - classification for profit/loss after costs
- Combined-signal analysis using `predicted_return` and `predicted_profit_probability`
- Regime-aware analysis for `bull`, `bear`, and `sideway`
- Walk-forward regime-aware robustness summaries across repeated folds
- Regime-conditioned and context-conditioned meta-selector layers for analysis-only candidate selection

## Quick Start

Run commands from the repository root.

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Run the fixed-window real-data backtest.

```bash
python scripts/run_backtest_real_data.py --tickers DGC ACB MWG HPG --train-start 2020-01-01 --train-end 2025-12-31 --eval-start 2026-01-01 --eval-end 2026-04-10 --output-dir artifacts/backtest
```

3. Run the forward-return workflow.

```bash
python scripts/run_backtest_forward_return.py --tickers DGC ACB MWG HPG --train-start 2020-01-01 --train-end 2025-12-31 --eval-start 2026-01-01 --eval-end 2026-04-10 --horizons 3d,5d,20d --output-dir artifacts/backtest_forward_return
```

4. Run the dual-task workflow and downstream analysis layers.

```bash
python scripts/run_dual_task_backtest.py --tickers DGC ACB MWG HPG --train-start 2020-01-01 --train-end 2025-12-31 --eval-start 2026-01-01 --eval-end 2026-04-10 --horizons 3d,5d,20d --output-dir artifacts/dual_task
python scripts/run_combined_signal_analysis.py --dual-task-dir artifacts/dual_task --output-dir artifacts/combined_signal
python scripts/run_regime_aware_analysis.py --dual-task-dir artifacts/dual_task --combined-signal-dir artifacts/combined_signal --output-dir artifacts/regime_aware_analysis
```

## Main Workflows

| Workflow | Script | Primary Output |
| --- | --- | --- |
| Fixed-window real-data backtest | `scripts/run_backtest_real_data.py` | `artifacts/backtest/` |
| Fixed-window model comparison | `scripts/run_backtest_model_comparison.py` | `artifacts/backtest_model_comparison/` |
| Multi-horizon forward-return backtest | `scripts/run_backtest_forward_return.py` | `artifacts/backtest_forward_return/` |
| Strategy backtest layer | `scripts/run_strategy_backtest.py` | `artifacts/strategy_backtest/` |
| Dual-task backtest | `scripts/run_dual_task_backtest.py` | `artifacts/dual_task/` |
| Combined-signal analysis | `scripts/run_combined_signal_analysis.py` | `artifacts/combined_signal/` |
| Regime-aware analysis | `scripts/run_regime_aware_analysis.py` | `artifacts/regime_aware_analysis/` |
| Walk-forward robustness | `scripts/run_walk_forward_regime_robustness.py` | `artifacts/walk_forward_regime_robustness/` |
| Regime-conditioned meta-selector | `scripts/run_meta_selector.py` | `artifacts/meta_selector/` |
| Benchmark audit + context selector | `scripts/run_context_meta_selector.py` | `artifacts/meta_selector_audit/`, `artifacts/context_meta_selector/` |

## Documentation

- [docs/CHANGELOG_SUMMARY.md](docs/CHANGELOG_SUMMARY.md): how the system evolved and why each layer was added
- [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md): runnable commands, dependencies, and artifact locations
- [docs/ML_IMPLEMENTATION_GUIDE.md](docs/ML_IMPLEMENTATION_GUIDE.md): internal architecture, targets, trainer path, and no-leakage rules
- [docs/EVALUATION_WORKFLOWS.md](docs/EVALUATION_WORKFLOWS.md): purpose, metrics, and outputs for each evaluation workflow
- [docs/RESEARCH_FINDINGS_AND_LIMITATIONS.md](docs/RESEARCH_FINDINGS_AND_LIMITATIONS.md): current empirical findings, limits, and next steps

## Repository Structure

```text
src/data/adapters/vnstock_adapter.py      Real-market data access and schema normalization
src/ml/feature_engineering.py             Daily feature generation
src/ml/trainer.py                         Target creation, model training, inference, manifests
src/ml/backtest/                          Evaluation and analysis workflows
scripts/run_*.py                          CLI entry points
artifacts/                                Saved workflow outputs
docs/                                     User-facing documentation
tests/ml/                                 Focused ML and workflow tests
```

## Current Status

The repository is a research and evaluation stack, not a live execution system.

- Forecasting quality improves in some slices, but there is no universal winner across all horizons and regimes.
- The combined-method family is more repeatable than any single exact model configuration.
- Profit classification and selector layers remain sample-sensitive.
- Bear-regime evidence is still relatively sparse in the saved walk-forward runs.

That makes the repo useful for disciplined comparative research, benchmark auditing, and strategy hypothesis testing. It does not justify strong live-trading claims without more history, more bearish samples, and better calibration.
