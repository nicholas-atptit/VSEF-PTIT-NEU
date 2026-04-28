# VSEF – Vietnam Stock Evaluation and Forecasting Framework

VSEF is a research-oriented framework for Vietnamese stock forecasting, evaluation, regime analysis, and strategy backtesting. It is designed for comparative empirical work rather than live order execution. The framework uses real market data through `vnstock_data`, supports multiple forecast model families, and emphasizes walk-forward validation, selector-based evaluation, risk-aware interpretation, and regime-conditioned analysis.

The current codebase supports:

- real Vietnamese market data ingestion through `vnstock_data` and the repository adapter layer
- multi-model forecasting across baseline, statistical, tree-based, boosting, linear, and sequence models
- risk-aware evaluation using tail-risk, drawdown, systemic-risk, and volatility features
- regime-aware analysis for bull, bear, sideway, high-volatility, and crisis contexts
- strategy backtesting with transaction-cost-aware evaluation surfaces
- walk-forward validation, combined-signal analysis, and selector-based model evaluation

**Important warning:** The repository is a research and evaluation framework, not a finished live-trading execution system. Backtest evidence and saved artifacts should not be presented as guaranteed trading performance.

## Repository Access and License Status

This repository is private and proprietary. It is not open-source.

Access to this repository does not grant permission to copy, modify, redistribute, publish, deploy, sublicense, or reuse any part of the codebase, documentation, models, scripts, or research artifacts.

See [LICENSE](LICENSE) and [SECURITY.md](SECURITY.md) for details.

## Current Model Coverage

| Area | Models / Methods | Current status |
| --- | --- | --- |
| Baseline forecasting | Naive, Moving Average | Implemented |
| Classical time series | SARIMAX, ETS | Implemented |
| Linear models | Linear Regression, Ridge, Lasso | Present, needs stronger evaluation and feature-selection review |
| Tree-based models | CART, Random Forest | Implemented |
| Boosting models | XGBoost, LightGBM | Primary research models |
| Deep learning | LSTM, BiLSTM | Implemented, needs stronger training/evaluation hardening |
| Ensemble | Stacking, combined-signal methods, selector layers | Implemented for research evaluation |
| Risk models | Historical VaR/CVaR, Monte Carlo VaR/CVaR, CoVaR, Delta-CoVaR, drawdown, GARCH | Implemented |
| Regime detection | rule-based high-vol/crisis detection, Markov-switching bull/bear/sideway detection | Implemented, needs further improvement |
| Portfolio/risk overlay | Risk-aware allocation and exposure penalties | Implemented as research logic |
| Reinforcement learning | DQN/PPO/A2C not yet implemented | Planned research extension |

## Important Algorithm Gap Notice

Compared with the original full algorithm checklist, the repository does not yet clearly implement or fully govern:

- GRU
- TCN
- Transformer
- TFT
- N-BEATS / N-HiTS
- CatBoost
- EGARCH / GJR-GARCH
- Reinforcement Learning methods such as DQN, PPO, and A2C

These models are future research extensions, not blockers for the VSEF name. Adding too many algorithms too early would create a model zoo rather than a stronger forecasting framework. The current priority is better evaluation depth, regime robustness, calibration, and reproducibility.

With the current codebase, the name **VSEF – Vietnam Stock Evaluation and Forecasting Framework** is defensible. A claim such as **state-of-the-art deep forecasting platform** is not yet defensible.

## Development Roadmap

### 1. Regime Layer Improvement

Planned improvements:

- strengthen bull, bear, and sideway detection
- compare Markov-switching outputs against threshold-based benchmark labels
- add regime stability metrics across walk-forward folds
- track regime-specific model performance more explicitly
- improve sparse bear-regime handling so bear conclusions are not overclaimed from too few observations

### 2. Linear Model Improvement

Planned improvements:

- re-evaluate Linear Regression, Ridge, and Lasso under the same feature sets used by boosting models
- add stronger feature-selection diagnostics
- compare coefficient stability across folds
- keep linear models as interpretable baselines and feature sanity checks

### 3. Deep Learning Improvement

Current deep learning coverage includes LSTM and BiLSTM, but it needs hardening.

Planned improvements:

- improve LSTM/BiLSTM sequence preparation, persistence, and evaluation consistency
- add GRU as a lighter recurrent baseline
- add TCN for convolutional time-series forecasting
- add Transformer-style forecasting only after sequence evaluation stability is verified
- evaluate TFT, N-BEATS, and N-HiTS as later-stage research candidates, not immediate production models

### 4. Advanced Risk Model Improvement

Planned improvements:

- extend the current GARCH layer toward EGARCH and GJR-GARCH
- compare volatility forecasting quality across GARCH-family variants
- track whether volatility forecasts improve regime detection and risk-aware allocation
- keep VaR/CVaR interpretation conservative

### 5. Reinforcement Learning Research Extension

Reinforcement learning should be added only after forecast, risk, and regime layers are stable.

Planned methods:

- DQN
- PPO
- A2C

Initial scope:

- treat RL as a policy-evaluation research layer
- use forecast, regime, and risk outputs as state inputs
- include transaction costs, slippage, position limits, and turnover penalties
- compare RL policies against threshold strategies and buy-and-hold baselines
- prevent lookahead leakage in environment design

## Safe Implementation Sequence

1. Improve Linear/Ridge/Lasso evaluation and diagnostics first.
2. Harden LSTM/BiLSTM sequence training and tests.
3. Add GRU as the first new deep learning model.
4. Add TCN only after GRU tests pass.
5. Add Transformer/N-BEATS/TFT only as later research candidates.
6. Add EGARCH/GJR-GARCH after checking current GARCH dependency and tests.
7. Add RL only as a separate experimental module, not part of core production inference.

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

- [docs/README.md](docs/README.md): documentation map and placement rules
- [docs/roadmap/VSEF_ROADMAP.md](docs/roadmap/VSEF_ROADMAP.md): conservative roadmap for model, regime, risk, and RL extensions
- [docs/audits/VSEF_MODEL_GAP_AUDIT.md](docs/audits/VSEF_MODEL_GAP_AUDIT.md): audit of implemented, mentioned, registered, and missing model coverage
- [docs/archive/cleanup/2026-04-15_Wednesday__CHANGELOG_SUMMARY.md](docs/archive/cleanup/2026-04-15_Wednesday__CHANGELOG_SUMMARY.md): how the system evolved and why each layer was added
- [docs/usage/USAGE_GUIDE.md](docs/usage/USAGE_GUIDE.md): runnable commands, dependencies, and artifact locations
- [docs/usage/ML_IMPLEMENTATION_GUIDE.md](docs/usage/ML_IMPLEMENTATION_GUIDE.md): internal architecture, targets, trainer path, and no-leakage rules
- [docs/EVALUATION_WORKFLOWS.md](docs/EVALUATION_WORKFLOWS.md): purpose, metrics, and outputs for each evaluation workflow
- [docs/archive/root/2026-04-15_Wednesday__RESEARCH_FINDINGS_AND_LIMITATIONS.md](docs/archive/root/2026-04-15_Wednesday__RESEARCH_FINDINGS_AND_LIMITATIONS.md): current empirical findings, limits, and next steps

## Repository Structure

See [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md) for the repository layout policy and placement rules.

```text
README.md, LICENSE, SECURITY.md, pyproject.toml
config/                                  Runtime and project configuration
src/data/adapters/vnstock_adapter.py      Real-market data access and schema normalization
src/forecast/                             Governed forecast model surface and model registry
src/ml/models/                            Trainable ML model implementations used by ML workflows
src/ml/feature_engineering.py             Daily feature generation
src/ml/trainer.py                         Target creation, model training, inference, manifests
src/ml/backtest/                          Evaluation, strategy, selector, and analysis workflows
src/regime/                               Markov-switching regime model and regime-label helpers
src/risk/                                 Standalone risk models including GARCH, VaR/CVaR, drawdown
scripts/run_*.py                          CLI entry points
artifacts/, outputs/, tmp/                Generated workflow outputs and scratch files, ignored by default
docs/                                     Documentation grouped by governance, audits, roadmap, archive
tests/ml/                                 Focused ML and workflow tests
```

## Current Status

The repository is a research and evaluation stack, not a live execution system.

- Forecasting quality improves in some slices, but there is no universal winner across all horizons and regimes.
- The combined-method family is more repeatable than any single exact model configuration.
- Profit classification and selector layers remain sample-sensitive.
- Bear-regime evidence is still relatively sparse in the saved walk-forward runs.
- Deep-learning and reinforcement-learning claims should remain narrow until sequence evaluation, calibration, and governance tests are stronger.

That makes the repo useful for disciplined comparative research, benchmark auditing, and strategy hypothesis testing. It does not justify strong live-trading claims without more history, more bearish samples, better calibration, and stronger reproducibility evidence.
