# VSEF Roadmap

This roadmap keeps VSEF positioned as a Vietnamese stock forecasting and evaluation research framework. It is not a live-trading execution roadmap, and it should not be read as a promise that any future model will improve realized trading performance.

## Current Priority

The near-term priority is stronger evaluation depth, regime robustness, calibration, and reproducibility. New algorithms should be added only when the existing training, persistence, walk-forward evaluation, and test surfaces can govern them cleanly.

## 1. Regime Layer Improvement

Planned improvements:

- strengthen bull, bear, and sideway detection
- compare Markov-switching outputs against threshold-based benchmark labels
- add regime stability metrics across walk-forward folds
- track regime-specific model performance more explicitly
- improve sparse bear-regime handling so bear conclusions are not overclaimed from too few observations

## 2. Linear Model Improvement

Planned improvements:

- re-evaluate Linear Regression, Ridge, and Lasso under the same feature sets used by boosting models
- add stronger feature-selection diagnostics
- compare coefficient stability across folds
- keep linear models as interpretable baselines and feature sanity checks

## 3. Deep Learning Improvement

Current deep learning coverage includes LSTM and BiLSTM, but it needs hardening before broader sequence models are added.

Planned improvements:

- improve LSTM/BiLSTM sequence preparation, persistence, and evaluation consistency
- add GRU as a lighter recurrent baseline
- add TCN for convolutional time-series forecasting
- add Transformer-style forecasting only after sequence evaluation stability is verified
- evaluate TFT, N-BEATS, and N-HiTS as later-stage research candidates, not immediate production models

## 4. Advanced Risk Model Improvement

Planned improvements:

- extend the current GARCH layer toward EGARCH and GJR-GARCH
- compare volatility forecasting quality across GARCH-family variants
- track whether volatility forecasts improve regime detection and risk-aware allocation
- keep VaR/CVaR interpretation conservative

## 5. Reinforcement Learning Research Extension

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

## Governance Notes

- Do not claim GRU, TCN, Transformer, TFT, N-BEATS, N-HiTS, CatBoost, EGARCH, GJR-GARCH, DQN, PPO, or A2C as implemented in the governed VSEF model surface until they have integrated code, persistence behavior, active workflow coverage, and tests.
- Keep RL separate from core inference until the environment design has explicit leakage controls and transaction-cost modeling.
- Treat saved backtest artifacts as research evidence, not guaranteed trading performance.
