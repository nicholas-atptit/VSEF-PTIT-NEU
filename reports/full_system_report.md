# Full System Report

## 1. System Architecture Summary
- Forecasting layer: existing CART/XGBoost/LightGBM/SARIMAX/ETS/LSTM/BiLSTM models through the manifest-driven trainer.
- Risk layer: rolling VaR/CVaR/CoVaR/Delta-CoVaR/drawdown features and system-level summaries.
- Regime layer: feature-based NORMAL/HIGH_VOL/CRISIS classification.
- Allocation layer: optional risk-aware exposure scaling using risk and regime state.

## 2. Benchmark Results
| benchmark_mode | cumulative_return | cagr | volatility | sharpe | sortino | calmar | max_drawdown | avg_drawdown | tail_loss | turnover | exposure | trade_count | rmse | mae | directional_accuracy | test_rows | delta_sharpe_vs_legacy | delta_cagr_vs_legacy | delta_mdd_vs_legacy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| forecast_plus_risk_and_regime | 0.227433 | 0.329467 | 0.070067 | 4.873333 | 7.663333 | 11.3838 | -0.0707 | -0.0453 | -0.006533 | 15.666667 | 0.328 | 8.0 | 0.19624 | 0.167484 | 0.562587 | 179.666667 | 0.0 | 0.0 | 0.0 |
| forecast_plus_risk_features | 0.227433 | 0.329467 | 0.070067 | 4.873333 | 7.663333 | 11.3838 | -0.0707 | -0.0453 | -0.006533 | 15.666667 | 0.328 | 8.0 | 0.19624 | 0.167484 | 0.562587 | 179.666667 | 0.0 | 0.0 | 0.0 |
| full_system | 0.2171 | 0.3149 | 0.059967 | 5.133333 | 15.806667 | 16.414667 | -0.034667 | -0.025233 | -0.003633 | 9.65 | 0.238067 | 8.0 | 0.19624 | 0.167484 | 0.562587 | 179.666667 | 0.26 | -0.014567 | 0.036033 |
| legacy_forecast_only | 0.227433 | 0.329467 | 0.070067 | 4.873333 | 7.663333 | 11.3838 | -0.0707 | -0.0453 | -0.006533 | 15.666667 | 0.328 | 8.0 | 0.19624 | 0.167484 | 0.562587 | 179.666667 | 0.0 | 0.0 | 0.0 |

## 3. Stress Test Results
| stress_scenario | benchmark_mode | stressed_sharpe | stressed_max_drawdown | stressed_tail_loss | stressed_exposure | stressed_turnover | delta_sharpe | delta_tail_loss | delta_drawdown | delta_exposure | regime_reaction_speed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drawdown_shock | forecast_plus_risk_and_regime | 4.6 | -0.080833 | -0.007833 | 0.328 | 15.666667 | -0.273333 | -0.0013 | -0.010133 | 0.0 | 0.0 |
| drawdown_shock | forecast_plus_risk_features | 4.6 | -0.080833 | -0.007833 | 0.328 | 15.666667 | -0.273333 | -0.0013 | -0.010133 | 0.0 | 0.0 |
| drawdown_shock | full_system | 5.206667 | -0.028467 | -0.003033 | 0.204633 | 8.15 | 0.073333 | 0.0006 | 0.0062 | -0.033433 | 0.0 |
| drawdown_shock | legacy_forecast_only | 4.6 | -0.080833 | -0.007833 | 0.328 | 15.666667 | -0.273333 | -0.0013 | -0.010133 | 0.0 | 24.666667 |
| liquidity_cost_shock | forecast_plus_risk_and_regime | 0.87 | -0.158733 | -0.014033 | 0.328 | 15.666667 | -4.003333 | -0.0075 | -0.088033 | 0.0 | 4.333333 |
| liquidity_cost_shock | forecast_plus_risk_features | 0.87 | -0.158733 | -0.014033 | 0.328 | 15.666667 | -4.003333 | -0.0075 | -0.088033 | 0.0 | 4.333333 |
| liquidity_cost_shock | full_system | 1.833333 | -0.081167 | -0.009133 | 0.238067 | 9.65 | -3.3 | -0.0055 | -0.0465 | 0.0 | 4.333333 |
| liquidity_cost_shock | legacy_forecast_only | 0.87 | -0.158733 | -0.014033 | 0.328 | 15.666667 | -4.003333 | -0.0075 | -0.088033 | 0.0 | 24.666667 |
| regime_persistence_shock | forecast_plus_risk_and_regime | 4.466667 | -0.086267 | -0.007367 | 0.328 | 15.666667 | -0.406667 | -0.000833 | -0.015567 | 0.0 | 0.0 |
| regime_persistence_shock | forecast_plus_risk_features | 4.466667 | -0.086267 | -0.007367 | 0.328 | 15.666667 | -0.406667 | -0.000833 | -0.015567 | 0.0 | 0.0 |
| regime_persistence_shock | full_system | 5.086667 | -0.0265 | -0.002533 | 0.195267 | 7.15 | -0.046667 | 0.0011 | 0.008167 | -0.0428 | 0.0 |
| regime_persistence_shock | legacy_forecast_only | 4.466667 | -0.086267 | -0.007367 | 0.328 | 15.666667 | -0.406667 | -0.000833 | -0.015567 | 0.0 | 0.0 |
| volatility_shock | forecast_plus_risk_and_regime | 4.456667 | -0.113767 | -0.011 | 0.328 | 15.666667 | -0.416667 | -0.004467 | -0.043067 | 0.0 | 0.0 |
| volatility_shock | forecast_plus_risk_features | 4.456667 | -0.113767 | -0.011 | 0.328 | 15.666667 | -0.416667 | -0.004467 | -0.043067 | 0.0 | 0.0 |
| volatility_shock | full_system | 5.08 | -0.0376 | -0.003667 | 0.1862 | 7.65 | -0.053333 | -3.3e-05 | -0.002933 | -0.051867 | 0.0 |
| volatility_shock | legacy_forecast_only | 4.456667 | -0.113767 | -0.011 | 0.328 | 15.666667 | -0.416667 | -0.004467 | -0.043067 | 0.0 | 6.666667 |

## 4. Risk Tuning Improvements
Best validation score: `-8.326453`

Best parameters:
- `risk_enabled`: `True`
- `enable_covar`: `True`
- `enable_risk_engine`: `True`
- `enable_regime_detection`: `True`
- `enable_regime_switching`: `True`
- `enable_risk_allocation`: `True`
- `regime_method`: `threshold`
- `random_seed`: `42`
- `simulations`: `10000`
- `confidence_levels`: `[0.95, 0.99]`
- `covar_quantile`: `0.0373818018663584`
- `covar_window`: `73`
- `risk_penalty_strength`: `1.3526405540621358`
- `high_vol_threshold`: `0.03392989411287273`
- `crisis_drawdown_threshold`: `-0.1276294210555241`
- `crisis_delta_covar_threshold`: `0.011277223729341883`
- `high_vol_exposure_cut`: `0.47528678912113087`
- `crisis_exposure_cut`: `0.2148628294821613`

## 5. Trade-offs
- Performance vs stability: richer risk/regime overlays can reduce exposure and headline return while improving drawdown behavior.
- Complexity vs benefit: benchmark/stress/tuning orchestration adds operational complexity but makes model comparisons and deployment decisions auditable.

## 6. Limitations
- Stress testing re-evaluates held-out predictions under shocked returns/costs instead of retraining on synthetic crisis histories.
- Regime logic remains rule-based Option A; no latent-state model is introduced.
- Allocation remains a modular overlay, not a mandatory execution engine.

## 7. Recommendation
- Use the full system in production only with benchmark plus stress plus tuning outputs reviewed together.
- Prefer the tuned full-system configuration when it improves Sharpe/Sortino without materially worsening max drawdown or turnover.