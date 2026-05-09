# Defense Script

## Short Version: 3 Minutes

Good morning. This project is VSEF v1, a governed stock forecasting research framework. The key point is that it is not a trading system. It does not produce BUY, SELL, or HOLD advice, it does not allocate capital, and it does not execute through a broker. It produces diagnostic evidence: forecasts, baselines, metrics, risk summaries, candidate diagnostics, regime labels, and reports.

The project began with governance. Phase 0 froze the v1 scope: the data provider is `vnstock_data`, the frequency is daily OHLCV, the supported models are SARIMAX, ETS, XGBoost, LightGBM, LSTM, BiLSTM, and Stacking, and anything like live trading, broker execution, capital allocation, or autonomous LLM decisions is outside v1.

Phase 1 then standardized experiments. The core pattern is config in, orchestrated run, standard artifacts out. The `ExperimentOrchestrator`, metrics engine, and baseline registry make the experiments reproducible and comparable.

Phase 2 tested the forecasting core against simple baselines. This is important because a model is not useful just because it is more complex. The evidence does not prove consistent model superiority over baselines on MAE or RMSE. Persistence remains highly competitive, although some models show bounded wins in specific contexts such as directional accuracy or a limited ticker/horizon case.

Phase 3 tested whether a risk-aware ranking layer improves diagnostic candidate utility. The answer in aggregate is no. Risk-aware ranking did not improve aggregate candidate utility over forecast-only ranking. That result is useful because it identifies risk feature design and eligibility rules as future work.

Phase 4 is the key academic contribution. It asks whether model quality, risk-layer utility, and horizon behavior depend on market regime. The regime detector labels bull, bear, sideway, high-volatility, and low-volatility states using transparent rolling rules. The Phase 4 evidence supports a no-universal-best-model thesis under the tested regime definitions. It also shows that persistence remains a strong baseline and that risk-aware value is context-specific rather than universal.

The final conclusion is honest: VSEF v1 does not prove that one ML model dominates all market conditions. Its strongest contribution is a governed, reproducible diagnostic framework for testing models, baselines, risk layers, and horizons under explicit regime definitions. Future work should focus on regime-aware filtering, model health gates, risk feature design, and forecast outlier control.

## Full Version: 8-10 Minutes

Good morning. I will present VSEF v1, a governed stock forecasting research framework. The first clarification is important: VSEF v1 is not a trading system. It does not produce investment advice, it does not allocate capital, and it does not execute orders. Its purpose is to produce research and diagnostic decision-support artifacts that can be inspected, reproduced, and defended.

The outputs of the system are forecasts, baselines, metrics, rankings, risk summaries, diagnostic candidate tables, regime labels, and reports. These outputs are useful for research because they make claims testable. They are not BUY, SELL, or HOLD recommendations.

The project was implemented in phases.

Phase 0 established governance. This phase froze the scope of VSEF v1. The official data provider is `vnstock_data`; the data frequency is daily OHLCV; the required schema is date, ticker, open, high, low, close, and volume. The supported v1 forecasting models are SARIMAX, ETS, XGBoost, LightGBM, LSTM, BiLSTM, and Stacking. Phase 0 also explicitly excluded live trading, broker execution, capital allocation, portfolio allocation authority, autonomous LLM decision agents, intraday trading, and new unsupported models.

This governance matters because forecasting projects often overclaim. Without a scope freeze, it becomes easy to mix research diagnostics with trading language. VSEF v1 avoids that by requiring every claim to be backed by an artifact path and by keeping diagnostic candidates separate from investment recommendations.

Phase 1 standardized experiments. The design principle was: config in, orchestrated run, standard artifacts out. The `ExperimentOrchestrator` loads a YAML config, validates governance boundaries, writes original and resolved configs, runs configured models and baselines, and writes predictions, metrics, logs, manifests, and summaries. The metrics engine standardizes model and baseline metrics in a single long-form table. The baseline registry makes simple comparison methods explicit.

This matters because reproducibility is part of the research contribution. A model comparison is not persuasive if every run has a different output structure. Phase 1 creates a consistent evidence contract.

Phase 2 evaluated the forecasting core. The key research discipline here is baseline comparison. A model is not valuable simply because it is complex. It must be compared against simple baselines in the same ticker, horizon, date-window, and metric context. Phase 2 generated forecast metrics, model rankings, stability metrics, horizon comparisons, and error distribution summaries.

The Phase 2 conclusion is direct: the current evidence does not prove that the forecasting layer consistently outperforms simple baselines. Persistence is highly competitive on MAE and RMSE. Some models show bounded strengths, for example in directional accuracy or specific ticker/horizon contexts, but the evidence is not broad enough to claim general superiority. Stacking ran, which is useful runtime evidence, but it did not prove clear superiority.

Phase 3 evaluated the risk-aware candidate layer. The idea was to compare forecast-only diagnostic candidate ranking against a risk-aware ranking that includes volatility, drawdown, VaR, CVaR, and missing-metric penalties. The output was candidate comparison tables, risk-adjusted ranking, drawdown comparison, hit-ratio comparison, and top-N diagnostic basket metrics.

The Phase 3 conclusion is also direct: the current evidence does not prove that risk-aware ranking improves candidate utility over forecast-only ranking in aggregate. Some individual rows show context-specific improvements, but the aggregate evidence weakens a universal risk-layer value claim. This is not a failure of the research process. It is useful evidence that the risk features and eligibility rules need better design.

Phase 4 is the key academic contribution. Earlier phases showed that model value was not obvious in aggregate. Phase 4 asks a better question: does model usefulness depend on market regime? The regime detector uses rule-based rolling return and realized-volatility logic to label bull, bear, sideway, high-volatility, and low-volatility states. This is deliberately transparent. It avoids treating regime labels as hidden ground truth.

The Phase 4 dataset contains 2,495 regime-label rows for ACB, DGC, FPT, HPG, and MWG from 2023-01-03 to 2024-12-31. It then evaluates model performance, risk-layer utility, and horizon behavior by trend regime, volatility regime, and combined regime. It also introduces a model health and eligibility gate. The gate flags outliers, weak prediction counts, missing prediction rates, and large baseline gaps. It does not hide bad rows; it marks them.

The Phase 4 conclusion is that the evidence supports a no-universal-best-model thesis under the tested regime definitions. It does not prove that one ML model dominates all conditions. In fact, persistence remains a strong baseline in many MAE contexts. Risk-aware utility is mixed and context-specific. Horizon behavior can be evaluated by regime, but T+1 outliers show why health gates and outlier control matter.

The final synthesis is that VSEF v1 is strongest as a governed diagnostic research framework. Its contribution is not a claim of profitable signals. Its contribution is a reproducible method for comparing models against baselines, testing risk layers, labeling regimes, and refusing to overclaim where evidence is weak.

Future work should therefore focus on regime-aware filtering, model health and eligibility thresholds, improved risk features, forecast outlier detection, candidate policy improvements, and robustness tests. Adding more models blindly would not solve the core issues exposed by the evidence.

To close, VSEF v1 should be defended as a disciplined research system. It creates evidence, exposes negative results, and gives a clear path for future improvements. It is not investment advice, not a trading engine, and not proof of guaranteed profitable trading.
