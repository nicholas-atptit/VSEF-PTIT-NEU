# Defense Slide Outline

## Slide 1: Title and Research Positioning

- Key message: VSEF v1 is a governed stock forecasting research framework, not a trading system.
- Bullet content:
  - VSEF v1: governance, reproducible experiments, baseline comparison, risk diagnostics, regime-aware evaluation.
  - Research position: no universal best model across all market regimes.
  - Output class: diagnostic research artifacts.
- Suggested visual: System title with five-phase timeline.
- Speaker note: Start by clarifying the boundary. The project evaluates forecasting and diagnostic evidence; it does not issue investment advice.

## Slide 2: Problem and Motivation

- Key message: Forecasting claims are weak without governance, baselines, and regime context.
- Bullet content:
  - Many forecasting systems overclaim model superiority.
  - Simple baselines can be hard to beat.
  - Market regimes change error behavior and risk utility.
- Suggested visual: Problem triangle: baselines, risk, regimes.
- Speaker note: The motivation is not to chase the most complex model. It is to build a defensible way to test whether models add value.

## Slide 3: VSEF v1 System Overview

- Key message: VSEF v1 produces auditable diagnostic artifacts.
- Bullet content:
  - Input: `vnstock_data` daily OHLCV.
  - Runtime: config-driven experiments.
  - Outputs: predictions, metrics, rankings, risk summaries, regime labels, reports.
- Suggested visual: Pipeline diagram from config and data to reports.
- Speaker note: The system is designed around traceability: every major claim should point to an artifact.

## Slide 4: Governance and Scope Freeze

- Key message: Phase 0 froze what v1 can and cannot claim.
- Bullet content:
  - Frozen provider: `vnstock_data`.
  - Frequency: daily OHLCV.
  - Supported models: SARIMAX, ETS, XGBoost, LightGBM, LSTM, BiLSTM, Stacking.
  - Excluded: live trading, broker execution, capital allocation, autonomous LLM decisions.
- Suggested visual: Included vs excluded table.
- Speaker note: This is important for defense because it prevents accidental claims beyond evidence.

## Slide 5: Experiment Standardization

- Key message: Phase 1 made experiments reproducible and comparable.
- Bullet content:
  - `CONFIG IN -> ORCHESTRATED RUN -> STANDARD ARTIFACTS OUT`.
  - `ExperimentOrchestrator`.
  - Metrics engine.
  - Baseline registry.
  - Provider-backed smoke evidence.
- Suggested visual: Output folder contract.
- Speaker note: The goal was not new model ambition. The goal was reproducibility and auditability.

## Slide 6: Forecasting Core Validation

- Key message: Phase 2 did not prove consistent ML superiority over baselines.
- Bullet content:
  - Forecast metrics rows: 1,295.
  - Model ranking rows: 1,295.
  - Horizon comparison rows: 185.
  - Persistence baseline frequently wins MAE/RMSE.
  - Some model wins exist but are bounded.
- Suggested visual: Baseline vs model evidence summary.
- Speaker note: This is a key honest result. The framework is useful partly because it makes this visible.

## Slide 7: Risk-Aware Candidate Research

- Key message: Phase 3 tested risk-aware ranking and found no aggregate improvement.
- Bullet content:
  - Candidate comparison rows: 1,780.
  - Top-N basket metric rows: 18.
  - Risk-aware ranking did not improve aggregate candidate utility.
  - Some specific rows improved, but not enough for a universal claim.
- Suggested visual: Forecast-only vs risk-aware comparison table.
- Speaker note: Candidates are diagnostics, not recommendations. The risk layer remains a research target.

## Slide 8: Regime-Aware Analysis

- Key message: Phase 4 is the main academic contribution.
- Bullet content:
  - Regime labels: 2,495 rows.
  - Trend regimes: bull, bear, sideway.
  - Volatility regimes: high_vol, low_vol.
  - Model/risk/horizon behavior evaluated by regime.
  - Health gate flags outliers and weak evidence.
- Suggested visual: Regime distribution chart or table.
- Speaker note: Phase 4 reframes the project around conditional behavior rather than a universal model winner.

## Slide 9: Main Evidence Matrix

- Key message: Every major claim maps to an artifact.
- Bullet content:
  - Governance documents.
  - Source modules.
  - Experiment configs.
  - Metrics CSVs.
  - Reports.
  - Health and eligibility tables.
- Suggested visual: Evidence matrix excerpt.
- Speaker note: This slide is for defensibility. It shows that claims are not based on narrative alone.

## Slide 10: Key Findings

- Key message: VSEF v1 is a governed diagnostic framework with honest negative results.
- Bullet content:
  - No single ML model dominates all tested conditions.
  - Baselines remain competitive.
  - Risk-aware ranking did not improve aggregate utility.
  - Regime-aware analysis is the strongest contribution.
  - Future work should improve filters, gates, risk features, and outlier control.
- Suggested visual: Findings scoreboard.
- Speaker note: The main value is disciplined evidence, not overstated performance.

## Slide 11: Limitations

- Key message: Results are bounded by data, models, risk features, and regime definitions.
- Bullet content:
  - Daily OHLCV only.
  - One provider policy.
  - Frozen model set.
  - Basic risk features.
  - Rule-based regimes.
  - Small samples in some regimes.
- Suggested visual: Threats-to-validity table.
- Speaker note: A defensible project states limitations clearly and uses them to guide next work.

## Slide 12: Future Work and Conclusion

- Key message: Next work should improve evidence quality, not broaden scope blindly.
- Bullet content:
  - Regime-aware filtering.
  - Model health and eligibility gate.
  - Better risk features.
  - Forecast outlier control.
  - Candidate policy improvement.
  - Robustness testing.
- Suggested visual: Roadmap ladder.
- Speaker note: Conclude that VSEF v1 establishes a governed research foundation. It is not investment advice and not a finished trading system.
