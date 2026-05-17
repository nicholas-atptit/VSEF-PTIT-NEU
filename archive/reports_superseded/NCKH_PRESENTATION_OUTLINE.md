# NCKH VN100 Presentation Outline

Target length: 7-10 minutes.

## Slide 1: Title and Research Problem

- Walk-forward evaluation of machine learning and ensemble models for VN100 stock direction forecasting.
- Core problem: forecasting claims can be overstated without chronological validation.
- Study uses a 2025 held-out evaluation design.
- Focus is evidence quality, not trading deployment.

Speaker note: Open by explaining that the research tests whether VN100 directional forecasts hold up under a leakage-aware walk-forward design.

Claim boundary warning: Do not imply trading readiness.

## Slide 2: Research Gap

- Many forecasting studies emphasize headline accuracy.
- Financial forecasting needs leakage control and baseline comparison.
- Selected slices must not be presented as full-market proof.
- This study separates global, confidence-filtered, and regime-specific claims.

Speaker note: Emphasize that the contribution is a disciplined benchmark and claim boundary.

Claim boundary warning: Do not claim full-market representativeness.

## Slide 3: Data and Scope

- Universe target: VN100.
- Official evaluated tickers: ANV, BCM, BID, BMP, BVH, BWE, CII.
- Training-label cutoff: 2024-12-31.
- Held-out evaluation window: 2025.
- Daily benchmark uses documented hybrid construction.

Speaker note: State clearly that the current artifact evidence evaluates seven tickers, so representativeness is limited.

Claim boundary warning: Seven-ticker evidence is not full VN100 evidence.

## Slide 4: Walk-Forward Methodology

- Chronological evaluation prevents future-label leakage.
- Rule: `target_timestamp <= train_cutoff`.
- Models are evaluated against held-out 2025 outcomes.
- Metrics include accuracy, baseline deltas, confidence filtering, regimes, and significance.

Speaker note: Use Figure 2 to explain the cutoff and evaluation timeline.

Claim boundary warning: Actual 2025 rows are used for evaluation labels, not training labels.

## Slide 5: Models and Baselines

- Models: LightGBM, XGBoost, random forest, stacking.
- Baselines: always-up, previous-direction, random seeded direction, moving-average signal.
- No new model family is added in this paper package.
- Baselines help avoid interpreting raw accuracy in isolation.

Speaker note: Explain that the benchmark compares existing supported models against simple directional rules.

Claim boundary warning: Do not claim model superiority across all VN100 conditions.

## Slide 6: Global Benchmark Results

- Daily accuracy: 53.19% over 26,104 predictions.
- Hourly accuracy: 51.29% over 127,944 predictions.
- Global 60% benchmark pass: no.
- Some model/horizon rows beat simple baselines.

Speaker note: Lead with the negative global result. This improves credibility and frames the conditional findings correctly.

Claim boundary warning: Do not claim a global 60% pass.

## Slide 7: Confidence-Filtered Diagnostic

- Selected slice: hourly stacking h=1 at threshold 0.57.
- Filtered accuracy: 60.03%.
- Coverage: 31.30%.
- Evaluated rows: 2,297.
- Under 50% and 40% coverage floors, no available row reaches 60%.

Speaker note: Explain that higher confidence improves the selected slice but reduces coverage materially.

Claim boundary warning: This is a strategy-level diagnostic, not a global pass.

## Slide 8: Regime-Specific Diagnostic

- Strongest diagnostic appears in daily bear-regime h=20.
- LightGBM h=20: 69.59% over 444 observations.
- XGBoost h=20: 69.14% over 444 observations.
- Finding is regime-specific and diagnostic.

Speaker note: Make clear that the regime result is conditional and requires ex-ante validation.

Claim boundary warning: Do not call this a stable full-market 63% method.

## Slide 9: Limitations

- Only seven evaluated tickers.
- Daily confidence-threshold sweep rows are missing.
- Confidence sweep evidence covers hourly stacking h=1 only.
- Selected confidence slice is concentrated.
- No official cost/slippage or trading-readiness artifacts.

Speaker note: Use this slide to show that the paper does not overclaim.

Claim boundary warning: Do not claim representativeness or profitability.

## Slide 10: Conclusion and Future Work

- Global benchmark pass: no.
- Conditional signal: yes.
- Stable full-market 63% method: no.
- Practical trading readiness: not established.
- Future work: cache expansion, multi-window validation, broader confidence sweeps, ex-ante regime validation, cost/slippage backtesting.

Speaker note: End with the safe final claim: the official benchmark does not pass globally, but conditional diagnostics show signal worth further validation.

Claim boundary warning: Future work is required before trading or full-market claims.
