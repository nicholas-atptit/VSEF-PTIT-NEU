# Limitations and Threats to Validity

## 1. Data Limitations

- VSEF v1 depends on `vnstock_data` availability.
- VSEF v1 uses daily OHLCV only.
- VSEF v1 does not include intraday data.
- VSEF v1 does not include alternative data in the frozen v1 scope.
- Provider/API changes may affect reproducibility.
- Local cached data and generated reports must be reviewed with provenance in mind.

## 2. Model Limitations

- The supported model set is frozen for v1.
- Deep learning evidence may be limited by sample size and configuration.
- Stacking did not prove clear superiority.
- Model performance is sensitive to ticker, horizon, date window, and regime.
- Forecast outliers were observed and should not be treated as valid evidence of predictive strength.

## 3. Baseline Competitiveness

- The persistence baseline remains strong.
- Model value should be judged against simple baselines.
- Weak model-vs-baseline evidence limits forecasting-value claims.
- Baseline competitiveness is a central result, not an inconvenience to hide.

## 4. Risk Layer Limitations

- Risk-aware ranking did not improve aggregate utility.
- Risk features are basic.
- Risk-aware value is context-specific.
- Future risk features need better design.
- Risk-aware ranking can inherit instability from forecast outputs.

## 5. Regime Definition Limitations

- Rule-based thresholds are subjective.
- Robustness checks exist but are not exhaustive.
- Some regimes have small sample sizes.
- Labels are diagnostics, not ground truth.
- Regime labels may shift under alternative return windows, volatility windows, or threshold policies.

## 6. Candidate Evaluation Limitations

- Diagnostic baskets are not real portfolios.
- No transaction costs are assumed except where explicitly configured.
- There is no broker execution.
- There is no investment advice.
- Realized outcomes are retrospective evidence only.
- Diagnostic candidates are not recommendations.

## 7. Reproducibility Limitations

- Local environment matters.
- Dependencies matter.
- Generated outputs under `outputs/` are not committed raw.
- Reports summarize evidence, but raw local artifacts should be retained during review.
- Some evidence tables summarize generated artifacts rather than replacing them.

## 8. Interpretation Limitations

The evidence supports a governed diagnostic framework, not a live decision system. Negative or weak results are part of the contribution because they constrain future work and prevent unsupported claims.
