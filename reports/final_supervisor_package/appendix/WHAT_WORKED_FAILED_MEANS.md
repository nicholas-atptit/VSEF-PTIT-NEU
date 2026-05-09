# What Worked, What Failed, What This Means

## What Worked

- Governance freeze worked. Phase 0 gave VSEF v1 a defensible boundary and prevented uncontrolled scope expansion.
- Experiment standardization worked. Phase 1 created a repeatable config-to-artifacts workflow.
- Provider-backed smoke execution worked. `EXP-SMOKE-001` generated non-empty provider-backed predictions and metrics.
- Baseline comparison worked. Phase 2 put models and baselines in the same metric table instead of assuming model value.
- Multi-horizon evaluation worked. T+1, T+3, and T+5 behavior can be compared from generated artifacts.
- Risk-aware candidate research worked as a diagnostic process. It tested a risk-layer claim and found that the claim was not supported in aggregate.
- Regime-aware analysis produced stronger academic framing. Phase 4 moved the project from "find the best model" toward "test whether model behavior depends on regime."
- Health gate exposed outliers instead of hiding them. Phase 4 marked eligible, flagged, and excluded rows without filtering results until they looked better.

## What Failed or Remained Weak

- Forecasting models did not consistently beat simple baselines on MAE/RMSE.
- Stacking did not prove superiority.
- Risk-aware ranking did not improve aggregate candidate utility.
- Some forecasts produced outliers.
- Some regimes have small samples.
- Risk feature design remains basic.
- Candidate scoring remains sensitive to unstable forecast outputs.

## What This Means

VSEF v1 should not be positioned as an investment recommendation system.

VSEF v1 should be positioned as a reproducible diagnostic research framework. The strongest contribution is regime-aware evaluation combined with governance discipline. It is valuable because it makes negative results visible, compares against baselines, preserves artifacts, and narrows future work toward evidence rather than model accumulation.

Future work should focus on eligibility gates, regime filters, better risk features, and forecast outlier control. Adding more models without resolving baseline competitiveness, outlier sensitivity, and regime-specific validity would not address the main weaknesses found by Phases 2 through 4.
