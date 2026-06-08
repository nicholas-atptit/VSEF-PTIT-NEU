# VN30 Paper Figure Captions EN

## Figure 1. Benchmark comparison summary across methods and horizons

- Data: Uses summary CSVs for the selected VN30 stock-only candidate, supported index benchmark, and exploratory joint panel.
- Shows: Compares directional accuracy and baseline context.
- Interpretation: Read stock-only selected evidence as the paper's primary result; treat index and joint panels as context.
- Caveats: Joint panel is summary-only and not a successful benchmark claim.

## Figure 2. Selected best candidate actual vs predicted over time

- Data: Uses selected L2 Logistic h40 row-level reproduction.
- Shows: Shows actual and predicted binary directions plus rolling accuracy.
- Interpretation: Supports the claim of meaningful directional signal around 60%+.
- Caveats: Time stability is mixed and validation-final gap must be disclosed.

## Figure 3. Representative instrument forecast-vs-actual comparison

- Data: Uses representative stock and index row-level prediction artifacts.
- Shows: Compares available core methods by instrument.
- Interpretation: Shows method and instrument heterogeneity.
- Caveats: Stacking row-level predictions are missing.

## Figure 4. Per-instrument accuracy distribution

- Data: Uses selected-candidate ticker slices and index benchmark summaries.
- Shows: Ranks stock and index accuracy separately.
- Interpretation: Shows that aggregate accuracy is not uniform.
- Caveats: Stock and index panels are not the same experiment.

## Figure 5. Selected-candidate stability

- Data: Uses rolling 250/500/1000, monthly, and quarterly summaries.
- Shows: Shows rolling and calendar-period accuracy.
- Interpretation: Aggregate signal coexists with instability.
- Caveats: Rolling windows are row-based.

## Figure 6. Model-vs-actual overlay

- Data: Uses one stock and one index row-level overlay.
- Shows: Shows actual and predictions on the same timeline.
- Interpretation: Highlights agreement and disagreement periods.
- Caveats: Index overlay lacks L2 Logistic and stacking.
