# NCKH Final Submission Notes

## What Is Ready

- The clean final-review manuscript is available at `reports/NCKH_FULL_PAPER_DRAFT_VN100_CLEAN.md`.
- Vietnamese and English abstracts and keywords are available at `reports/NCKH_ABSTRACT_AND_KEYWORDS.md`.
- APA 7 reference scaffolding is available at `reports/NCKH_REFERENCES_APA7.md`.
- Tables 1-8 are available as CSV and Markdown under `reports/generated/paper_tables/`.
- Figures 1-5 are available under `reports/generated/paper_figures/`.
- Defense materials are available in `reports/NCKH_DEFENSE_SUMMARY.md`, `reports/NCKH_PRESENTATION_OUTLINE.md`, and `reports/NCKH_QUESTIONS_AND_ANSWERS.md`.
- Claim governance is available in `reports/NCKH_RESULTS_CLAIM_REGISTER.md`.

## What Remains Partial

- Table 5 is partial because the confidence-threshold sweep evidence is limited to hourly stacking h=1, and daily threshold-sweep rows are missing.
- Table 7 is partial because statistical significance rows do not by themselves establish full multiple-testing control or trading readiness.
- Figure 4 is partial because it visualizes the available threshold sweep, which is not a full sweep across all model/horizon/frequency combinations.
- The APA 7 references are locally scaffolded and should receive final supervisor or library verification before formal submission.

## What Must Not Be Claimed

- Do not claim a global 60% benchmark pass.
- Do not claim practical trading readiness.
- Do not claim a stable full-market 63% method.
- Do not claim representativeness for the full VN100 universe.
- Do not claim profitability, investment suitability, broker execution readiness, or portfolio deployment.
- Do not claim that new model families were added.

## Final Evidence Gaps

- Official evaluation covers only seven tickers: ANV, BCM, BID, BMP, BVH, BWE, and CII.
- Daily confidence-threshold sweep rows are missing.
- Confidence sweep evidence is limited to hourly stacking h=1.
- The selected confidence slice is concentrated in five tickers, with the top three contributing most selected rows.
- No official selected-slice cost/slippage, turnover, drawdown, profit-factor, trade-list, equity-curve, or cost-adjusted return artifacts exist.
- Evidence is limited to one official 2025 evaluation window.
- Ex-ante regime validation has not yet been completed.

## Supervisor-Facing Summary

The submission package is ready for academic review as a careful empirical benchmarking paper. The strongest contribution is the leakage-aware VN100 evaluation framework and disciplined claim boundary. The global benchmark does not pass 60%, but conditional diagnostics show signal that merits future validation. The paper is not positioned as a trading system or full-market VN100 proof.

## Committee-Facing Summary

The paper evaluates LightGBM, XGBoost, random forest, and stacking for VN100 direction forecasting under a walk-forward design with a 2024-12-31 training-label cutoff and 2025 held-out evaluation. The official benchmark does not pass the global 60% threshold. A selected hourly stacking confidence-filtered slice reaches 60.03% accuracy at 31.30% coverage, and daily bear-regime h=20 diagnostics exceed 63%, but both are conditional diagnostics. The main limitations are seven-ticker coverage, partial confidence-sweep evidence, selected-slice concentration, and missing cost/slippage trading-readiness artifacts.
