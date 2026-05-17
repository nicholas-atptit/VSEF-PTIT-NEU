# NCKH Full Paper Final Review Checklist

Use this checklist before submitting or presenting the VN100 NCKH paper draft.

## Front Matter

- [ ] Title consistency: Vietnamese and English titles match across `NCKH_FULL_PAPER_DRAFT_VN100.md`, `NCKH_FULL_PAPER_DRAFT_VN100_CLEAN.md`, and `NCKH_ABSTRACT_AND_KEYWORDS.md`.
- [ ] Abstract consistency: Vietnamese and English abstracts preserve the same empirical claims and limitations.
- [ ] Keywords consistency: Vietnamese and English keyword lists cover VN100, walk-forward validation, machine learning, ensemble/stacking, directional accuracy, confidence filtering, regime diagnostics, and leakage control.

## Claim Boundary Compliance

- [ ] No global 60% pass claim.
- [ ] No trading-readiness claim.
- [ ] No stable full-market 63% claim.
- [ ] No claim that seven-ticker evidence represents the full VN100 universe.
- [ ] Conditional signal is described as diagnostic evidence only.
- [ ] Statistical significance is not described as profitability evidence.

## Evidence and Scope

- [ ] Data scope clarity: raw daily range, raw hourly range, hybrid daily construction, train cutoff, and 2025 evaluation window are described separately.
- [ ] Seven-ticker limitation is stated in methodology, results, conclusion, and defense notes.
- [ ] Confidence-sweep limitation is stated: available sweep rows cover hourly stacking h=1 and daily threshold-sweep rows are missing.
- [ ] Cost/slippage limitation is stated: no official selected-slice cost-adjusted return, slippage, turnover, drawdown, profit-factor, trade-list, or equity-curve artifacts.
- [ ] Appendix artifact map is present.

## Tables and Figures

- [ ] Table and figure numbering is consistent with `reports/NCKH_PAPER_TABLES_AND_FIGURES_PLAN.md`.
- [ ] `[Insert Table 1 here]` through `[Insert Table 8 here]` markers are present where appropriate.
- [ ] `[Insert Figure 1 here]` through `[Insert Figure 5 here]` markers are present where appropriate.
- [ ] Table 5 is marked partial because confidence-sweep evidence is incomplete.
- [ ] Table 7 is marked partial because significance evidence does not include full multiple-testing controls.
- [ ] Figure 4 is marked partial because the available threshold sweep is limited.

## Citations and References

- [ ] Chapter 2 citation placeholders are present for Breiman (2001), Wolpert (1992), Chen and Guestrin (2016), Ke et al. (2017), Diebold and Mariano (1995), Christoffersen and Diebold (2006), Bergmeir and Benítez (2012), Hyndman and Athanasopoulos (2021), and Almgren and Chriss (2001).
- [ ] `reports/NCKH_REFERENCES_APA7.md` includes APA 7 scaffold entries.
- [ ] Citation completeness has been checked against final institutional formatting rules.
- [ ] Entries marked or noted for final verification are checked before submission.

## Final Readiness

- [ ] No production/runtime/model code changes are required for the paper draft.
- [ ] No new model family is claimed.
- [ ] No heavy benchmark rerun is implied as already completed.
- [ ] Remaining evidence gaps are listed in Chapter 5 or the appendix.
