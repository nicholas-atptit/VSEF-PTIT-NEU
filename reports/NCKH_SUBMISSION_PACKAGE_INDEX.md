# NCKH VN100 Submission Package Index

## Purpose

This index organizes the final-review submission and defense materials for the VN100 stock-direction forecasting NCKH paper. It does not introduce new empirical claims, model logic, runtime behavior, or benchmark results.

## Core Submission Files

| Item | File or directory | Use |
|---|---|---|
| Final manuscript file | `reports/NCKH_FULL_PAPER_DRAFT_VN100_CLEAN.md` | Clean final-review manuscript with citation placeholders and table/figure insertion markers. |
| Abstract and keywords | `reports/NCKH_ABSTRACT_AND_KEYWORDS.md` | Vietnamese and English titles, abstracts, and keywords. |
| References | `reports/NCKH_REFERENCES_APA7.md` | APA 7 reference scaffold using locally verified metadata. |
| Final review checklist | `reports/NCKH_FULL_PAPER_FINAL_REVIEW_CHECKLIST.md` | Checklist for title, abstract, claim boundaries, tables, figures, references, and scope. |
| Defense summary | `reports/NCKH_DEFENSE_SUMMARY.md` | 1-minute, 3-minute, and 5-minute defense summaries with safe answers. |
| Claim register | `reports/NCKH_RESULTS_CLAIM_REGISTER.md` | Safe, conditional, and unsafe claims with allowed and forbidden wording. |
| Paper tables | `reports/generated/paper_tables/` | Tables 1-8 in CSV and Markdown form. |
| Paper figures | `reports/generated/paper_figures/` | Figures 1-5 as Markdown schematics or PNG charts. |
| Generated notes | `reports/generated/paper_notes/` | Artifact-pack status and claim-boundary notes. |
| Presentation outline | `reports/NCKH_PRESENTATION_OUTLINE.md` | 7-10 minute slide plan. |
| Questions and answers | `reports/NCKH_QUESTIONS_AND_ANSWERS.md` | Committee Q&A preparation with safe wording. |
| Final submission notes | `reports/NCKH_FINAL_SUBMISSION_NOTES.md` | Readiness, partial evidence, prohibited claims, and final evidence gaps. |
| V2 evidence addendum | `reports/NCKH_EVIDENCE_UPGRADE_ADDENDUM_V2.md` | Evidence-upgrade addendum after gap-closure diagnostics; not a full-paper rewrite. |
| V2 update plan | `reports/NCKH_FULL_PAPER_V2_UPDATE_PLAN.md` | Exact instructions for later table, figure, Chapter 4, appendix, abstract, and conclusion updates. |
| V2 claim register | `reports/NCKH_RESULTS_CLAIM_REGISTER_V2.md` | Updated claim boundaries for derived diagnostics. |
| Supervisor gap-closure brief | `reports/NCKH_SUPERVISOR_BRIEF_AFTER_GAP_CLOSURE.md` | One-page supervisor-facing explanation and cautious wording. |

## Artifact Map

| Artifact | Supports | Status |
|---|---|---|
| `reports/generated/paper_tables/table1_dataset_evaluation_scope.md` | Dataset and evaluation scope | ready |
| `reports/generated/paper_tables/table2_model_baseline_list.md` | Model and baseline list | ready |
| `reports/generated/paper_tables/table3_global_benchmark_results.md` | Global benchmark results | ready |
| `reports/generated/paper_tables/table4_baseline_delta_summary.md` | Baseline comparison | ready |
| `reports/generated/paper_tables/table5_confidence_filtered_diagnostics.md` | Confidence-filtered strategy diagnostics | partial |
| `reports/generated/paper_tables/table6_regime_specific_diagnostics.md` | Regime diagnostics | ready |
| `reports/generated/paper_tables/table7_statistical_significance_summary.md` | Statistical significance | partial |
| `reports/generated/paper_tables/table8_robustness_limitation_matrix.md` | Robustness and limitation matrix | ready |
| `reports/generated/paper_figures/figure1_research_pipeline.md` | Research pipeline schematic | ready |
| `reports/generated/paper_figures/figure2_walk_forward_design.md` | Walk-forward design schematic | ready |
| `reports/generated/paper_figures/figure3_accuracy_by_model_horizon.png` | Accuracy by model and horizon | ready |
| `reports/generated/paper_figures/figure4_confidence_threshold_coverage_accuracy.png` | Confidence threshold versus coverage/accuracy | partial |
| `reports/generated/paper_figures/figure5_regime_specific_accuracy.png` | Regime-specific accuracy | ready |
| `reports/generated/vn100_artifact_date_schema_verification.md` | Date and schema verification | ready |
| `reports/generated/vn100_confidence_coverage_review.md` | Coverage-floor review | partial |
| `reports/generated/vn100_cost_slippage_readiness_review.md` | Cost/slippage readiness gap | ready |
| `reports/generated/vn100_ticker_concentration_summary.md` | Ticker concentration diagnostics | ready |
| `reports/generated/evidence_gap_closure/` | Evidence-upgrade diagnostics for cache coverage, v2 confidence sweep, ex-ante regime proxy, multi-window availability, and cost/slippage proxy | partial |
| `reports/NCKH_EVIDENCE_UPGRADE_ADDENDUM_V2.md` | Paper V2 evidence-upgrade addendum | ready |
| `reports/NCKH_FULL_PAPER_V2_UPDATE_PLAN.md` | Paper V2 update instructions | ready |

## Recommended Submission Order

1. `reports/NCKH_ABSTRACT_AND_KEYWORDS.md`
2. `reports/NCKH_FULL_PAPER_DRAFT_VN100_CLEAN.md`
3. `reports/generated/paper_tables/`
4. `reports/generated/paper_figures/`
5. `reports/NCKH_REFERENCES_APA7.md`
6. `reports/NCKH_FULL_PAPER_FINAL_REVIEW_CHECKLIST.md`
7. `reports/NCKH_RESULTS_CLAIM_REGISTER.md`
8. `reports/NCKH_DEFENSE_SUMMARY.md`
9. `reports/NCKH_PRESENTATION_OUTLINE.md`
10. `reports/NCKH_QUESTIONS_AND_ANSWERS.md`
11. `reports/NCKH_FINAL_SUBMISSION_NOTES.md`
12. `reports/NCKH_EVIDENCE_UPGRADE_ADDENDUM_V2.md`
13. `reports/NCKH_FULL_PAPER_V2_UPDATE_PLAN.md`
14. `reports/NCKH_RESULTS_CLAIM_REGISTER_V2.md`
15. `reports/NCKH_SUPERVISOR_BRIEF_AFTER_GAP_CLOSURE.md`

## Final Claim Boundary

- Global benchmark pass: no.
- Conditional signal: yes, selected confidence-filtered and bear-regime diagnostics.
- Stable full-market 63% method: no.
- Practical trading readiness: not established.
- Full VN100 representativeness: not established because official evidence covers seven evaluated tickers.
