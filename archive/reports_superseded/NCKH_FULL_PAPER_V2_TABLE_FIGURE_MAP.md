# NCKH Full Paper V2 Table and Figure Map

## Purpose

This map records the tables and figures inserted into `reports/NCKH_FULL_PAPER_DRAFT_VN100_V2_WITH_FIGURES.md`.

| Table | Figure | Source artifact | Manuscript location | Status | Claim supported | Limitation |
|---|---|---|---|---|---|---|
| Table 1: Dataset and Evaluation Scope |  | `reports/generated/paper_tables/table1_dataset_evaluation_scope.md`; `reports/generated/evidence_gap_closure/vn100_cache_coverage_audit.md` | Chapter 3.3 | Ready | Official evaluation uses a 2025 held-out window and seven benchmark-usable tickers. | Full VN100 representativeness is not established. |
|  | Figure 1: Research Pipeline | `reports/generated/paper_figures/figure1_research_pipeline.md` | Chapter 1.1 | Ready | The research pipeline separates cache validation, train-cutoff enforcement, evaluation, diagnostics, and claim governance. | Methodology schematic only. |
|  | Figure 2: Walk-Forward Validation Design | `reports/generated/paper_figures/figure2_walk_forward_design.md` | Chapter 3.2 | Ready | The 2025 evaluation is separated from the 2024-12-31 training-label cutoff. | Current evidence covers one official 2025 window. |
| Table 3: Global Benchmark Results | Figure 3: Accuracy by model/horizon | `reports/generated/paper_tables/table3_global_benchmark_results.md`; `reports/generated/paper_figures/figure3_accuracy_by_model_horizon.png` | Chapter 4.1 | Ready | Daily and hourly global benchmarks do not pass 60%. | Seven evaluated tickers and single 2025 window. |
| Table 4: Baseline Delta Summary |  | `reports/generated/paper_tables/table4_baseline_delta_summary.md` | Chapter 4.2 | Ready | Some model/horizon rows beat simple baselines. | Baseline deltas are not trading-return evidence. |
| Table 5: Original Confidence-Filtered Diagnostics |  | `reports/generated/paper_tables/table5_confidence_filtered_diagnostics.md` | Chapter 4.3 | Partial | Hourly stacking h=1 threshold 0.57 reaches 60.03% at 31.30% coverage. | Narrow selected slice, not a global pass. |
| Table 5B: V2 Coverage-Floor Confidence Candidates | Figure 4: V2 confidence threshold versus coverage/accuracy | `reports/generated/evidence_gap_closure/vn100_full_confidence_sweep_report.md`; `reports/generated/evidence_gap_closure/vn100_confidence_threshold_coverage_accuracy_v2.png` | Chapter 4.4 | Partial | Derived V2 sweep broadens confidence-filter evidence across 32 combinations and 154,048 rows. | Derived from official predictions; not a new official rerun. |
| Table 6: Regime-Specific Diagnostics | Figure 5: Regime-specific accuracy | `reports/generated/paper_tables/table6_regime_specific_diagnostics.md`; `reports/generated/paper_figures/figure5_regime_specific_accuracy.png` | Chapter 4.5 | Ready | Daily bear-regime h=20 diagnostics exceed 63% in 2025. | Original regime diagnostics are post-hoc unless paired with ex-ante proxy. |
| Table 6B: Lagged Ex-Ante Bear-Regime Diagnostics |  | `reports/generated/evidence_gap_closure/vn100_exante_regime_accuracy_summary.csv` | Chapter 4.6 | Partial | Lagged ex-ante bear-regime proxy supports the 2025 daily h=20 diagnostic. | Single 2025 window; no multi-window regime stability. |
| Table 7: Statistical Significance Summary |  | `reports/generated/paper_tables/table7_statistical_significance_summary.md` | Chapter 4.7 | Partial | Several rows are statistically above a 50% null. | Multiple-testing and selected-slice issues remain. |
| Table 8: Evidence Gap Closure Status |  | `reports/NCKH_EVIDENCE_GAP_CLOSURE_REGISTER.md` | Chapter 4.10 and Appendix C | Ready | Evidence gaps are explicitly tracked as closed, partially closed, or not closed. | Several material gaps remain unresolved. |
| Appendix Table B1: Model-Signal Proxy Results | Appendix Figure B1: Cost/slippage proxy equity curve | `reports/generated/evidence_gap_closure/vn100_cost_slippage_validation_report.md`; `reports/generated/evidence_gap_closure/vn100_equity_curve.png` | Appendix B and Chapter 4.9 | Partial | Cost/slippage proxy diagnostics now exist. | Not execution-ready trading evidence. |

## Embedded PNG Figures

- `reports/generated/paper_figures/figure3_accuracy_by_model_horizon.png`
- `reports/generated/evidence_gap_closure/vn100_confidence_threshold_coverage_accuracy_v2.png`
- `reports/generated/paper_figures/figure5_regime_specific_accuracy.png`
- `reports/generated/evidence_gap_closure/vn100_equity_curve.png`

## Markdown Schematics Included as Text

- `reports/generated/paper_figures/figure1_research_pipeline.md`
- `reports/generated/paper_figures/figure2_walk_forward_design.md`
