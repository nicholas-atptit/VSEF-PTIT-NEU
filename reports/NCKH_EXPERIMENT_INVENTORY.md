# NCKH Experiment Inventory - VN100 Benchmark Foundation

## Purpose

This inventory maps the existing repository evidence relevant to a VN100 NCKH paper. It does not introduce new runtime behavior or new model logic. It identifies what is already available, what is generated locally, and what remains missing for a defensible research paper.

## Relevant Scripts

| Script | Role in NCKH workflow |
|---|---|
| `scripts/run_vn100_hybrid_frequency_accuracy_benchmark.py` | Main VN100 hybrid-frequency benchmark runner; supports cache-only/fetch-only modes, train cutoff, classification/regression targets, confidence filtering, regime diagnostics, significance summaries, and strategy-selection summaries. |
| `scripts/run_backtest_real_data.py` | Real-data backtest runner useful for future economic validation. |
| `scripts/run_backtest_model_comparison.py` | Compares model families in a backtest setting. |
| `scripts/run_backtest_forward_return.py` | Forward-return backtest runner for horizon-based experiments. |
| `scripts/run_dual_task_backtest.py` | Dual-task backtesting support for directional and return-style outcomes. |
| `scripts/run_strategy_backtest.py` | Strategy backtest entry point for future trading-readiness checks. |
| `scripts/run_signal_effectiveness_backtest.py` | Signal-effectiveness analysis support. |
| `scripts/run_walk_forward_regime_robustness.py` | Walk-forward regime robustness runner. |
| `scripts/run_regime_aware_analysis.py` | Regime-aware research analysis runner. |
| `scripts/research/analyze_vn100_ticker_concentration.py` | Lightweight official-artifact reader for ticker concentration diagnostics. |
| `scripts/research/build_vn100_paper_artifact_pack.py` | Deterministic official-artifact reader that builds paper-ready tables, figures, and notes without rerunning training or benchmarking. |
| `scripts/check_repo_hygiene.py` | Repository hygiene validation. |
| `scripts/check_runtime_preflight.py` | Environment and artifact preflight validation. |

## Relevant Source Modules

| Module | Relevance |
|---|---|
| `src/ml/metrics.py` | Directional accuracy helper used by tests and benchmark interpretation. |
| `src/ml/evaluation/metrics_engine.py` | Standardized metrics engine for experiment outputs. |
| `src/ml/benchmark/acceptance.py` | Acceptance-gate logic for benchmark claims. |
| `src/ml/benchmark/system_benchmark.py` | System-benchmark infrastructure for broader validation. |
| `src/ml/benchmark/baselines.py` | Baseline model support. |
| `src/ml/backtest/model_comparison.py` | Model comparison backtest support. |
| `src/ml/backtest/real_data.py` | Real-data backtest support. |
| `src/ml/backtest/forward_return.py` | Forward-return backtest support. |
| `src/ml/backtest/strategy_backtest.py` | Strategy backtest support. |
| `src/ml/backtest/regime_aware_analysis.py` | Regime-aware analysis support. |
| `src/ml/backtest/signal_effectiveness.py` | Signal-effectiveness diagnostics. |
| `src/ml/backtest/walk_forward_all_models_stacking.py` | Older walk-forward stacking path; useful for comparison but not the official VN100 runner. |
| `src/regime/labels.py` and `src/ml/regime/regime_detector.py` | Regime labeling/detection support. |
| `src/risk/*` and `src/ml/risk/*` | Risk metrics and future practical-readiness checks. |
| `src/strategy/*` | Strategy thresholding, sizing, and execution-policy support for future cost-aware evaluation. |

## Existing Reports

| Report | Use |
|---|---|
| `reports/WALK_FORWARD_BENCHMARKING_ENSEMBLE_BASELINE_MODELS_VN_STOCK_MARKET.md` | Current evidence-backed research narrative for VN100 walk-forward benchmark. |
| `reports/VN100_HYBRID_BENCHMARK_CLOSEOUT.md` | Closeout summary of official 2025 train-cutoff diagnostics and extended monitoring. |
| `reports/PHASE6_STATISTICAL_ACCEPTANCE_EVIDENCE.md` | Acceptance-governance evidence. |
| `reports/forecasting_core/FORECASTING_CORE_VALIDATION_REPORT.md` | Forecasting-core baseline and ensemble context. |
| `reports/regime_analysis/REGIME_AWARE_ANALYSIS_REPORT.md` | Prior regime-aware analysis evidence. |
| `reports/risk_aware/RISK_AWARE_DECISION_RESEARCH_REPORT.md` | Risk-aware decision research context. |
| `reports/audits/backtest_risk_audit.md` | Audit material; currently dirty and not part of this documentation-only commit. |

## Existing Outputs and Artifacts Present Locally

These output directories exist locally and should be treated as generated artifacts, not source files to commit:

| Output directory | Notes |
|---|---|
| `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff` | Official 2025 train-cutoff artifact family used by current report. |
| `outputs/vn100_hybrid_official_2025_confidence_sweep` | Earlier official-window attempt before train-cutoff separation; produced zero predictions. |
| `outputs/vn100_hybrid_accuracy_benchmark_phase123_full` | Extended monitoring diagnostics for Phase 1-3. |
| `outputs/vn100_hybrid_accuracy_benchmark_hourly_h1_tuned` | Focused tuning diagnostics. |
| `outputs/vn100_hybrid_accuracy_benchmark_cache_partial_usable` | Cache-only partial-usable extended run. |
| `outputs/vn100_hybrid_confidence_sweep_smoke` | Smoke confidence-sweep artifact. |
| `outputs/vn100_hybrid_fetch_full` | Fetch/full-cache artifact family. |
| `outputs/walkforward_*` | Earlier walk-forward experiment families useful as background, not official VN100 evidence. |

## Generated Evidence-Hardening Reports

| Report | Evidence added |
|---|---|
| `reports/generated/vn100_artifact_date_schema_verification.md` | Artifact-backed date/schema verification and corrected interpretation of the raw daily versus hybrid daily benchmark range. |
| `reports/generated/vn100_ticker_concentration_summary.csv` | Per-ticker prediction counts, accuracy, contribution share, and positive-edge share for global, selected-confidence, and best-regime scopes. |
| `reports/generated/vn100_ticker_concentration_summary.md` | Paper-readable concentration assessment; selected hourly confidence slice is concentrated in five tickers. |
| `reports/generated/vn100_confidence_coverage_review.md` | Coverage-floor review at 50%, 40%, and 30%; only the 30% floor has an available selected 60% pass, and daily sweep rows are missing. |
| `reports/generated/vn100_cost_slippage_readiness_review.md` | Trading-readiness gap review showing that official selected VN100 slices do not yet have cost-adjusted return, slippage, turnover, drawdown, or profit-factor artifacts. |
| `reports/NCKH_PAPER_TABLES_AND_FIGURES_PLAN.md` | Paper table and figure source map with fields, target chapter, supported claim, and evidence readiness status. |
| `reports/generated/paper_tables/` | Paper-ready CSV and Markdown versions of Tables 1-8. |
| `reports/generated/paper_figures/` | Paper-ready Markdown schematics for Figures 1-2 and PNG charts for Figures 3-5. |
| `reports/generated/paper_notes/` | Artifact-pack status and claim-boundary notes generated by the paper artifact builder. |
| `reports/NCKH_CHAPTER3_METHODOLOGY_DRAFT.md` | Evidence-backed Chapter 3 methodology draft. |
| `reports/NCKH_CHAPTER4_EMPIRICAL_RESULTS_DRAFT.md` | Evidence-backed Chapter 4 empirical results draft. |
| `reports/NCKH_RESULTS_CLAIM_REGISTER.md` | Claim register separating safe, conditional, and unsafe result claims. |
| `reports/NCKH_FULL_PAPER_DRAFT_VN100.md` | Full paper draft assembled from the design, outline, artifact pack, Chapter 3 draft, Chapter 4 draft, and claim register. |
| `reports/NCKH_ABSTRACT_AND_KEYWORDS.md` | Vietnamese and English titles, abstracts, and keywords. |
| `reports/NCKH_DEFENSE_SUMMARY.md` | Defense-ready 1-minute, 3-minute, and 5-minute summaries with safe committee answers. |
| `reports/NCKH_REFERENCES_APA7.md` | APA 7 reference scaffold using locally verified citation metadata from the existing benchmark report. |
| `reports/NCKH_FULL_PAPER_FINAL_REVIEW_CHECKLIST.md` | Final-review checklist for title, abstract, claims, tables, figures, references, and evidence limitations. |
| `reports/NCKH_FULL_PAPER_DRAFT_VN100_CLEAN.md` | Clean final-review manuscript with softened abstract artifact wording, citation placeholders, and table/figure insertion markers. |
| `reports/NCKH_SUBMISSION_PACKAGE_INDEX.md` | Final submission package index and recommended submission order. |
| `reports/NCKH_FINAL_SUBMISSION_NOTES.md` | Readiness summary, partial evidence list, prohibited claims, and final evidence gaps. |
| `reports/NCKH_PRESENTATION_OUTLINE.md` | 7-10 minute defense presentation outline with slide claim boundaries. |
| `reports/NCKH_QUESTIONS_AND_ANSWERS.md` | Committee Q&A preparation with safe answers. |

## Existing Tests Related to the Research

| Test path | Coverage area |
|---|---|
| `tests/ml/test_directional_accuracy_metrics.py` | Directional accuracy, confidence filtering, threshold sweep, train-cutoff leakage checks, and strategy-selection summary. |
| `tests/research/test_vn100_hybrid_60pct_accuracy_gate.py` | Fail-by-default VN100 60% research gate with output-directory override. |
| `tests/test_feature_engineering_vn100.py` | VN100 feature engineering contract. |
| `tests/test_data_loader_vn100.py` | VN100 data loader behavior. |
| `tests/test_vn100_adapters.py` | VN100 adapter behavior. |
| `tests/ml/test_walk_forward_all_models_stacking.py` | Walk-forward stacking behavior in legacy path. |
| `tests/ml/test_walk_forward_regime_robustness.py` | Regime robustness validation. |
| `tests/ml/test_model_comparison.py` | Model comparison backtest validation. |
| `tests/ml/test_forward_return_backtest.py` | Forward-return backtest validation. |
| `tests/ml/test_real_data_backtest.py` | Real-data backtest validation. |
| `tests/ml/test_strategy_backtest.py` | Strategy backtest artifacts and behavior. |
| `tests/ml/test_signal_effectiveness.py` and related heldout/rolling tests | Signal effectiveness and robustness. |
| `tests/ml/test_risk.py`, `tests/ml/test_risk_tuning.py`, `tests/risk_governance/*` | Risk and governance validation. |
| `tests/ml/test_benchmark_acceptance.py` | Acceptance-status evidence and benchmark promotion safeguards. |

## Current Validation Commands

Use these lightweight checks before committing documentation or benchmark-code changes:

```powershell
python -m py_compile scripts/run_vn100_hybrid_frequency_accuracy_benchmark.py
python -m pytest tests/ml/test_directional_accuracy_metrics.py -q
python scripts/check_repo_hygiene.py
python scripts/check_runtime_preflight.py
```

Heavy benchmark reruns should be avoided unless the experiment protocol explicitly calls for them.

## Missing Experiments Needed for a Defensible Paper

1. Expanded usable-cache coverage for a larger share of VN100.
2. Official 2025 rerun after coverage improvement with the same train-cutoff rule.
3. Multi-window walk-forward validation beyond the single 2025 evaluation year.
4. Expanded ticker concentration diagnostics after cache coverage improves; an initial concentration report now exists for the current seven evaluated tickers.
5. Cost-adjusted backtests with transaction costs, slippage, turnover, drawdown, and profit factor.
6. Broader coverage-constrained confidence sweeps at multiple minimum coverage levels; the current official artifact supports hourly stacking h=1 only, while daily sweep rows are missing.
7. Ex-ante regime rule validation so regime-specific findings are not post-hoc.
8. Baseline robustness across additional simple policies.
9. Statistical tests for selected strategy-level and regime-specific slices after multiple-testing controls are defined.
10. Paper-ready tables and charts generated from official artifacts rather than manually copied results.

## Current Claim Boundary

Safe claim: the global benchmark did not pass 60%, but selected confidence-filtered and bear-regime diagnostics show conditional signal.

Unsafe claim: the system is ready for live trading, guarantees profitability, or demonstrates a stable full-market 63% method.

## Evidence Gap Closure V1 Artifacts

This v2/evidence-upgrade section records the follow-up experiment-expansion artifacts. The phase did not add model families, did not change runtime/model logic, and did not rerun heavy benchmarks.

| Artifact | Use |
|---|---|
| `scripts/research/audit_vn100_cache_coverage.py` | Audits local VN100 cache coverage and official cache usability for expanded benchmark readiness. |
| `scripts/research/run_vn100_full_confidence_sweep.py` | Derives all available daily/hourly model/horizon confidence-threshold diagnostics from official prediction rows. |
| `scripts/research/run_vn100_multiwindow_validation.py` | Records 2022-2025 window availability and missing multi-window evidence without rerunning benchmarks. |
| `scripts/research/run_vn100_exante_regime_validation.py` | Builds lagged ex-ante regime diagnostics from prior realized rows only. |
| `scripts/research/run_vn100_cost_slippage_validation.py` | Produces cost/slippage-aware selected-signal diagnostic proxies and baselines from official predictions. |
| `reports/generated/evidence_gap_closure/` | Paper-readable CSV, Markdown, and PNG outputs for cache coverage, confidence sweep, multi-window availability, ex-ante regime validation, and cost/slippage diagnostics. |
| `reports/NCKH_EVIDENCE_GAP_CLOSURE_REGISTER.md` | Gap-by-gap closure status and claim-boundary impact. |
| `reports/NCKH_POST_GAP_CLOSURE_PAPER_UPDATE_NOTES.md` | Table, figure, abstract, and conclusion update guidance after the evidence upgrade. |

Evidence-upgrade conclusion: cache coverage and multi-window validation remain open gaps; confidence sweep breadth, ex-ante regime diagnostics, and cost/slippage diagnostic artifacts are partially closed. The global benchmark and trading-readiness claim boundaries do not change.

## Paper V2 Addendum Planning Artifacts

This paper-addendum phase converts the evidence-gap-closure outputs into manuscript update guidance without rewriting the full paper.

| Artifact | Use |
|---|---|
| `reports/NCKH_EVIDENCE_UPGRADE_ADDENDUM_V2.md` | Supervisor-readable addendum summarizing new evidence, remaining gaps, and Chapter 4 implications. |
| `reports/NCKH_FULL_PAPER_V2_UPDATE_PLAN.md` | Exact table, figure, Chapter 4, appendix, abstract, and conclusion update instructions for a later V2 manuscript. |
| `reports/NCKH_RESULTS_CLAIM_REGISTER_V2.md` | Updated claim governance separating unchanged safe claims, new derived-diagnostic claims, upgraded conditional claims, and still-unsafe claims. |
| `reports/NCKH_SUPERVISOR_BRIEF_AFTER_GAP_CLOSURE.md` | One-page supervisor-facing explanation of what improved, what did not, and recommended cautious wording. |

## Full Paper V2 Manuscript Artifacts

This full-manuscript phase rewrites the paper into a complete V2 draft using existing evidence-upgrade outputs. It does not add experiments, model logic, runtime behavior, or official benchmark artifacts.

| Artifact | Use |
|---|---|
| `reports/NCKH_FULL_PAPER_DRAFT_VN100_V2_WITH_FIGURES.md` | Complete V2 manuscript with updated evidence, embedded tables, embedded PNG figures, references, and appendices. |
| `reports/NCKH_FULL_PAPER_V2_TABLE_FIGURE_MAP.md` | Table/figure insertion map with source artifacts, claim support, status, and limitations. |
| `reports/NCKH_FULL_PAPER_V2_CHANGELOG.md` | Change summary from clean V1, new V2 evidence, strengthened claims, unchanged boundaries, unsafe claims, and remaining gaps. |
