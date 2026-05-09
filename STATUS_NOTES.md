[35mconfigs/policies/candidate_policy_forecast_only.yaml[m[36m:[m[32m66[m[36m:[m    - BUY [1;31mrecommendation[m
[35mconfigs/policies/candidate_policy_forecast_only.yaml[m[36m:[m[32m67[m[36m:[m    - SELL [1;31mrecommendation[m
[35mconfigs/policies/candidate_policy_forecast_only.yaml[m[36m:[m[32m68[m[36m:[m    - HOLD [1;31mrecommendation[m
[35mconfigs/policies/candidate_policy_risk_aware.yaml[m[36m:[m[32m72[m[36m:[m    - BUY [1;31mrecommendation[m
[35mconfigs/policies/candidate_policy_risk_aware.yaml[m[36m:[m[32m73[m[36m:[m    - SELL [1;31mrecommendation[m
[35mconfigs/policies/candidate_policy_risk_aware.yaml[m[36m:[m[32m74[m[36m:[m    - HOLD [1;31mrecommendation[m
[35mdocs/AUTHORITY_BOUNDARY.md[m[36m:[m[32m19[m[36m:[mSELL [1;31mrecommendation[ms, live execution instructions, production trading authority,
[35mdocs/AUTHORITY_BOUNDARY.md[m[36m:[m[32m43[m[36m:[m- BUY [1;31mrecommendation[m
[35mdocs/AUTHORITY_BOUNDARY.md[m[36m:[m[32m44[m[36m:[m- SELL [1;31mrecommendation[m
[35mdocs/AUTHORITY_BOUNDARY.md[m[36m:[m[32m55[m[36m:[m| Quant Core | Forecast, consensus, model-health, regime, risk, policy, packet, and legacy candidate diagnostics. | BUY/SELL [1;31mrecommendation[m or production trading readiness. |
[35mdocs/AUTHORITY_BOUNDARY.md[m[36m:[m[32m60[m[36m:[m| Phase 3 Router v1 | `route_allocation_candidate`, `hold`, `reject`, or `no_candidate` diagnostics. | Live execution, final [1;31mrecommendation[m, learned meta-router, or production trading authority. |
[35mdocs/AUTHORITY_BOUNDARY.md[m[36m:[m[32m75[m[36m:[m- [1;31mrecommendation[m
[35mdocs/DECISION_DIAGNOSTIC_CHAIN.md[m[36m:[m[32m26[m[36m:[mbounded review decisions. It does not create BUY or SELL [1;31mrecommendation[ms.
[35mdocs/DECISION_DIAGNOSTIC_CHAIN.md[m[36m:[m[32m38[m[36m:[m| Diagnostic-only boundary | Forecasts, consensus, policy, and candidate surfaces are diagnostics. They are not BUY or SELL [1;31mrecommendation[ms. |
[35mdocs/DECISION_DIAGNOSTIC_CHAIN.md[m[36m:[m[32m62[m[36m:[m| Diagnostic-only boundary | Risk actions are candidate filters and confidence adjustments. They are not [1;31mrecommendation[m or execution instructions. |
[35mdocs/DECISION_DIAGNOSTIC_CHAIN.md[m[36m:[m[32m84[m[36m:[m| Downstream consumer | Phase 3 Router v1 consumes [1;31mportfolio allocation[m, summary, risk summary, and optional allocator manifest. |
[35mdocs/DECISION_DIAGNOSTIC_CHAIN.md[m[36m:[m[32m86[m[36m:[m| Diagnostic-only boundary | `allocation_candidate` is a bounded diagnostic allocation row, not a portfolio order or [1;31mrecommendation[m. |
[35mdocs/README.md[m[36m:[m[32m57[m[36m:[m| [1;31mPortfolio allocation[m rules | [docs/governance/PORTFOLIO_ALLOCATOR_OUTPUT_SCHEMA.md](governance/PORTFOLIO_ALLOCATOR_OUTPUT_SCHEMA.md) |
[35mdocs/SYSTEM_OVERVIEW.md[m[36m:[m[32m21[m[36m:[mThe system does not emit BUY or SELL [1;31mrecommendation[ms. It does not execute
[35mdocs/architecture/VSEF_v1_ARCHITECTURE.md[m[36m:[m[32m125[m[36m:[m[1;31mrecommendation[ms, not order instructions, and not proof of profitable trades.
[35mdocs/archive/legacy_docs_2026_05_05/DEPRECATION_MAP.md[m[36m:[m[32m34[m[36m:[mDeprecation mapping does not grant BUY or SELL [1;31mrecommendation[m authority, live
[35mdocs/archive/legacy_docs_2026_05_05/INDEX.md[m[36m:[m[32m33[m[36m:[mdo not grant BUY or SELL [1;31mrecommendation[m authority, live execution authority,
[35mdocs/archive/phases/2026-04-19_Sunday__PHASE26_CALIBRATION_DECISION.md[m[36m:[m[32m40[m[36m:[m## [1;31mRecommendation[m
[35mdocs/archive/phases/2026-04-19_Sunday__PHASE26_CALIBRATION_DECISION.md[m[36m:[m[32m42[m[36m:[m- direct [1;31mrecommendation[m: `stop policy work and revisit forecast layer`
[35mdocs/archive/phases/2026-04-19_Sunday__PHASEF15_NARROW_REHAB_DECISION.md[m[36m:[m[32m61[m[36m:[m## [1;31mRecommendation[m
[35mdocs/archive/phases/2026-04-19_Sunday__PHASEF15_NARROW_REHAB_DECISION.md[m[36m:[m[32m63[m[36m:[mDirect [1;31mrecommendation[m: `freeze all but the best one or two model families`
[35mdocs/archive/phases/2026-04-19_Sunday__PHASEF15_NARROW_REHAB_DECISION.md[m[36m:[m[32m82[m[36m:[m- The narrow slice is materially better than the broader F1 rehab, but the evidence is still too concentrated to justify routing, stacking, [1;31mportfolio allocation[m, or deep sequence expansion.
[35mdocs/archive/phases/2026-04-19_Sunday__PHASEF1_FORECAST_REHAB_DECISION.md[m[36m:[m[32m15[m[36m:[m## [1;31mRecommendation[m
[35mdocs/archive/root/2026-03-29_Sunday__upgrade_execution_plan_local_ollama_2026-03-28.md[m[36m:[m[32m36[m[36m:[m2. **Multi-agent**: có pipeline debate Bull/Bear + Risk veto + [1;31mPortfolio allocation[m.  
[35mdocs/archive/root/BASELINE.md[m[36m:[m[32m19[m[36m:[m- [1;31mPortfolio allocation[m
[35mdocs/audits/DECISION_CHAIN_AUDIT_2026_05_05.md[m[36m:[m[32m70[m[36m:[mThis audit does not grant BUY or SELL [1;31mrecommendation[m authority, live execution
[35mdocs/audits/VSEF_HELDOUT_THRESHOLD_SELECTION.md[m[36m:[m[32m48[m[36m:[m1. maximize BUY precision subject to the configured minimum [1;31mBUY signal[m count
[35mdocs/audits/VSEF_HELDOUT_THRESHOLD_SELECTION.md[m[36m:[m[32m50[m[36m:[m3. tie-break by larger [1;31mBUY signal[m count
[35mdocs/audits/VSEF_QUANT_CORE_REPEATED_SEED_STABILITY_AUDIT.md[m[36m:[m[32m21[m[36m:[mThe audit does not claim improved accuracy, production trading readiness, robust edge, or final buy [1;31mrecommendation[ms.
[35mdocs/audits/VSEF_QUANT_CORE_REPEATED_SEED_STABILITY_AUDIT.md[m[36m:[m[32m236[m[36m:[mDecision-lane candidates, if present, remain diagnostic outputs. They are not final buy [1;31mrecommendation[ms.
[35mdocs/audits/VSEF_QUANT_CORE_SMOKE_AUDIT.md[m[36m:[m[32m19[m[36m:[mThis audit does not claim production trading readiness, robust edge, or final buy [1;31mrecommendation[ms.
[35mdocs/audits/VSEF_QUANT_CORE_SMOKE_AUDIT.md[m[36m:[m[32m198[m[36m:[mThese rows are smoke-mode review candidates only. They are not final buy [1;31mrecommendation[ms, not a live decision policy, and not evidence of robust trading edge.
[35mdocs/audits/VSEF_QUANT_CORE_SMOKE_AUDIT.md[m[36m:[m[32m251[m[36m:[m- It does not prove that decision candidates are final buy [1;31mrecommendation[ms.
[35mdocs/audits/VSEF_ROLLING_REGIME_THRESHOLD_SELECTION.md[m[36m:[m[32m60[m[36m:[m2. maximize BUY precision subject to the configured minimum [1;31mBUY signal[m count
[35mdocs/audits/VSEF_ROLLING_REGIME_THRESHOLD_SELECTION.md[m[36m:[m[32m111[m[36m:[m- `insufficient`: not enough [1;31mBUY signal[ms
[35mdocs/audits/VSEF_ROLLING_REGIME_THRESHOLD_SELECTION.md[m[36m:[m[32m146[m[36m:[mInterpretation: the 70% BUY precision target did not survive across rolling folds. It passed only for `short_20d` in fold 2, where held-out BUY precision was `0.840000` with 150 [1;31mBUY signal[ms. No regime-conditioned results were produced because the saved prediction output did not contain a recognized regime column. A later join layer can enrich a copy of the prediction file once safe precomputed regime labels are available.
[35mdocs/audits/VSEF_SIGNAL_EFFECTIVENESS_BACKTEST.md[m[36m:[m[32m69[m[36m:[msuccessful [1;31mBUY signal[ms / total [1;31mBUY signal[ms
[35mdocs/audits/VSEF_SIGNAL_EFFECTIVENESS_BACKTEST.md[m[36m:[m[32m75[m[36m:[msuccessful [1;31mBUY signal[ms / all successful realized cases under the selected success definition
[35mdocs/audits/VSEF_SIGNAL_EFFECTIVENESS_BACKTEST.md[m[36m:[m[32m121[m[36m:[m`buy_precision_by_model_horizon.csv` includes only model-horizon-threshold rows that pass the configured minimum [1;31mBUY signal[m count. The full frontier remains available in `precision_coverage_frontier.csv`.
[35mdocs/experiments/PHASE1_EXPERIMENT_STANDARDIZATION.md[m[36m:[m[32m344[m[36m:[m  - [1;31mPortfolio allocation[m, broker execution, live trading, and autonomous agents
[35mdocs/governance/DECISION_LANE_OUTPUT_SCHEMA.md[m[36m:[m[32m17[m[36m:[mdoes not create BUY or SELL [1;31mrecommendation[ms. It enriches the legacy
[35mdocs/governance/DECISION_LANE_OUTPUT_SCHEMA.md[m[36m:[m[32m154[m[36m:[m- no BUY/SELL [1;31mrecommendation[m authority
[35mdocs/governance/PHASE3_ROUTER_OUTPUT_SCHEMA.md[m[36m:[m[32m18[m[36m:[mSELL [1;31mrecommendation[m authority and does not train models.
[35mdocs/governance/PHASE3_ROUTER_OUTPUT_SCHEMA.md[m[36m:[m[32m115[m[36m:[m- `no_buy_sell_[1;31mrecommendation[m_authority`
[35mdocs/governance/PHASE3_ROUTER_OUTPUT_SCHEMA.md[m[36m:[m[32m131[m[36m:[m- `no_buy_sell_[1;31mrecommendation[m_authority`
[35mdocs/governance/PHASE3_ROUTER_OUTPUT_SCHEMA.md[m[36m:[m[32m147[m[36m:[m- no BUY/SELL [1;31mrecommendation[m authority
[35mdocs/governance/PHASE3_ROUTER_V1.md[m[36m:[m[32m20[m[36m:[m[1;31mrecommendation[m or execution layer is considered.
[35mdocs/governance/PHASE3_ROUTER_V1.md[m[36m:[m[32m23[m[36m:[m[1;31mrecommendation[ms, does not execute trades, and does not train a learned
[35mdocs/governance/PHASE3_ROUTER_V1.md[m[36m:[m[32m119[m[36m:[m## Why This Is Not a BUY/SELL [1;31mRecommendation[m
[35mdocs/governance/PHASE3_ROUTER_V1.md[m[36m:[m[32m132[m[36m:[m| The router creates final BUY or SELL [1;31mrecommendation[ms. | No | [1;31mRecommendation[m authority remains out of scope. |
[35mdocs/governance/PHASE3_ROUTER_V1.md[m[36m:[m[32m134[m[36m:[m| The router grants production trading authority. | No | Live execution, monitoring, validation, and [1;31mrecommendation[m governance are not implemented. |
[35mdocs/governance/PHASE3_ROUTER_V1.md[m[36m:[m[32m178[m[36m:[m- Clear governance labels that separate candidates from [1;31mrecommendation[ms.
[35mdocs/governance/PORTFOLIO_ALLOCATOR_OUTPUT_SCHEMA.md[m[36m:[m[32m18[m[36m:[m[1;31mrecommendation[m authority.
[35mdocs/governance/PORTFOLIO_ALLOCATOR_OUTPUT_SCHEMA.md[m[36m:[m[32m199[m[36m:[m- `no_buy_sell_[1;31mrecommendation[m_authority`
[35mdocs/governance/PORTFOLIO_ALLOCATOR_OUTPUT_SCHEMA.md[m[36m:[m[32m233[m[36m:[m- `no_buy_sell_[1;31mrecommendation[m_authority`
[35mdocs/governance/PORTFOLIO_ALLOCATOR_OUTPUT_SCHEMA.md[m[36m:[m[32m247[m[36m:[m- no BUY/SELL [1;31mrecommendation[m authority
[35mdocs/governance/PORTFOLIO_ALLOCATOR_V1.md[m[36m:[m[32m33[m[36m:[mPortfolio Allocator v1 does not emit BUY or SELL [1;31mrecommendation[ms, live
[35mdocs/governance/PORTFOLIO_ALLOCATOR_V1.md[m[36m:[m[32m135[m[36m:[m| The allocator emits BUY or SELL [1;31mrecommendation[ms. | No | The layer emits only diagnostic `allocation_candidate` or `no_allocation` states. |
[35mdocs/governance/PROJECT_TRACKER.md[m[36m:[m[32m19[m[36m:[mengines, [1;31mportfolio allocation[m logic, agents, APIs, dashboards, data providers,
[35mdocs/governance/QUANT_CORE_GOVERNANCE.md[m[36m:[m[32m28[m[36m:[mQuant Core is not a [1;31mrecommendation[m layer. Its outputs are forecast diagnostics,
[35mdocs/governance/QUANT_CORE_GOVERNANCE.md[m[36m:[m[32m51[m[36m:[mNo layer emits BUY or SELL [1;31mrecommendation[ms, live execution instructions,
[35mdocs/governance/QUANT_CORE_GOVERNANCE.md[m[36m:[m[32m156[m[36m:[m- BUY or SELL [1;31mrecommendation[m authority
[35mdocs/governance/QUANT_CORE_OUTPUT_SCHEMA.md[m[36m:[m[32m20[m[36m:[mThey do not emit BUY or SELL [1;31mrecommendation[ms, live execution instructions,
[35mdocs/governance/QUANT_CORE_OUTPUT_SCHEMA.md[m[36m:[m[32m146[m[36m:[mallocation artifacts and `run_counts` records [1;31mportfolio allocation[m rows and
[35mdocs/governance/QUANT_CORE_OUTPUT_SCHEMA.md[m[36m:[m[32m465[m[36m:[mdiagnostic-only authority, and no BUY/SELL [1;31mrecommendation[m authority.
[35mdocs/governance/QUANT_CORE_OUTPUT_SCHEMA.md[m[36m:[m[32m493[m[36m:[m  no forced trade rule, and no BUY/SELL [1;31mrecommendation[m authority.
[35mdocs/governance/QUANT_CORE_OUTPUT_SCHEMA.md[m[36m:[m[32m549[m[36m:[m  artifact paths, and no BUY/SELL [1;31mrecommendation[m authority.
[35mdocs/governance/QUANT_CORE_OUTPUT_SCHEMA.md[m[36m:[m[32m576[m[36m:[m- `no_buy_sell_[1;31mrecommendation[m_authority`
[35mdocs/governance/RISK_GOVERNANCE_OUTPUT_SCHEMA.md[m[36m:[m[32m18[m[36m:[mreason codes. It does not emit BUY or SELL [1;31mrecommendation[ms, live execution
[35mdocs/governance/RISK_GOVERNANCE_OUTPUT_SCHEMA.md[m[36m:[m[32m227[m[36m:[m- no BUY/SELL [1;31mrecommendation[m authority
[35mdocs/governance/SCENARIO_EVALUATION_ENGINE.md[m[36m:[m[32m20[m[36m:[mThe engine is diagnostic only. It does not emit BUY or SELL [1;31mrecommendation[ms,
[35mdocs/governance/SCENARIO_OUTPUT_SCHEMA.md[m[36m:[m[32m17[m[36m:[mdoes not emit BUY or SELL [1;31mrecommendation[ms, live execution instructions,
[35mdocs/prompt_runs/archive/analysis_feed/12_daily_brief.md[m[36m:[m[32m24[m[36m:[m- **Action [1;31mRecommendation[ms**: Maps predictions to specific actions (BUY, HOLD, SELL, WAIT).
[35mdocs/prompt_runs/archive/analysis_feed/12_daily_brief.md[m[36m:[m[32m47[m[36m:[m- Action [1;31mrecommendation[ms are rule-based and not backtested yet.
[35mdocs/roadmap/DECISION_PHYSICS_NEXT_STEPS.md[m[36m:[m[32m27[m[36m:[mDo not add BUY or SELL [1;31mrecommendation[m authority, live execution authority,
[35mdocs/roadmap/SOCIAL_LISTENING_INTEGRATION.md[m[36m:[m[32m56[m[36m:[m- create BUY or SELL [1;31mrecommendation[ms
[35mreports/README.md[m[36m:[m[32m15[m[36m:[mDo not treat reports as BUY or SELL [1;31mrecommendation[m authority, live execution
[35mreports/VSEF_1000_SEED_SMOKE_STABILITY_REPORT.md[m[36m:[m[32m204[m[36m:[mDecision-candidate rows are diagnostics only. They are not final buy [1;31mrecommendation[ms and should not be converted into trading actions without separate validation.
[35mreports/VSEF_PHASE3_ALLOCATOR_ROUTER_REAL_OUTPUT_SMOKE_REPORT.md[m[36m:[m[32m21[m[36m:[mPortfolio Allocator v1 and Phase 3 Router v1 emit governed candidates and route decisions only; they do not emit final BUY [1;31mrecommendation[ms.
[35mreports/VSEF_PHASE3_ALLOCATOR_ROUTER_REAL_OUTPUT_SMOKE_REPORT.md[m[36m:[m[32m25[m[36m:[mThis report validates artifact compatibility and conservative orchestration behavior on 10 real saved Quant Core seed outputs. It is not a heavy training run, not a 1000-seed rerun, not medium or `decision_core` validation, not production validation, and not final [1;31mrecommendation[m generation.
[35mreports/VSEF_PHASE3_ALLOCATOR_ROUTER_REAL_OUTPUT_SMOKE_REPORT.md[m[36m:[m[32m61[m[36m:[m## [1;31mRecommendation[m Boundary Scan
[35mreports/VSEF_PHASE3_ALLOCATOR_ROUTER_REAL_OUTPUT_SMOKE_REPORT.md[m[36m:[m[32m63[m[36m:[mA broad forbidden-term scan was run across the smoke output root. The scan found the lowercase word `buy` only inside negative governance text such as `not a buy [1;31mrecommendation[m` in allocator manifests and allocator decision cards.
[35mreports/VSEF_PHASE3_ALLOCATOR_ROUTER_REAL_OUTPUT_SMOKE_REPORT.md[m[36m:[m[32m65[m[36m:[mNo `BUY`, `SELL`, `final_[1;31mrecommendation[m`, or `production_ready` labels or [1;31mrecommendation[m outputs were emitted by the saved allocator or router artifacts.
[35mreports/VSEF_PHASE3_ALLOCATOR_ROUTER_REAL_OUTPUT_SMOKE_REPORT.md[m[36m:[m[32m128[m[36m:[mThis is a successful conservative smoke validation, not a failure. The allocator and router processed real saved artifacts, wrote their expected output surfaces, preserved governance boundaries, and did not emit [1;31mrecommendation[m labels.
[35mreports/VSEF_PHASE3_ALLOCATOR_ROUTER_REAL_OUTPUT_SMOKE_REPORT.md[m[36m:[m[32m138[m[36m:[m- This did not generate final [1;31mrecommendation[ms.
[35mreports/VSEF_QUANT_CORE_PHASE3_GAP_CLOSURE_PLAN.md[m[36m:[m[32m20[m[36m:[mThat does not close the decision architecture. The current Quant Core has governed forecasts, consensus summaries, model-health summaries, analysis packets, and conservative decision-lane candidates, but those candidates are diagnostic artifacts. They are not official BUY [1;31mrecommendation[ms, they are not [1;31mportfolio allocation[ms, and they are not Phase 3 routed decisions.
[35mreports/VSEF_QUANT_CORE_PHASE3_GAP_CLOSURE_PLAN.md[m[36m:[m[32m36[m[36m:[m- Generated artifacts remain evidence surfaces, not final [1;31mrecommendation[ms.
[35mreports/VSEF_QUANT_CORE_PHASE3_GAP_CLOSURE_PLAN.md[m[36m:[m[32m42[m[36m:[m| Decision lane | Produces conservative `decision_lane_candidates.csv` rows for analyst review. | A formal separation between diagnostic candidates, allocation candidates, and final [1;31mrecommendation[ms. |
[35mreports/VSEF_QUANT_CORE_PHASE3_GAP_CLOSURE_PLAN.md[m[36m:[m[32m48[m[36m:[mThe current decision lane is a candidate generator. In `src/reporting/analysis_packets.py`, `build_decision_lane_candidates` filters packets that are tradable, positively predicted by the primary model, in a medium or high agreement bucket, and supported by at least one active policy signal. This is useful for analyst review, but it is not a final [1;31mrecommendation[m engine.
[35mreports/VSEF_QUANT_CORE_PHASE3_GAP_CLOSURE_PLAN.md[m[36m:[m[32m55[m[36m:[m| `[1;31mRecommendation[mCard` | Describes a final [1;31mrecommendation[m after allocation, risk checks, and validation gates. | Disabled until allocator and validation exist. |
[35mreports/VSEF_QUANT_CORE_PHASE3_GAP_CLOSURE_PLAN.md[m[36m:[m[32m60[m[36m:[m- `allocation_candidate`: candidate accepted by allocator pre-checks but not a final [1;31mrecommendation[m.
[35mreports/VSEF_QUANT_CORE_PHASE3_GAP_CLOSURE_PLAN.md[m[36m:[m[32m61[m[36m:[m- `final_[1;31mrecommendation[m`: disabled until Portfolio Allocator v1 and validation gates exist.
[35mreports/VSEF_QUANT_CORE_PHASE3_GAP_CLOSURE_PLAN.md[m[36m:[m[32m63[m[36m:[m`final_[1;31mrecommendation[m` should remain unavailable in manifests, cards, and UI copy until [1;31mportfolio allocation[m, risk checks, and validation evidence are implemented. This avoids implying BUY [1;31mrecommendation[m capability from a diagnostic candidate file.
[35mreports/VSEF_QUANT_CORE_PHASE3_GAP_CLOSURE_PLAN.md[m[36m:[m[32m99[m[36m:[mAllocator v1 should produce allocation candidates, not final BUY [1;31mrecommendation[ms. The output can say that a ticker receives a bounded proposed weight under a diagnostic allocation policy. It should not say the system recommends buying that ticker.
[35mreports/VSEF_QUANT_CORE_PHASE3_GAP_CLOSURE_PLAN.md[m[36m:[m[32m145[m[36m:[m| Phase A | Governance/schema clarification | Add `CandidateCard`, `[1;31mRecommendation[mCard`, and decision-label rules; keep `final_[1;31mrecommendation[m` disabled. |
[35mreports/VSEF_QUANT_CORE_PHASE3_GAP_CLOSURE_PLAN.md[m[36m:[m[32m156[m[36m:[m| The system has final BUY [1;31mrecommendation[ms. | No | Current `decision_lane_candidates.csv` is a diagnostic candidate file, not an allocator-backed [1;31mrecommendation[m surface. |
[35mreports/VSEF_QUANT_CORE_PHASE3_GAP_CLOSURE_PLAN.md[m[36m:[m[32m173[m[36m:[mDo NOT claim BUY [1;31mrecommendation[m capability.
[35mreports/VSEF_QUANT_CORE_PHASE3_GAP_CLOSURE_PLAN.md[m[36m:[m[32m203[m[36m:[m- label outputs as allocation_candidate, not final_[1;31mrecommendation[m
[35mreports/VSEF_QUANT_CORE_PHASE3_GAP_CLOSURE_PLAN.md[m[36m:[m[32m210[m[36m:[m5. Add documentation that final_[1;31mrecommendation[m remains disabled.
[35mreports/audits/AUDIT_REPORT_MASTER.md[m[36m:[m[32m155[m[36m:[m**Next Step [1;31mRecommendation[m**: Perform **Phase 1 (Essential Validity)** immediately to ensure prediction accuracy before further model tuning.
[35mreports/audits/backtest_risk_audit.md[m[36m:[m[32m37[m[36m:[m- **[1;31mRecommendation[m**: Integrate the `VNINDEX` benchmark into the `PaperTradingEngine` for every cycle.
[35mreports/audits/backtest_risk_audit.md[m[36m:[m[32m46[m[36m:[m- **[1;31mRecommendation[m**: Implement a `WalkForwardEngine` that retrains models every N sessions during long-span backtests to capture regime shifts.
[35mreports/audits/data_flow_audit.md[m[36m:[m[32m13[m[36m:[m- **[1;31mRecommendation[m**: Standardize all ingestion through a central `DataLoader` that handles version tracking.
[35mreports/audits/data_flow_audit.md[m[36m:[m[32m30[m[36m:[m- **[1;31mRecommendation[m**: Transition to a partitioned Parquet structure for faster I/O and better compression.
[35mreports/audits/data_flow_audit.md[m[36m:[m[32m44[m[36m:[m- **[1;31mRecommendation[m**: Use an artifact registry or a compressed model bundle for daily releases.
[35mreports/audits/data_flow_audit.md[m[36m:[m[32m49[m[36m:[m- **[1;31mRecommendation[m**: Centralize all data paths in `src/data/universe.py` or a dedicated `data_registry.py`.
[35mreports/audits/repo_structure_audit.md[m[36m:[m[32m29[m[36m:[m- **[1;31mRecommendation[m**: Move all ML parameters to `config/settings.py` or a dedicated `ml_params.yaml`.
[35mreports/audits/repo_structure_audit.md[m[36m:[m[32m39[m[36m:[m- **[1;31mRecommendation[m**: Restrict the training universe to VN100 by default and move other models to a `legacy/` or `archive/` folder.
[35mreports/audits/technical_prediction_audit.md[m[36m:[m[32m17[m[36m:[m- **[1;31mRecommendation[m**: Standardize on Yang-Zhang for open-gap sensitive volatility and HV for simpler drift.
[35mreports/audits/technical_prediction_audit.md[m[36m:[m[32m36[m[36m:[m- **[1;31mRecommendation[m**: Create a `labeling.py` or `feature_registry.py` that strictly isolates `close_raw` for all target/label generation.
[35mreports/audits/technical_prediction_audit.md[m[36m:[m[32m63[m[36m:[m- **[1;31mRecommendation[m**: Centralize data loading into a single robust class to ensure features and scaling are identical between `fit` and `predict`.
[35mreports/comparison_qwen_4b_vs_8b.md[m[36m:[m[32m37[m[36m:[m**[1;31mRecommendation[m**: The system's resilience mechanism works identically across both models. `qwen3:4b` is a viable alternative to reduce inference wait times if pulled.
[35mreports/full_system_report.md[m[36m:[m[32m69[m[36m:[m## 7. [1;31mRecommendation[m
[35mreports/risk_aware/EXP-RK-001_CANDIDATE_COMPARISON.md[m[36m:[m[32m64[m[36m:[mAll Phase 3 outputs are diagnostic decision-support research artifacts only. They are not BUY / SELL / HOLD advice, capital allocation guidance, broker execution instructions, portfolio [1;31mrecommendation[ms, or proof of guaranteed profitable trading.
[35mreports/risk_aware/EXP-RK-002_BASKET_EVALUATION.md[m[36m:[m[32m60[m[36m:[mAll Phase 3 outputs are diagnostic decision-support research artifacts only. They are not BUY / SELL / HOLD advice, capital allocation guidance, broker execution instructions, portfolio [1;31mrecommendation[ms, or proof of guaranteed profitable trading.
[35mreports/risk_aware/RISK_AWARE_DECISION_RESEARCH_REPORT.md[m[36m:[m[32m205[m[36m:[m- Candidate baskets are equal-weight diagnostic baskets, not [1;31mportfolio allocation[ms.
[35mreports/risk_aware/RISK_AWARE_DECISION_RESEARCH_REPORT.md[m[36m:[m[32m229[m[36m:[mAll Phase 3 outputs are diagnostic decision-support research artifacts only. They are not BUY / SELL / HOLD advice, capital allocation guidance, broker execution instructions, portfolio [1;31mrecommendation[ms, or proof of guaranteed profitable trading.
[35mscripts/generate_risk_aware_report.py[m[36m:[m[32m561[m[36m:[m            "- Candidate baskets are equal-weight diagnostic baskets, not [1;31mportfolio allocation[ms.",
[35mscripts/legacy/run_phase25_hardening.py[m[36m:[m[32m354[m[36m:[m    print(f"[1;31mRecommendation[m: {assessment['[1;31mrecommendation[m']}")
[35mscripts/legacy/run_phase26_calibration.py[m[36m:[m[32m341[m[36m:[m    print(f"[1;31mRecommendation[m: {assessment['[1;31mrecommendation[m']}")
[35mscripts/run_agent_decision.py[m[36m:[m[32m68[m[36m:[m    print(f"Action [1;31mRecommendation[m: {output['trading_decision']['action']}")
[35mscripts/run_candidate_research.py[m[36m:[m[32m35[m[36m:[m    "execution instructions, portfolio [1;31mrecommendation[ms, or proof of guaranteed "
[35mscripts/run_candidate_research.py[m[36m:[m[32m329[m[36m:[m        "trade " + "[1;31mrecommendation[m",
[35mscripts/run_forecast_rehab.py[m[36m:[m[32m422[m[36m:[m    print(f"[1;31mRecommendation[m: {assessment.get('[1;31mrecommendation[m')}")
[35mscripts/run_forecast_rehab_narrow.py[m[36m:[m[32m462[m[36m:[m    print(f"[1;31mRecommendation[m: {assessment.get('[1;31mrecommendation[m')}")
[35mscripts/run_quant_core.py[m[36m:[m[32m443[m[36m:[m        print(f"[1;31mPortfolio allocation[m rows: {len(portfolio_allocator_result.allocation)}")
[35mscripts/run_signal_effectiveness_backtest.py[m[36m:[m[32m119[m[36m:[m        help="Comma-separated minimum [1;31mBUY signal[m counts for filtered precision tables",
