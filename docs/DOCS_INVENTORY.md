# Documentation Inventory
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Documentation inventory |
| Created / authored | Tuesday, 2026-05-05 00:00:00 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-05-05 00:00:00 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Timestamp source | Local documentation inventory and legacy migration |
| Status | Active |

## Scope

This inventory covers:

- `docs/**/*.md`
- `docs/*.md`
- top-level `reports/*.md` for listing only

Reports are historical snapshots and are not canonical governance docs. Active canonical docs override archived docs and historical reports.

## Categories

- `ACTIVE_CANONICAL`: source-of-truth docs for the current deterministic decision-diagnostic chain.
- `ACTIVE_REFERENCE`: retained reference docs, navigation, usage, audits, runbooks, or governance notes that are not the final schema authority.
- `DEPRECATED_LEGACY`: archived or superseded docs retained for historical context only.
- `REPORT_ARCHIVE`: report snapshots listed but not rewritten.
- `UNKNOWN_NEEDS_REVIEW`: file needs a future human classification pass.

## Inventory

| file path | category | reason | current source of truth if deprecated | action taken |
| --- | --- | --- | --- | --- |
| docs/architecture/architecture_map.md | ACTIVE_REFERENCE | Architecture reference retained for codebase orientation. | docs/SYSTEM_OVERVIEW.md; docs/DECISION_DIAGNOSTIC_CHAIN.md | Kept in place. |
| docs/architecture/CODEBASE_STRUCTURE_REF.md | ACTIVE_REFERENCE | Architecture reference retained for codebase orientation. | docs/SYSTEM_OVERVIEW.md; docs/DECISION_DIAGNOSTIC_CHAIN.md | Kept in place. |
| docs/archive/cleanup/2026-04-15_Wednesday__CHANGELOG_SUMMARY.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/cleanup/2026-04-20_Monday__CLEANUP_CHANGELOG.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/cleanup/2026-04-20_Monday__CODEBASE_CLEANUP_REPORT.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/cleanup/2026-04-20_Monday__TECH_SUMMARY_CLEANUP_AND_REFACTOR.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/legacy_docs_2026_05_05/DEPRECATION_MAP.md | ACTIVE_REFERENCE | Index/deprecation map for the 2026-05-05 legacy-doc migration. | docs/README.md; docs/DOCS_INVENTORY.md | Created. |
| docs/archive/legacy_docs_2026_05_05/INDEX.md | ACTIVE_REFERENCE | Index/deprecation map for the 2026-05-05 legacy-doc migration. | docs/README.md; docs/DOCS_INVENTORY.md | Created. |
| docs/archive/legacy_docs_2026_05_05/QUANT_CORE_SURFACE_AUDIT.md | DEPRECATED_LEGACY | Historical Quant Core surface audit predates the implemented allocator/router diagnostic chain and contains obsolete phase scope language. | docs/SYSTEM_OVERVIEW.md; docs/DECISION_DIAGNOSTIC_CHAIN.md; docs/governance/PIPELINE_CONTRACTS.md; docs/governance/QUANT_CORE_OUTPUT_SCHEMA.md | Moved from docs/governance/QUANT_CORE_SURFACE_AUDIT.md to legacy archive. |
| docs/archive/phases/2026-03-27_Friday__review_phase5_vs_phase45_2026-03-27.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/phases/2026-04-18_Saturday__PHASE1_REUSE_MAP.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/phases/2026-04-18_Saturday__PHASE2_REUSE_MAP.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/phases/2026-04-18_Saturday__PHASE25_ENTRYPOINT_AUDIT.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/phases/2026-04-19_Sunday__PHASE26_CALIBRATION_DECISION.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/phases/2026-04-19_Sunday__PHASE26_DECISION_LAYER_AUDIT.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/phases/2026-04-19_Sunday__PHASEF1_FORECAST_AUDIT.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/phases/2026-04-19_Sunday__PHASEF1_FORECAST_REHAB_DECISION.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/phases/2026-04-19_Sunday__PHASEF15_NARROW_REHAB_DECISION.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/phases/2026-04-19_Sunday__PHASEF15_NARROW_SCOPE.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/README.md | ACTIVE_REFERENCE | Archive navigation and policy index. | docs/README.md | Updated. |
| docs/archive/retrieval/2026-04-20_Monday__RETRIEVAL_ADAPTER_LAYER.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/retrieval/2026-04-20_Monday__RETRIEVAL_BACKEND_STRATEGY.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/retrieval/2026-04-20_Monday__RETRIEVAL_INDEXING_STRATEGY.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/retrieval/2026-04-20_Monday__RETRIEVAL_PREP_LAYER.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/root/2026-03-29_Sunday__competitive_analysis_vs_tradingagents_2026-03-28.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/root/2026-03-29_Sunday__tech_radar_systematic_trading.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/root/2026-03-29_Sunday__upgrade_execution_plan_local_ollama_2026-03-28.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/root/2026-04-02_Thursday__2026_04_02_0315_daily_brief_module_implementation.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/root/2026-04-02_Thursday__codebase_audit.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/root/2026-04-02_Thursday__daily_brief.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/root/2026-04-02_Thursday__walkthrough-skeleton.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/root/2026-04-15_Wednesday__RESEARCH_FINDINGS_AND_LIMITATIONS.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/root/2026-04-20_Monday__ANALYSIS_FEED_LAYER.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/root/2026-04-20_Monday__ANALYSIS_FEED_SCHEMA.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/root/AUDIT_REPORT.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/root/BASELINE.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/root/IMPROVEMENT_ROADMAP.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/root/REFACTOR_LOG.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/root/STRUCTURE_FINAL.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/root/walkthrough.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/archive/vn100/README.md | DEPRECATED_LEGACY | Existing archived historical documentation retained for traceability. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/audits/DECISION_CHAIN_AUDIT_2026_05_05.md | ACTIVE_CANONICAL | Required source-of-truth document for the current deterministic decision-diagnostic chain. | Self | Kept active. |
| docs/audits/VSEF_10Y_WALKFORWARD_AUDIT.md | ACTIVE_REFERENCE | Audit evidence or historical validation note retained as reference, not schema authority. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/audits/VSEF_15Y_BROADER_TICKER_MULTIHORIZON_AUDIT.md | ACTIVE_REFERENCE | Audit evidence or historical validation note retained as reference, not schema authority. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/audits/VSEF_15Y_DAILY_WALKFORWARD_AUDIT.md | ACTIVE_REFERENCE | Audit evidence or historical validation note retained as reference, not schema authority. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/audits/VSEF_15Y_DAILY_WALKFORWARD_NO_FOREIGN_FLOW_AUDIT.md | ACTIVE_REFERENCE | Audit evidence or historical validation note retained as reference, not schema authority. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/audits/VSEF_15Y_MULTIHORIZON_WALKFORWARD_AUDIT.md | ACTIVE_REFERENCE | Audit evidence or historical validation note retained as reference, not schema authority. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/audits/VSEF_BROADER_FEATURE_GOVERNANCE_AUDIT.md | ACTIVE_REFERENCE | Audit evidence or historical validation note retained as reference, not schema authority. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/audits/VSEF_FOREIGN_FLOW_COVERAGE_INVESTIGATION.md | ACTIVE_REFERENCE | Audit evidence or historical validation note retained as reference, not schema authority. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/audits/VSEF_FOREIGN_FLOW_CURATED_SAMPLE.md | ACTIVE_REFERENCE | Audit evidence or historical validation note retained as reference, not schema authority. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/audits/VSEF_FOREIGN_FLOW_PROVIDER_CURATION_ATTEMPT.md | ACTIVE_REFERENCE | Audit evidence or historical validation note retained as reference, not schema authority. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/audits/VSEF_FOREIGN_FLOW_REAL_COVERAGE_AUDIT.md | ACTIVE_REFERENCE | Audit evidence or historical validation note retained as reference, not schema authority. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/audits/VSEF_HELDOUT_THRESHOLD_SELECTION.md | ACTIVE_REFERENCE | Audit evidence or historical validation note retained as reference, not schema authority. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/audits/VSEF_MODEL_GAP_AUDIT.md | ACTIVE_REFERENCE | Audit evidence or historical validation note retained as reference, not schema authority. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/audits/VSEF_OHLCV_CACHE_10Y_AVAILABILITY_AUDIT.md | ACTIVE_REFERENCE | Audit evidence or historical validation note retained as reference, not schema authority. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/audits/VSEF_QUANT_CORE_REPEATED_SEED_STABILITY_AUDIT.md | ACTIVE_REFERENCE | Audit evidence or historical validation note retained as reference, not schema authority. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/audits/VSEF_QUANT_CORE_SMOKE_AUDIT.md | ACTIVE_REFERENCE | Audit evidence or historical validation note retained as reference, not schema authority. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/audits/VSEF_REAL_FEATURE_GOVERNANCE_AUDIT.md | ACTIVE_REFERENCE | Audit evidence or historical validation note retained as reference, not schema authority. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/audits/VSEF_ROLLING_REGIME_THRESHOLD_SELECTION.md | ACTIVE_REFERENCE | Audit evidence or historical validation note retained as reference, not schema authority. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/audits/VSEF_SIGNAL_EFFECTIVENESS_BACKTEST.md | ACTIVE_REFERENCE | Audit evidence or historical validation note retained as reference, not schema authority. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/audits/VSEF_SIGNAL_REGIME_JOIN.md | ACTIVE_REFERENCE | Audit evidence or historical validation note retained as reference, not schema authority. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/audits/VSEF_TEST_HARDENING_NOTES.md | ACTIVE_REFERENCE | Audit evidence or historical validation note retained as reference, not schema authority. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/AUTHORITY_BOUNDARY.md | ACTIVE_CANONICAL | Required source-of-truth document for the current deterministic decision-diagnostic chain. | Self | Kept active. |
| docs/DECISION_DIAGNOSTIC_CHAIN.md | ACTIVE_CANONICAL | Required source-of-truth document for the current deterministic decision-diagnostic chain. | Self | Kept active. |
| docs/DOCS_INVENTORY.md | ACTIVE_CANONICAL | Repository documentation inventory created by this cleanup. | Self | Created. |
| docs/EVALUATION_WORKFLOWS.md | ACTIVE_REFERENCE | Root-level active documentation retained for navigation or workflow reference. | docs/README.md | Kept in place. |
| docs/governance/data_quality.md | ACTIVE_REFERENCE | Governance reference retained; canonical chain schemas override if conflicts appear. | docs/governance/PIPELINE_CONTRACTS.md and relevant active schema docs. | Kept in place. |
| docs/governance/DECISION_LANE_OUTPUT_SCHEMA.md | ACTIVE_CANONICAL | Required source-of-truth document for the current deterministic decision-diagnostic chain. | Self | Kept active. |
| docs/governance/experiment_tracking.md | ACTIVE_REFERENCE | Governance reference retained; canonical chain schemas override if conflicts appear. | docs/governance/PIPELINE_CONTRACTS.md and relevant active schema docs. | Kept in place. |
| docs/governance/PHASE3_ROUTER_OUTPUT_SCHEMA.md | ACTIVE_CANONICAL | Required source-of-truth document for the current deterministic decision-diagnostic chain. | Self | Kept active. |
| docs/governance/PHASE3_ROUTER_V1.md | ACTIVE_REFERENCE | Governance reference retained; canonical chain schemas override if conflicts appear. | docs/governance/PIPELINE_CONTRACTS.md and relevant active schema docs. | Kept in place. |
| docs/governance/PIPELINE_CONTRACTS.md | ACTIVE_CANONICAL | Required source-of-truth document for the current deterministic decision-diagnostic chain. | Self | Kept active. |
| docs/governance/PORTFOLIO_ALLOCATOR_OUTPUT_SCHEMA.md | ACTIVE_CANONICAL | Required source-of-truth document for the current deterministic decision-diagnostic chain. | Self | Kept active. |
| docs/governance/PORTFOLIO_ALLOCATOR_V1.md | ACTIVE_REFERENCE | Governance reference retained; canonical chain schemas override if conflicts appear. | docs/governance/PIPELINE_CONTRACTS.md and relevant active schema docs. | Kept in place. |
| docs/governance/QUANT_CORE_GOVERNANCE.md | ACTIVE_CANONICAL | Required source-of-truth document for the current deterministic decision-diagnostic chain. | Self | Kept active. |
| docs/governance/QUANT_CORE_OUTPUT_SCHEMA.md | ACTIVE_CANONICAL | Required source-of-truth document for the current deterministic decision-diagnostic chain. | Self | Kept active. |
| docs/governance/RISK_GOVERNANCE_OUTPUT_SCHEMA.md | ACTIVE_CANONICAL | Required source-of-truth document for the current deterministic decision-diagnostic chain. | Self | Kept active. |
| docs/governance/SCENARIO_EVALUATION_ENGINE.md | ACTIVE_CANONICAL | Required source-of-truth document for the current deterministic decision-diagnostic chain. | Self | Kept active. |
| docs/governance/SCENARIO_OUTPUT_SCHEMA.md | ACTIVE_CANONICAL | Required source-of-truth document for the current deterministic decision-diagnostic chain. | Self | Kept active. |
| docs/governance/VSEF_CONTEXT_AVAILABILITY_METADATA.md | ACTIVE_REFERENCE | Governance reference retained; canonical chain schemas override if conflicts appear. | docs/governance/PIPELINE_CONTRACTS.md and relevant active schema docs. | Kept in place. |
| docs/governance/VSEF_CONTEXT_COVERAGE_DIAGNOSTICS.md | ACTIVE_REFERENCE | Governance reference retained; canonical chain schemas override if conflicts appear. | docs/governance/PIPELINE_CONTRACTS.md and relevant active schema docs. | Kept in place. |
| docs/governance/VSEF_CONTEXT_TIMING_GOVERNANCE.md | ACTIVE_REFERENCE | Governance reference retained; canonical chain schemas override if conflicts appear. | docs/governance/PIPELINE_CONTRACTS.md and relevant active schema docs. | Kept in place. |
| docs/governance/VSEF_FEATURE_GOVERNANCE_REVIEW.md | ACTIVE_REFERENCE | Governance reference retained; canonical chain schemas override if conflicts appear. | docs/governance/PIPELINE_CONTRACTS.md and relevant active schema docs. | Kept in place. |
| docs/governance/VSEF_FEATURE_IMPORTANCE_DIAGNOSTICS.md | ACTIVE_REFERENCE | Governance reference retained; canonical chain schemas override if conflicts appear. | docs/governance/PIPELINE_CONTRACTS.md and relevant active schema docs. | Kept in place. |
| docs/governance/VSEF_FOREIGN_FLOW_ARTIFACT_POLICY.md | ACTIVE_REFERENCE | Governance reference retained; canonical chain schemas override if conflicts appear. | docs/governance/PIPELINE_CONTRACTS.md and relevant active schema docs. | Kept in place. |
| docs/governance/VSEF_FOREIGN_FLOW_DISABLE_MODE.md | ACTIVE_REFERENCE | Governance reference retained; canonical chain schemas override if conflicts appear. | docs/governance/PIPELINE_CONTRACTS.md and relevant active schema docs. | Kept in place. |
| docs/governance/VSEF_LINEAR_FOLD_DIAGNOSTICS.md | ACTIVE_REFERENCE | Governance reference retained; canonical chain schemas override if conflicts appear. | docs/governance/PIPELINE_CONTRACTS.md and relevant active schema docs. | Kept in place. |
| docs/prompt_runs/archive/analysis_feed/11_ranked_predictions.md | DEPRECATED_LEGACY | Historical prompt-run archive, not active implementation governance. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/prompt_runs/archive/analysis_feed/12_daily_brief.md | DEPRECATED_LEGACY | Historical prompt-run archive, not active implementation governance. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/prompt_runs/archive/phase1/001_tracking_and_quality.md | DEPRECATED_LEGACY | Historical prompt-run archive, not active implementation governance. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/prompt_runs/archive/phase1/01_vnstock_integration_audit.md | DEPRECATED_LEGACY | Historical prompt-run archive, not active implementation governance. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/prompt_runs/archive/phase1/02_adapter_layer.md | DEPRECATED_LEGACY | Historical prompt-run archive, not active implementation governance. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/prompt_runs/archive/phase1/03_universe_loader.md | DEPRECATED_LEGACY | Historical prompt-run archive, not active implementation governance. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/prompt_runs/archive/phase1/04_daily_sync.md | DEPRECATED_LEGACY | Historical prompt-run archive, not active implementation governance. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/prompt_runs/archive/phase1/14_data_quality.md | DEPRECATED_LEGACY | Historical prompt-run archive, not active implementation governance. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/prompt_runs/archive/phase2/05_data_loader_extension.md | DEPRECATED_LEGACY | Historical prompt-run archive, not active implementation governance. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/prompt_runs/archive/phase2/06_feature_engineering_extension.md | DEPRECATED_LEGACY | Historical prompt-run archive, not active implementation governance. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/prompt_runs/archive/phase2/07_label_engineering.md | DEPRECATED_LEGACY | Historical prompt-run archive, not active implementation governance. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/prompt_runs/archive/phase2/08_training_pipeline_integration.md | DEPRECATED_LEGACY | Historical prompt-run archive, not active implementation governance. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/prompt_runs/archive/phase2/09_batch_vn100_training.md | DEPRECATED_LEGACY | Historical prompt-run archive, not active implementation governance. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/prompt_runs/archive/phase2/10_batch_inference.md | DEPRECATED_LEGACY | Historical prompt-run archive, not active implementation governance. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/prompt_runs/archive/phase2/13_experiment_tracking.md | DEPRECATED_LEGACY | Historical prompt-run archive, not active implementation governance. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/prompt_runs/archive/phase2/15_experiment_tracking_validation.md | DEPRECATED_LEGACY | Historical prompt-run archive, not active implementation governance. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/prompt_runs/archive/phase2/16_feature_parity_unification.md | DEPRECATED_LEGACY | Historical prompt-run archive, not active implementation governance. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/prompt_runs/archive/phase2/17_batch2_patch_cleanup.md | DEPRECATED_LEGACY | Historical prompt-run archive, not active implementation governance. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/prompt_runs/archive/phase2/19_ml_engine_stability_hardening.md | DEPRECATED_LEGACY | Historical prompt-run archive, not active implementation governance. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/prompt_runs/archive/quant_core/18_quantitative_architecture_expansion.md | DEPRECATED_LEGACY | Historical prompt-run archive, not active implementation governance. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/prompt_runs/archive/quant_core/20_walk_forward_experiment_hard_requirements.md | DEPRECATED_LEGACY | Historical prompt-run archive, not active implementation governance. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/prompt_runs/archive/quant_core/21_quant_core_operational_validation.md | DEPRECATED_LEGACY | Historical prompt-run archive, not active implementation governance. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/prompt_runs/archive/README.md | DEPRECATED_LEGACY | Historical prompt-run archive, not active implementation governance. | docs/README.md and active canonical docs for current behavior. | Already archived; retained. |
| docs/prompt_runs/CHANGELOG_SUMMARY.md | ACTIVE_REFERENCE | Prompt-run navigation or changelog retained as operational reference. | docs/README.md | Kept in place. |
| docs/prompt_runs/INDEX.md | ACTIVE_REFERENCE | Prompt-run navigation or changelog retained as operational reference. | docs/README.md | Kept in place. |
| docs/README.md | ACTIVE_CANONICAL | Navigation index for active docs, schemas, runbooks, roadmap, audits, and archive. | Self | Updated as active navigation index. |
| docs/reports/VSEF_15Y_DAILY_MULTIHORIZON_TECHNICAL_REPORT.md | REPORT_ARCHIVE | Docs report snapshot retained for historical interpretation. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/reports/VSEF_EMPIRICAL_FINDINGS_AND_LIMITATIONS_SUMMARY.md | REPORT_ARCHIVE | Docs report snapshot retained for historical interpretation. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/REPOSITORY_STRUCTURE.md | ACTIVE_REFERENCE | Root-level active documentation retained for navigation or workflow reference. | docs/README.md | Kept in place. |
| docs/roadmap/DECISION_PHYSICS_NEXT_STEPS.md | ACTIVE_CANONICAL | Required source-of-truth document for the current deterministic decision-diagnostic chain. | Self | Kept active. |
| docs/roadmap/SOCIAL_LISTENING_INTEGRATION.md | ACTIVE_CANONICAL | Required source-of-truth document for the current deterministic decision-diagnostic chain. | Self | Kept active. |
| docs/roadmap/VSEF_ROADMAP.md | ACTIVE_REFERENCE | Roadmap or future-work reference; canonical docs govern current behavior. | docs/README.md and docs/AUTHORITY_BOUNDARY.md | Kept in place. |
| docs/runbooks/RUN_FULL_DECISION_CHAIN_SMOKE.md | ACTIVE_CANONICAL | Required source-of-truth document for the current deterministic decision-diagnostic chain. | Self | Kept active. |
| docs/runbooks/TROUBLESHOOTING.md | ACTIVE_CANONICAL | Required source-of-truth document for the current deterministic decision-diagnostic chain. | Self | Kept active. |
| docs/SYSTEM_OVERVIEW.md | ACTIVE_CANONICAL | Required source-of-truth document for the current deterministic decision-diagnostic chain. | Self | Kept active. |
| docs/usage/batch_vn100_inference.md | ACTIVE_REFERENCE | Usage guide retained for local workflows; governance schemas override current chain contracts. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/usage/daily_sync_vn100.md | ACTIVE_REFERENCE | Usage guide retained for local workflows; governance schemas override current chain contracts. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/usage/data_loader_vn100.md | ACTIVE_REFERENCE | Usage guide retained for local workflows; governance schemas override current chain contracts. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/usage/feature_engineering_vn100.md | ACTIVE_REFERENCE | Usage guide retained for local workflows; governance schemas override current chain contracts. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/usage/label_engineering.md | ACTIVE_REFERENCE | Usage guide retained for local workflows; governance schemas override current chain contracts. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/usage/ML_IMPLEMENTATION_GUIDE.md | ACTIVE_REFERENCE | Usage guide retained for local workflows; governance schemas override current chain contracts. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/usage/ranked_predictions.md | ACTIVE_REFERENCE | Usage guide retained for local workflows; governance schemas override current chain contracts. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/usage/train_ml_vn100.md | ACTIVE_REFERENCE | Usage guide retained for local workflows; governance schemas override current chain contracts. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/usage/train_pipeline_labels.md | ACTIVE_REFERENCE | Usage guide retained for local workflows; governance schemas override current chain contracts. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/usage/universe_loader.md | ACTIVE_REFERENCE | Usage guide retained for local workflows; governance schemas override current chain contracts. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/usage/USAGE_GUIDE.md | ACTIVE_REFERENCE | Usage guide retained for local workflows; governance schemas override current chain contracts. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/usage/vnstock_adapter.md | ACTIVE_REFERENCE | Usage guide retained for local workflows; governance schemas override current chain contracts. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| docs/usage/VSEF_FOREIGN_FLOW_AUDIT_COMMANDS.md | ACTIVE_REFERENCE | Usage guide retained for local workflows; governance schemas override current chain contracts. | docs/README.md and active governance schemas under docs/governance/ | Kept in place. |
| reports/baseline_local_qwen_4b_report.md | REPORT_ARCHIVE | Top-level report snapshot listed for inventory only; content not rewritten. | docs/README.md and active governance schemas under docs/governance/ | Listed only; not modified. |
| reports/comparison_qwen_4b_vs_8b.md | REPORT_ARCHIVE | Top-level report snapshot listed for inventory only; content not rewritten. | docs/README.md and active governance schemas under docs/governance/ | Listed only; not modified. |
| reports/full_system_report.md | REPORT_ARCHIVE | Top-level report snapshot listed for inventory only; content not rewritten. | docs/README.md and active governance schemas under docs/governance/ | Listed only; not modified. |
| reports/README.md | ACTIVE_REFERENCE | Reports directory policy note; reports are snapshots, not canonical governance. | docs/SYSTEM_OVERVIEW.md; docs/DECISION_DIAGNOSTIC_CHAIN.md; docs/AUTHORITY_BOUNDARY.md | Created. |
| reports/risk_tuning_report.md | REPORT_ARCHIVE | Top-level report snapshot listed for inventory only; content not rewritten. | docs/README.md and active governance schemas under docs/governance/ | Listed only; not modified. |
| reports/stress_test_report.md | REPORT_ARCHIVE | Top-level report snapshot listed for inventory only; content not rewritten. | docs/README.md and active governance schemas under docs/governance/ | Listed only; not modified. |
| reports/system_benchmark.md | REPORT_ARCHIVE | Top-level report snapshot listed for inventory only; content not rewritten. | docs/README.md and active governance schemas under docs/governance/ | Listed only; not modified. |
| reports/VSEF_1000_SEED_SMOKE_EXECUTIVE_SUMMARY.md | REPORT_ARCHIVE | Top-level report snapshot listed for inventory only; content not rewritten. | docs/README.md and active governance schemas under docs/governance/ | Listed only; not modified. |
| reports/VSEF_1000_SEED_SMOKE_STABILITY_REPORT.md | REPORT_ARCHIVE | Top-level report snapshot listed for inventory only; content not rewritten. | docs/README.md and active governance schemas under docs/governance/ | Listed only; not modified. |
| reports/VSEF_PHASE3_ALLOCATOR_ROUTER_REAL_OUTPUT_SMOKE_REPORT.md | REPORT_ARCHIVE | Top-level report snapshot listed for inventory only; content not rewritten. | docs/README.md and active governance schemas under docs/governance/ | Listed only; not modified. |
| reports/VSEF_QUANT_CORE_PHASE3_GAP_CLOSURE_PLAN.md | REPORT_ARCHIVE | Top-level report snapshot listed for inventory only; content not rewritten. | docs/README.md and active governance schemas under docs/governance/ | Listed only; not modified. |
