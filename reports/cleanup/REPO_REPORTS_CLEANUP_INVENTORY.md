# Reports Root Cleanup Inventory

- Branch inventoried: `research/vn100-evidence-hardening-v1`.
- Scope: root-level `reports/*.md` files only.
- Excluded: `reports/generated/`, existing report subfolders, non-Markdown artifacts, raw data, outputs, and archive snapshots.
- Move policy: tracked root Markdown reports move with `git mv`; generated evidence folders do not move.

## Summary

- `active_claim`: 18
- `active_index`: 6
- `active_protocol`: 31
- `active_result`: 25
- `cleanup_governance`: 31
- `historical_or_superseded`: 18
- `manual_review`: 2
- `paper_support`: 8

## Inventory

| current path | proposed new path | category | active | reason | references found | needs reference update | move |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `reports/ACTIVE_CODE_MAP.md` | `reports/_index/ACTIVE_CODE_MAP.md` | active_index | yes | Central navigation or active status index. | yes | yes | yes |
| `reports/ACTIVE_EVIDENCE_INDEX.md` | `reports/_index/ACTIVE_EVIDENCE_INDEX.md` | active_index | yes | Central navigation or active status index. | yes | yes | yes |
| `reports/AUDIT_REMEDIATION_CHECKLIST.md` | `reports/cleanup/AUDIT_REMEDIATION_CHECKLIST.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/baseline_local_qwen_4b_report.md` | `reports/superseded/baseline_local_qwen_4b_report.md` | historical_or_superseded | no | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes | yes | yes |
| `reports/CODE_AUDIT_REMEDIATION_CLOSEOUT.md` | `reports/cleanup/CODE_AUDIT_REMEDIATION_CLOSEOUT.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/CODE_AUDIT_REMEDIATION_PLAN.md` | `reports/cleanup/CODE_AUDIT_REMEDIATION_PLAN.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/CODE_AUDIT_REPORT.md` | `reports/cleanup/CODE_AUDIT_REPORT.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/CODE_CLEANUP_CHANGES.md` | `reports/cleanup/CODE_CLEANUP_CHANGES.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/CODE_CLEANUP_SCOPE.md` | `reports/cleanup/CODE_CLEANUP_SCOPE.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/CODE_CLEANUP_VALIDATION_FIX_PLAN.md` | `reports/cleanup/CODE_CLEANUP_VALIDATION_FIX_PLAN.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/CODE_CLEANUP_VALIDATION_FIX_RESULT.md` | `reports/cleanup/CODE_CLEANUP_VALIDATION_FIX_RESULT.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/comparison_qwen_4b_vs_8b.md` | `reports/superseded/comparison_qwen_4b_vs_8b.md` | historical_or_superseded | no | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes | yes | yes |
| `reports/DOCS_AND_TASK_RUNNER_CLEANUP_RESULT.md` | `reports/cleanup/DOCS_AND_TASK_RUNNER_CLEANUP_RESULT.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/FULL_DATA_PUSH_INVENTORY.md` | `reports/cleanup/FULL_DATA_PUSH_INVENTORY.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/FULL_DATA_PUSH_RESULT.md` | `reports/cleanup/FULL_DATA_PUSH_RESULT.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/FULL_DATA_PUSH_STAGING_REVIEW.md` | `reports/cleanup/FULL_DATA_PUSH_STAGING_REVIEW.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/full_system_report.md` | `reports/superseded/full_system_report.md` | historical_or_superseded | no | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes | yes | yes |
| `reports/GENERATED_REPORTS_STATUS.md` | `reports/_index/GENERATED_REPORTS_STATUS.md` | active_index | yes | Central navigation or active status index. | yes | yes | yes |
| `reports/INDEX_HOURLY_FETCH_README.md` | `reports/protocols/INDEX_HOURLY_FETCH_README.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/MANUAL_REVIEW_SCRIPT_RESOLUTION.md` | `reports/manual_review/MANUAL_REVIEW_SCRIPT_RESOLUTION.md` | manual_review | no | Manual review or unresolved cleanup note. | yes | yes | yes |
| `reports/NCKH_EVIDENCE_GAP_CLOSURE_REGISTER.md` | `reports/superseded/NCKH_EVIDENCE_GAP_CLOSURE_REGISTER.md` | historical_or_superseded | no | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes | yes | yes |
| `reports/NCKH_EVIDENCE_UPGRADE_ADDENDUM_V2.md` | `reports/superseded/NCKH_EVIDENCE_UPGRADE_ADDENDUM_V2.md` | historical_or_superseded | no | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes | yes | yes |
| `reports/NCKH_EXPERIMENT_INVENTORY.md` | `reports/superseded/NCKH_EXPERIMENT_INVENTORY.md` | historical_or_superseded | no | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes | yes | yes |
| `reports/NCKH_RESEARCH_DESIGN_VN100.md` | `reports/superseded/NCKH_RESEARCH_DESIGN_VN100.md` | historical_or_superseded | no | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes | yes | yes |
| `reports/NCKH_RESULTS_CLAIM_REGISTER.md` | `reports/superseded/NCKH_RESULTS_CLAIM_REGISTER.md` | historical_or_superseded | no | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes | yes | yes |
| `reports/NCKH_RESULTS_CLAIM_REGISTER_V2.md` | `reports/superseded/NCKH_RESULTS_CLAIM_REGISTER_V2.md` | historical_or_superseded | no | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes | yes | yes |
| `reports/NEW_REMOTE_PUSH_VERIFICATION.md` | `reports/cleanup/NEW_REMOTE_PUSH_VERIFICATION.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/PHASE2_MOCK_PROVENANCE_EVIDENCE.md` | `reports/cleanup/PHASE2_MOCK_PROVENANCE_EVIDENCE.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/PHASE3_REPOSITORY_HYGIENE_EVIDENCE.md` | `reports/cleanup/PHASE3_REPOSITORY_HYGIENE_EVIDENCE.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/PHASE4_API_GOVERNANCE_EVIDENCE.md` | `reports/cleanup/PHASE4_API_GOVERNANCE_EVIDENCE.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/PHASE4A_FRONTEND_DESCOPE_EVIDENCE.md` | `reports/cleanup/PHASE4A_FRONTEND_DESCOPE_EVIDENCE.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/PHASE5_REPRODUCIBILITY_COMMAND_REGISTRY_EVIDENCE.md` | `reports/cleanup/PHASE5_REPRODUCIBILITY_COMMAND_REGISTRY_EVIDENCE.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/PHASE6_STATISTICAL_ACCEPTANCE_EVIDENCE.md` | `reports/cleanup/PHASE6_STATISTICAL_ACCEPTANCE_EVIDENCE.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/PHASE7_RUNTIME_PATH_REGISTRY_EVIDENCE.md` | `reports/cleanup/PHASE7_RUNTIME_PATH_REGISTRY_EVIDENCE.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/PHASE8_GOVERNED_INTEGRATION_TESTING_EVIDENCE.md` | `reports/cleanup/PHASE8_GOVERNED_INTEGRATION_TESTING_EVIDENCE.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/PROVIDER_STANDARDIZATION_AUDIT.md` | `reports/cleanup/PROVIDER_STANDARDIZATION_AUDIT.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/README.md` | `reports/README.md` | active_index | yes | Root reports pointer file remains at reports/README.md. | yes | no | no |
| `reports/README_REWRITE_AND_IDENTITY_CLEANUP.md` | `reports/cleanup/README_REWRITE_AND_IDENTITY_CLEANUP.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/REPO_CLEANUP_INVENTORY.md` | `reports/_index/REPO_CLEANUP_INVENTORY.md` | active_index | yes | Central navigation or active status index. | yes | yes | yes |
| `reports/REPO_CLEANUP_MANUAL_REVIEW.md` | `reports/manual_review/REPO_CLEANUP_MANUAL_REVIEW.md` | manual_review | no | Manual review or unresolved cleanup note. | yes | yes | yes |
| `reports/REPO_RENAME_CLEANUP_INVENTORY.md` | `reports/cleanup/REPO_RENAME_CLEANUP_INVENTORY.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/REPO_RENAME_CLEANUP_RESULT.md` | `reports/cleanup/REPO_RENAME_CLEANUP_RESULT.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/REPO_REPORTS_CLEANUP_INVENTORY.md` | `reports/cleanup/REPO_REPORTS_CLEANUP_INVENTORY.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | no | no | yes |
| `reports/RESEARCH_SCRIPT_STATUS.md` | `reports/_index/RESEARCH_SCRIPT_STATUS.md` | active_index | yes | Central navigation or active status index. | yes | yes | yes |
| `reports/RESEARCH_TOPIC_AND_CODE_AUDIT_REPORT.md` | `reports/results/RESEARCH_TOPIC_AND_CODE_AUDIT_REPORT.md` | active_result | yes | Current result, audit, diagnosis, inventory, or analysis file. | yes | yes | yes |
| `reports/risk_tuning_report.md` | `reports/superseded/risk_tuning_report.md` | historical_or_superseded | no | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes | yes | yes |
| `reports/SCRIPT_REORGANIZATION_PLAN.md` | `reports/cleanup/SCRIPT_REORGANIZATION_PLAN.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/SCRIPT_REORGANIZATION_RESULT.md` | `reports/cleanup/SCRIPT_REORGANIZATION_RESULT.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/stress_test_report.md` | `reports/superseded/stress_test_report.md` | historical_or_superseded | no | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes | yes | yes |
| `reports/system_benchmark.md` | `reports/superseded/system_benchmark.md` | historical_or_superseded | no | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes | yes | yes |
| `reports/VN100_HYBRID_BENCHMARK_CLOSEOUT.md` | `reports/superseded/VN100_HYBRID_BENCHMARK_CLOSEOUT.md` | historical_or_superseded | no | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes | yes | yes |
| `reports/VN30_DAILY_2015_BCM_VIB_RECOVERY_REPORT.md` | `reports/results/VN30_DAILY_2015_BCM_VIB_RECOVERY_REPORT.md` | active_result | yes | Current result, audit, diagnosis, inventory, or analysis file. | yes | yes | yes |
| `reports/VN30_DAILY_2015_CLAIM_REGISTER.md` | `reports/claims/VN30_DAILY_2015_CLAIM_REGISTER.md` | active_claim | yes | Claim register, claim note, or claim-boundary file. | yes | yes | yes |
| `reports/VN30_DAILY_2015_RESULT_SUMMARY.md` | `reports/results/VN30_DAILY_2015_RESULT_SUMMARY.md` | active_result | yes | Current result, audit, diagnosis, inventory, or analysis file. | yes | yes | yes |
| `reports/VN30_DAILY_2015_SCOPE_AND_DESIGN.md` | `reports/protocols/VN30_DAILY_2015_SCOPE_AND_DESIGN.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_DAILY_2015_TARGET60_CLAIM_REGISTER.md` | `reports/claims/VN30_DAILY_2015_TARGET60_CLAIM_REGISTER.md` | active_claim | yes | Claim register, claim note, or claim-boundary file. | yes | yes | yes |
| `reports/VN30_DAILY_2015_TARGET60_FAILURE_POSTMORTEM_PROTOCOL.md` | `reports/protocols/VN30_DAILY_2015_TARGET60_FAILURE_POSTMORTEM_PROTOCOL.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_DAILY_2015_TARGET60_NEXT_STEP_DECISION.md` | `reports/protocols/VN30_DAILY_2015_TARGET60_NEXT_STEP_DECISION.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_DAILY_2015_TARGET60_PROTOCOL.md` | `reports/protocols/VN30_DAILY_2015_TARGET60_PROTOCOL.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_DAILY_2015_TARGET60_RESULT_SUMMARY.md` | `reports/results/VN30_DAILY_2015_TARGET60_RESULT_SUMMARY.md` | active_result | yes | Current result, audit, diagnosis, inventory, or analysis file. | yes | yes | yes |
| `reports/VN30_DAILY_2015_TARGET60_V2_CLAIM_REGISTER.md` | `reports/claims/VN30_DAILY_2015_TARGET60_V2_CLAIM_REGISTER.md` | active_claim | yes | Claim register, claim note, or claim-boundary file. | yes | yes | yes |
| `reports/VN30_DAILY_2015_TARGET60_V2_PROTOCOL.md` | `reports/protocols/VN30_DAILY_2015_TARGET60_V2_PROTOCOL.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_DAILY_2015_TARGET60_V2_RESULT_SUMMARY.md` | `reports/results/VN30_DAILY_2015_TARGET60_V2_RESULT_SUMMARY.md` | active_result | yes | Current result, audit, diagnosis, inventory, or analysis file. | yes | yes | yes |
| `reports/VN30_HOURLY_2015_BASELINE60_RESULT_LOCK.md` | `reports/results/VN30_HOURLY_2015_BASELINE60_RESULT_LOCK.md` | active_result | yes | Current result, audit, diagnosis, inventory, or analysis file. | yes | yes | yes |
| `reports/VN30_HOURLY_2015_BENCHMARK_CLAIM_REGISTER.md` | `reports/claims/VN30_HOURLY_2015_BENCHMARK_CLAIM_REGISTER.md` | active_claim | yes | Claim register, claim note, or claim-boundary file. | yes | yes | yes |
| `reports/VN30_HOURLY_2015_CANONICAL_EVALUATOR_DECISION.md` | `reports/protocols/VN30_HOURLY_2015_CANONICAL_EVALUATOR_DECISION.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_HOURLY_2015_DATA_READINESS_PLAN.md` | `reports/protocols/VN30_HOURLY_2015_DATA_READINESS_PLAN.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_HOURLY_2015_JAN2025_BENCHMARK_RESULT_SUMMARY.md` | `reports/results/VN30_HOURLY_2015_JAN2025_BENCHMARK_RESULT_SUMMARY.md` | active_result | yes | Current result, audit, diagnosis, inventory, or analysis file. | yes | yes | yes |
| `reports/VN30_HOURLY_2015_JAN2025_UNIVERSE_DECISION.md` | `reports/protocols/VN30_HOURLY_2015_JAN2025_UNIVERSE_DECISION.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_HOURLY_2015_POST_BENCHMARK_DIAGNOSTICS_SUMMARY.md` | `reports/results/VN30_HOURLY_2015_POST_BENCHMARK_DIAGNOSTICS_SUMMARY.md` | active_result | yes | Current result, audit, diagnosis, inventory, or analysis file. | yes | yes | yes |
| `reports/VN30_HOURLY_2015_TOPK_75_VERIFICATION_DECISION.md` | `reports/protocols/VN30_HOURLY_2015_TOPK_75_VERIFICATION_DECISION.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_HOURLY_2015_TOPK_75_VERIFICATION_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_2015_TOPK_75_VERIFICATION_PROTOCOL.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_HOURLY_2015_TOPK_RANKING_CLAIM_REGISTER.md` | `reports/claims/VN30_HOURLY_2015_TOPK_RANKING_CLAIM_REGISTER.md` | active_claim | yes | Claim register, claim note, or claim-boundary file. | yes | yes | yes |
| `reports/VN30_HOURLY_2015_TOPK_RANKING_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_2015_TOPK_RANKING_PROTOCOL.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_HOURLY_2015_TOPK_RANKING_RESULT_SUMMARY.md` | `reports/results/VN30_HOURLY_2015_TOPK_RANKING_RESULT_SUMMARY.md` | active_result | yes | Current result, audit, diagnosis, inventory, or analysis file. | yes | yes | yes |
| `reports/VN30_HOURLY_2015_VALIDATION_FINAL_MISMATCH_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_2015_VALIDATION_FINAL_MISMATCH_PROTOCOL.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_HOURLY_BASELINE60_CLAIM_NOTE.md` | `reports/claims/VN30_HOURLY_BASELINE60_CLAIM_NOTE.md` | active_claim | yes | Claim register, claim note, or claim-boundary file. | yes | yes | yes |
| `reports/VN30_HOURLY_DATA_FORENSICS_DIAGNOSIS.md` | `reports/results/VN30_HOURLY_DATA_FORENSICS_DIAGNOSIS.md` | active_result | yes | Current result, audit, diagnosis, inventory, or analysis file. | yes | yes | yes |
| `reports/VN30_HOURLY_DUAL_TRACK_MODEL_COMPARISON_CLAIM_REGISTER.md` | `reports/claims/VN30_HOURLY_DUAL_TRACK_MODEL_COMPARISON_CLAIM_REGISTER.md` | active_claim | yes | Claim register, claim note, or claim-boundary file. | yes | yes | yes |
| `reports/VN30_HOURLY_DUAL_TRACK_MODEL_COMPARISON_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_DUAL_TRACK_MODEL_COMPARISON_PROTOCOL.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_HOURLY_DUAL_TRACK_MODEL_COMPARISON_RESULT.md` | `reports/results/VN30_HOURLY_DUAL_TRACK_MODEL_COMPARISON_RESULT.md` | active_result | yes | Current result, audit, diagnosis, inventory, or analysis file. | yes | yes | yes |
| `reports/VN30_HOURLY_EXPANDED_MODEL_POOL_STACKING_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_EXPANDED_MODEL_POOL_STACKING_PROTOCOL.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_HOURLY_EXPANDED_MODEL_POOL_STACKING_V1_CLAIM_REGISTER.md` | `reports/claims/VN30_HOURLY_EXPANDED_MODEL_POOL_STACKING_V1_CLAIM_REGISTER.md` | active_claim | yes | Claim register, claim note, or claim-boundary file. | yes | yes | yes |
| `reports/VN30_HOURLY_EXPANDED_MODEL_POOL_STACKING_V1_RESULT.md` | `reports/results/VN30_HOURLY_EXPANDED_MODEL_POOL_STACKING_V1_RESULT.md` | active_result | yes | Current result, audit, diagnosis, inventory, or analysis file. | yes | yes | yes |
| `reports/VN30_HOURLY_MARKET_INDEX_CONTEXT_PLAN.md` | `reports/protocols/VN30_HOURLY_MARKET_INDEX_CONTEXT_PLAN.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_HOURLY_PAPER_DOCX_MISSING_FOR_FIGURE_INSERTION.md` | `reports/paper/VN30_HOURLY_PAPER_DOCX_MISSING_FOR_FIGURE_INSERTION.md` | paper_support | yes | VN30 hourly paper/source support artifact. | yes | yes | yes |
| `reports/VN30_HOURLY_PAPER_FIGURE_DATA_SOURCE_INVENTORY.md` | `reports/paper/VN30_HOURLY_PAPER_FIGURE_DATA_SOURCE_INVENTORY.md` | paper_support | yes | VN30 hourly paper/source support artifact. | yes | yes | yes |
| `reports/VN30_HOURLY_PAPER_LITERATURE_DATA_TODO.md` | `reports/paper/VN30_HOURLY_PAPER_LITERATURE_DATA_TODO.md` | paper_support | yes | VN30 hourly paper/source support artifact. | yes | yes | yes |
| `reports/VN30_HOURLY_PAPER_MISSING_METRICS_TODO.md` | `reports/paper/VN30_HOURLY_PAPER_MISSING_METRICS_TODO.md` | paper_support | yes | VN30 hourly paper/source support artifact. | yes | yes | yes |
| `reports/VN30_HOURLY_PAPER_ROW_LEVEL_FIGURE_TODO.md` | `reports/paper/VN30_HOURLY_PAPER_ROW_LEVEL_FIGURE_TODO.md` | paper_support | yes | VN30 hourly paper/source support artifact. | yes | yes | yes |
| `reports/VN30_HOURLY_PAPER_TABLE_FIGURE_CAPTIONS_EN.md` | `reports/paper/VN30_HOURLY_PAPER_TABLE_FIGURE_CAPTIONS_EN.md` | paper_support | yes | VN30 hourly paper/source support artifact. | yes | yes | yes |
| `reports/VN30_HOURLY_PAPER_TABLE_FIGURE_CAPTIONS_VI.md` | `reports/paper/VN30_HOURLY_PAPER_TABLE_FIGURE_CAPTIONS_VI.md` | paper_support | yes | VN30 hourly paper/source support artifact. | yes | yes | yes |
| `reports/VN30_HOURLY_PAPER_WITH_FIGURES_LAYOUT_QA.md` | `reports/paper/VN30_HOURLY_PAPER_WITH_FIGURES_LAYOUT_QA.md` | paper_support | yes | VN30 hourly paper/source support artifact. | yes | yes | yes |
| `reports/VN30_HOURLY_RF_H60_BASELINE_REPRODUCTION_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_RF_H60_BASELINE_REPRODUCTION_PROTOCOL.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_HOURLY_RF_H60_BASELINE_REPRODUCTION_RESULT.md` | `reports/results/VN30_HOURLY_RF_H60_BASELINE_REPRODUCTION_RESULT.md` | active_result | yes | Current result, audit, diagnosis, inventory, or analysis file. | yes | yes | yes |
| `reports/VN30_HOURLY_SELECTED_CANDIDATE_ARTIFACT_DISCOVERY.md` | `reports/results/VN30_HOURLY_SELECTED_CANDIDATE_ARTIFACT_DISCOVERY.md` | active_result | yes | Current result, audit, diagnosis, inventory, or analysis file. | yes | yes | yes |
| `reports/VN30_HOURLY_SELECTED_CANDIDATE_CLAIM_BOUNDARY.md` | `reports/claims/VN30_HOURLY_SELECTED_CANDIDATE_CLAIM_BOUNDARY.md` | active_claim | yes | Claim register, claim note, or claim-boundary file. | yes | yes | yes |
| `reports/VN30_HOURLY_SELECTED_CANDIDATE_ROLLING_STABILITY_RESULT.md` | `reports/results/VN30_HOURLY_SELECTED_CANDIDATE_ROLLING_STABILITY_RESULT.md` | active_result | yes | Current result, audit, diagnosis, inventory, or analysis file. | yes | yes | yes |
| `reports/VN30_HOURLY_SELECTED_CANDIDATE_ROW_LEVEL_RERUN_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_SELECTED_CANDIDATE_ROW_LEVEL_RERUN_PROTOCOL.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_AUDIT_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_AUDIT_PROTOCOL.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_AUDIT_RESULT.md` | `reports/results/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_AUDIT_RESULT.md` | active_result | yes | Current result, audit, diagnosis, inventory, or analysis file. | yes | yes | yes |
| `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_CLAIM_BOUNDARY.md` | `reports/claims/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_CLAIM_BOUNDARY.md` | active_claim | yes | Claim register, claim note, or claim-boundary file. | yes | yes | yes |
| `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_ROBUSTNESS_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_ROBUSTNESS_PROTOCOL.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_ROBUSTNESS_RESULT.md` | `reports/results/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_ROBUSTNESS_RESULT.md` | active_result | yes | Current result, audit, diagnosis, inventory, or analysis file. | yes | yes | yes |
| `reports/VN30_HOURLY_TRACK_A_BASELINE60_IMPROVEMENT_V1_CLAIM_REGISTER.md` | `reports/claims/VN30_HOURLY_TRACK_A_BASELINE60_IMPROVEMENT_V1_CLAIM_REGISTER.md` | active_claim | yes | Claim register, claim note, or claim-boundary file. | yes | yes | yes |
| `reports/VN30_HOURLY_TRACK_A_BASELINE60_IMPROVEMENT_V1_RESULT.md` | `reports/results/VN30_HOURLY_TRACK_A_BASELINE60_IMPROVEMENT_V1_RESULT.md` | active_result | yes | Current result, audit, diagnosis, inventory, or analysis file. | yes | yes | yes |
| `reports/VN30_HOURLY_TRACK_A_DIAGNOSTIC_65_ROW_AUDIT_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_TRACK_A_DIAGNOSTIC_65_ROW_AUDIT_PROTOCOL.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_HOURLY_TRACK_A_DIAGNOSTIC_65_ROW_CLAIM_BOUNDARY.md` | `reports/claims/VN30_HOURLY_TRACK_A_DIAGNOSTIC_65_ROW_CLAIM_BOUNDARY.md` | active_claim | yes | Claim register, claim note, or claim-boundary file. | yes | yes | yes |
| `reports/VN30_HOURLY_TRACK_A_REGIME_FEATURE_IMPROVEMENT_V2_CLAIM_REGISTER.md` | `reports/claims/VN30_HOURLY_TRACK_A_REGIME_FEATURE_IMPROVEMENT_V2_CLAIM_REGISTER.md` | active_claim | yes | Claim register, claim note, or claim-boundary file. | yes | yes | yes |
| `reports/VN30_HOURLY_TRACK_A_REGIME_FEATURE_IMPROVEMENT_V2_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_TRACK_A_REGIME_FEATURE_IMPROVEMENT_V2_PROTOCOL.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_HOURLY_TRACK_A_REGIME_FEATURE_IMPROVEMENT_V2_RESULT.md` | `reports/results/VN30_HOURLY_TRACK_A_REGIME_FEATURE_IMPROVEMENT_V2_RESULT.md` | active_result | yes | Current result, audit, diagnosis, inventory, or analysis file. | yes | yes | yes |
| `reports/VN30_HOURLY_TRACK_A_SELECTION_MISMATCH_ANALYSIS.md` | `reports/results/VN30_HOURLY_TRACK_A_SELECTION_MISMATCH_ANALYSIS.md` | active_result | yes | Current result, audit, diagnosis, inventory, or analysis file. | yes | yes | yes |
| `reports/VN30_HOURLY_TRACK_A_TARGET62_VALIDATION_SAFE_CLAIM_REGISTER.md` | `reports/claims/VN30_HOURLY_TRACK_A_TARGET62_VALIDATION_SAFE_CLAIM_REGISTER.md` | active_claim | yes | Claim register, claim note, or claim-boundary file. | yes | yes | yes |
| `reports/VN30_HOURLY_TRACK_A_TARGET62_VALIDATION_SAFE_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_TRACK_A_TARGET62_VALIDATION_SAFE_PROTOCOL.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_HOURLY_TRACK_A_TARGET62_VALIDATION_SAFE_RESULT.md` | `reports/results/VN30_HOURLY_TRACK_A_TARGET62_VALIDATION_SAFE_RESULT.md` | active_result | yes | Current result, audit, diagnosis, inventory, or analysis file. | yes | yes | yes |
| `reports/VN30_HOURLY_TRUE_STACKING_ALL_ALGORITHMS_CLAIM_REGISTER.md` | `reports/claims/VN30_HOURLY_TRUE_STACKING_ALL_ALGORITHMS_CLAIM_REGISTER.md` | active_claim | yes | Claim register, claim note, or claim-boundary file. | yes | yes | yes |
| `reports/VN30_HOURLY_TRUE_STACKING_ALL_ALGORITHMS_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_TRUE_STACKING_ALL_ALGORITHMS_PROTOCOL.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_HOURLY_TRUE_STACKING_ALL_ALGORITHMS_RESULT.md` | `reports/results/VN30_HOURLY_TRUE_STACKING_ALL_ALGORITHMS_RESULT.md` | active_result | yes | Current result, audit, diagnosis, inventory, or analysis file. | yes | yes | yes |
| `reports/VN30_INDEX_BENCHMARK_CLAIM_REGISTER.md` | `reports/claims/VN30_INDEX_BENCHMARK_CLAIM_REGISTER.md` | active_claim | yes | Claim register, claim note, or claim-boundary file. | yes | yes | yes |
| `reports/VN30_INDEX_BENCHMARK_RESULT_SUMMARY.md` | `reports/results/VN30_INDEX_BENCHMARK_RESULT_SUMMARY.md` | active_result | yes | Current result, audit, diagnosis, inventory, or analysis file. | yes | yes | yes |
| `reports/VN30_RESEARCH_CLAIM_REGISTER.md` | `reports/claims/VN30_RESEARCH_CLAIM_REGISTER.md` | active_claim | yes | Claim register, claim note, or claim-boundary file. | yes | yes | yes |
| `reports/VN30_STOCK_INDEX_JOINT_PANEL_DATA_SOURCE_RECOVERY_PROTOCOL.md` | `reports/protocols/VN30_STOCK_INDEX_JOINT_PANEL_DATA_SOURCE_RECOVERY_PROTOCOL.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_STOCK_INDEX_JOINT_PANEL_NEXT_ACTION_AFTER_UNIVERSE_FIX.md` | `reports/protocols/VN30_STOCK_INDEX_JOINT_PANEL_NEXT_ACTION_AFTER_UNIVERSE_FIX.md` | active_protocol | yes | Current next-action protocol note. | yes | yes | yes |
| `reports/VN30_STOCK_INDEX_JOINT_PANEL_READINESS_REPAIR_DECISION.md` | `reports/protocols/VN30_STOCK_INDEX_JOINT_PANEL_READINESS_REPAIR_DECISION.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_STOCK_INDEX_JOINT_PANEL_TRAINING_PROTOCOL.md` | `reports/protocols/VN30_STOCK_INDEX_JOINT_PANEL_TRAINING_PROTOCOL.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VN30_STOCK_INDEX_JOINT_PANEL_TRAINING_V1_CLAIM_REGISTER.md` | `reports/claims/VN30_STOCK_INDEX_JOINT_PANEL_TRAINING_V1_CLAIM_REGISTER.md` | active_claim | yes | Claim register, claim note, or claim-boundary file. | yes | yes | yes |
| `reports/VN30_STOCK_INDEX_JOINT_PANEL_TRAINING_V1_RESULT.md` | `reports/results/VN30_STOCK_INDEX_JOINT_PANEL_TRAINING_V1_RESULT.md` | active_result | yes | Current result, audit, diagnosis, inventory, or analysis file. | yes | yes | yes |
| `reports/VN30_STOCK_INDEX_JOINT_PANEL_UNIVERSE_CORRECTION.md` | `reports/protocols/VN30_STOCK_INDEX_JOINT_PANEL_UNIVERSE_CORRECTION.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VNSTOCK_AGENT_DATA_GUIDE.md` | `reports/protocols/VNSTOCK_AGENT_DATA_GUIDE.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VNSTOCK_AGENT_DATA_GUIDE_SUMMARY.md` | `reports/protocols/VNSTOCK_AGENT_DATA_GUIDE_SUMMARY.md` | active_protocol | yes | Current protocol, design, guide, or decision file. | yes | yes | yes |
| `reports/VNSTOCK_DATA_INTERPRETER_FIX_PLAN.md` | `reports/cleanup/VNSTOCK_DATA_INTERPRETER_FIX_PLAN.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/VNSTOCK_DATA_REPO_SOURCE_INVESTIGATION.md` | `reports/results/VNSTOCK_DATA_REPO_SOURCE_INVESTIGATION.md` | active_result | yes | Data-source investigation result used by active data-forensics evidence. | yes | yes | yes |
| `reports/VNSTOCK_PROVIDER_STANDARDIZATION_CHANGELOG.md` | `reports/cleanup/VNSTOCK_PROVIDER_STANDARDIZATION_CHANGELOG.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/VNSTOCK_PROVIDER_STANDARDIZATION_GUIDE.md` | `reports/cleanup/VNSTOCK_PROVIDER_STANDARDIZATION_GUIDE.md` | cleanup_governance | yes | Repository cleanup, audit remediation, push, or governance report. | yes | yes | yes |
| `reports/VSEF_1000_SEED_SMOKE_EXECUTIVE_SUMMARY.md` | `reports/superseded/VSEF_1000_SEED_SMOKE_EXECUTIVE_SUMMARY.md` | historical_or_superseded | no | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes | yes | yes |
| `reports/VSEF_1000_SEED_SMOKE_STABILITY_REPORT.md` | `reports/superseded/VSEF_1000_SEED_SMOKE_STABILITY_REPORT.md` | historical_or_superseded | no | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes | yes | yes |
| `reports/VSEF_PHASE3_ALLOCATOR_ROUTER_REAL_OUTPUT_SMOKE_REPORT.md` | `reports/superseded/VSEF_PHASE3_ALLOCATOR_ROUTER_REAL_OUTPUT_SMOKE_REPORT.md` | historical_or_superseded | no | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes | yes | yes |
| `reports/VSEF_QUANT_CORE_PHASE3_GAP_CLOSURE_PLAN.md` | `reports/superseded/VSEF_QUANT_CORE_PHASE3_GAP_CLOSURE_PLAN.md` | historical_or_superseded | no | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes | yes | yes |
| `reports/WALK_FORWARD_BENCHMARKING_ENSEMBLE_BASELINE_MODELS_VN_STOCK_MARKET.md` | `reports/superseded/WALK_FORWARD_BENCHMARKING_ENSEMBLE_BASELINE_MODELS_VN_STOCK_MARKET.md` | historical_or_superseded | no | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes | yes | yes |
