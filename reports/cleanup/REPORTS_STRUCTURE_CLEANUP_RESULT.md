# Reports Structure Cleanup Result

- Branch: `research/vn100-evidence-hardening-v1`
- Files moved count: 138
- Folders created count: 8
- `reports/generated/` touched: no
- Data/output/archive touched: no
- Active evidence updated: yes
- References updated: yes
- Files left in root `reports/` count: 1 Markdown file (`reports/README.md`); 10 total root files including pre-existing non-Markdown artifacts.
- Manual review count: 2
- Superseded count: 18
- Benchmark run: no
- Data fetch: no
- Model training: no
- Paper/DOCX generated: no
- Tags created: no
- Tags pushed: no
- Main touched: no

## Files Moved By Category

- `active_claim`: 18
- `active_index`: 5
- `active_protocol`: 31
- `active_result`: 25
- `cleanup_governance`: 31
- `historical_or_superseded`: 18
- `manual_review`: 2
- `paper_support`: 8

## Validation Results

- `python scripts/check_repo_hygiene.py`: passed after updating the moved cleanup-report allowlist path.
- `python scripts/check_runtime_preflight.py`: passed with warnings only; summary `ok=40 warn=20 fail=0`.
- `C:\Users\luong\.venv\Scripts\python.exe scripts\check_runtime_preflight.py`: passed with warnings only; summary `ok=44 warn=16 fail=0`.
- `C:\Users\luong\.venv\Scripts\python.exe scripts\check_provider_usage_policy.py`: passed.
- `C:\Users\luong\.venv\Scripts\python.exe -m pytest tests/data/test_provider_usage_policy.py -q`: passed, 2 tests.
- `C:\Users\luong\.venv\Scripts\python.exe -m pytest tests/data/test_vn_price_gateway_contract.py -q`: passed, 7 tests.
- `C:\Users\luong\.venv\Scripts\python.exe -m pytest tests/ml/test_directional_accuracy_metrics.py -q`: passed, 12 tests.
- `powershell -ExecutionPolicy Bypass -File scripts/dev_tasks.ps1 -Task validate-all`: passed.

Observed warnings were environment/cache warnings only: optional packages or local services unavailable, `data\market_proxy.csv` missing, pytest cache write warning under `.pytest_cache`, and the `requests` character-detection dependency warning. No validation command failed.

## Reference Review Notes

- Active references in `README.md`, `docs/`, active reports, and scripts were updated to category paths.
- Historical mapping tables in cleanup reports intentionally retain old paths in `old_path` or `current path` columns.
- Non-Markdown root artifacts and `reports/generated/` were not edited or moved.

## Mapping

| old_path | new_path | category | reason | references_updated |
| --- | --- | --- | --- | --- |
| `reports/ACTIVE_CODE_MAP.md` | `reports/_index/ACTIVE_CODE_MAP.md` | active_index | Central navigation or active status index. | yes |
| `reports/ACTIVE_EVIDENCE_INDEX.md` | `reports/_index/ACTIVE_EVIDENCE_INDEX.md` | active_index | Central navigation or active status index. | yes |
| `reports/AUDIT_REMEDIATION_CHECKLIST.md` | `reports/cleanup/AUDIT_REMEDIATION_CHECKLIST.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/baseline_local_qwen_4b_report.md` | `reports/superseded/baseline_local_qwen_4b_report.md` | historical_or_superseded | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes |
| `reports/CODE_AUDIT_REMEDIATION_CLOSEOUT.md` | `reports/cleanup/CODE_AUDIT_REMEDIATION_CLOSEOUT.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/CODE_AUDIT_REMEDIATION_PLAN.md` | `reports/cleanup/CODE_AUDIT_REMEDIATION_PLAN.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/CODE_AUDIT_REPORT.md` | `reports/cleanup/CODE_AUDIT_REPORT.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/CODE_CLEANUP_CHANGES.md` | `reports/cleanup/CODE_CLEANUP_CHANGES.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/CODE_CLEANUP_SCOPE.md` | `reports/cleanup/CODE_CLEANUP_SCOPE.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/CODE_CLEANUP_VALIDATION_FIX_PLAN.md` | `reports/cleanup/CODE_CLEANUP_VALIDATION_FIX_PLAN.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/CODE_CLEANUP_VALIDATION_FIX_RESULT.md` | `reports/cleanup/CODE_CLEANUP_VALIDATION_FIX_RESULT.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/comparison_qwen_4b_vs_8b.md` | `reports/superseded/comparison_qwen_4b_vs_8b.md` | historical_or_superseded | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes |
| `reports/DOCS_AND_TASK_RUNNER_CLEANUP_RESULT.md` | `reports/cleanup/DOCS_AND_TASK_RUNNER_CLEANUP_RESULT.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/FULL_DATA_PUSH_INVENTORY.md` | `reports/cleanup/FULL_DATA_PUSH_INVENTORY.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/FULL_DATA_PUSH_RESULT.md` | `reports/cleanup/FULL_DATA_PUSH_RESULT.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/FULL_DATA_PUSH_STAGING_REVIEW.md` | `reports/cleanup/FULL_DATA_PUSH_STAGING_REVIEW.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/full_system_report.md` | `reports/superseded/full_system_report.md` | historical_or_superseded | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes |
| `reports/GENERATED_REPORTS_STATUS.md` | `reports/_index/GENERATED_REPORTS_STATUS.md` | active_index | Central navigation or active status index. | yes |
| `reports/INDEX_HOURLY_FETCH_README.md` | `reports/protocols/INDEX_HOURLY_FETCH_README.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/MANUAL_REVIEW_SCRIPT_RESOLUTION.md` | `reports/manual_review/MANUAL_REVIEW_SCRIPT_RESOLUTION.md` | manual_review | Manual review or unresolved cleanup note. | yes |
| `reports/NCKH_EVIDENCE_GAP_CLOSURE_REGISTER.md` | `reports/superseded/NCKH_EVIDENCE_GAP_CLOSURE_REGISTER.md` | historical_or_superseded | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes |
| `reports/NCKH_EVIDENCE_UPGRADE_ADDENDUM_V2.md` | `reports/superseded/NCKH_EVIDENCE_UPGRADE_ADDENDUM_V2.md` | historical_or_superseded | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes |
| `reports/NCKH_EXPERIMENT_INVENTORY.md` | `reports/superseded/NCKH_EXPERIMENT_INVENTORY.md` | historical_or_superseded | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes |
| `reports/NCKH_RESEARCH_DESIGN_VN100.md` | `reports/superseded/NCKH_RESEARCH_DESIGN_VN100.md` | historical_or_superseded | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes |
| `reports/NCKH_RESULTS_CLAIM_REGISTER.md` | `reports/superseded/NCKH_RESULTS_CLAIM_REGISTER.md` | historical_or_superseded | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes |
| `reports/NCKH_RESULTS_CLAIM_REGISTER_V2.md` | `reports/superseded/NCKH_RESULTS_CLAIM_REGISTER_V2.md` | historical_or_superseded | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes |
| `reports/NEW_REMOTE_PUSH_VERIFICATION.md` | `reports/cleanup/NEW_REMOTE_PUSH_VERIFICATION.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/PHASE2_MOCK_PROVENANCE_EVIDENCE.md` | `reports/cleanup/PHASE2_MOCK_PROVENANCE_EVIDENCE.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/PHASE3_REPOSITORY_HYGIENE_EVIDENCE.md` | `reports/cleanup/PHASE3_REPOSITORY_HYGIENE_EVIDENCE.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/PHASE4_API_GOVERNANCE_EVIDENCE.md` | `reports/cleanup/PHASE4_API_GOVERNANCE_EVIDENCE.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/PHASE4A_FRONTEND_DESCOPE_EVIDENCE.md` | `reports/cleanup/PHASE4A_FRONTEND_DESCOPE_EVIDENCE.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/PHASE5_REPRODUCIBILITY_COMMAND_REGISTRY_EVIDENCE.md` | `reports/cleanup/PHASE5_REPRODUCIBILITY_COMMAND_REGISTRY_EVIDENCE.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/PHASE6_STATISTICAL_ACCEPTANCE_EVIDENCE.md` | `reports/cleanup/PHASE6_STATISTICAL_ACCEPTANCE_EVIDENCE.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/PHASE7_RUNTIME_PATH_REGISTRY_EVIDENCE.md` | `reports/cleanup/PHASE7_RUNTIME_PATH_REGISTRY_EVIDENCE.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/PHASE8_GOVERNED_INTEGRATION_TESTING_EVIDENCE.md` | `reports/cleanup/PHASE8_GOVERNED_INTEGRATION_TESTING_EVIDENCE.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/PROVIDER_STANDARDIZATION_AUDIT.md` | `reports/cleanup/PROVIDER_STANDARDIZATION_AUDIT.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/README_REWRITE_AND_IDENTITY_CLEANUP.md` | `reports/cleanup/README_REWRITE_AND_IDENTITY_CLEANUP.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/REPO_CLEANUP_INVENTORY.md` | `reports/_index/REPO_CLEANUP_INVENTORY.md` | active_index | Central navigation or active status index. | yes |
| `reports/REPO_CLEANUP_MANUAL_REVIEW.md` | `reports/manual_review/REPO_CLEANUP_MANUAL_REVIEW.md` | manual_review | Manual review or unresolved cleanup note. | yes |
| `reports/REPO_RENAME_CLEANUP_INVENTORY.md` | `reports/cleanup/REPO_RENAME_CLEANUP_INVENTORY.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/REPO_RENAME_CLEANUP_RESULT.md` | `reports/cleanup/REPO_RENAME_CLEANUP_RESULT.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/REPO_REPORTS_CLEANUP_INVENTORY.md` | `reports/cleanup/REPO_REPORTS_CLEANUP_INVENTORY.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | not needed |
| `reports/RESEARCH_SCRIPT_STATUS.md` | `reports/_index/RESEARCH_SCRIPT_STATUS.md` | active_index | Central navigation or active status index. | yes |
| `reports/RESEARCH_TOPIC_AND_CODE_AUDIT_REPORT.md` | `reports/results/RESEARCH_TOPIC_AND_CODE_AUDIT_REPORT.md` | active_result | Current result, audit, diagnosis, inventory, or analysis file. | yes |
| `reports/risk_tuning_report.md` | `reports/superseded/risk_tuning_report.md` | historical_or_superseded | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes |
| `reports/SCRIPT_REORGANIZATION_PLAN.md` | `reports/cleanup/SCRIPT_REORGANIZATION_PLAN.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/SCRIPT_REORGANIZATION_RESULT.md` | `reports/cleanup/SCRIPT_REORGANIZATION_RESULT.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/stress_test_report.md` | `reports/superseded/stress_test_report.md` | historical_or_superseded | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes |
| `reports/system_benchmark.md` | `reports/superseded/system_benchmark.md` | historical_or_superseded | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes |
| `reports/VN100_HYBRID_BENCHMARK_CLOSEOUT.md` | `reports/superseded/VN100_HYBRID_BENCHMARK_CLOSEOUT.md` | historical_or_superseded | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes |
| `reports/VN30_DAILY_2015_BCM_VIB_RECOVERY_REPORT.md` | `reports/results/VN30_DAILY_2015_BCM_VIB_RECOVERY_REPORT.md` | active_result | Current result, audit, diagnosis, inventory, or analysis file. | yes |
| `reports/VN30_DAILY_2015_CLAIM_REGISTER.md` | `reports/claims/VN30_DAILY_2015_CLAIM_REGISTER.md` | active_claim | Claim register, claim note, or claim-boundary file. | yes |
| `reports/VN30_DAILY_2015_RESULT_SUMMARY.md` | `reports/results/VN30_DAILY_2015_RESULT_SUMMARY.md` | active_result | Current result, audit, diagnosis, inventory, or analysis file. | yes |
| `reports/VN30_DAILY_2015_SCOPE_AND_DESIGN.md` | `reports/protocols/VN30_DAILY_2015_SCOPE_AND_DESIGN.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_DAILY_2015_TARGET60_CLAIM_REGISTER.md` | `reports/claims/VN30_DAILY_2015_TARGET60_CLAIM_REGISTER.md` | active_claim | Claim register, claim note, or claim-boundary file. | yes |
| `reports/VN30_DAILY_2015_TARGET60_FAILURE_POSTMORTEM_PROTOCOL.md` | `reports/protocols/VN30_DAILY_2015_TARGET60_FAILURE_POSTMORTEM_PROTOCOL.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_DAILY_2015_TARGET60_NEXT_STEP_DECISION.md` | `reports/protocols/VN30_DAILY_2015_TARGET60_NEXT_STEP_DECISION.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_DAILY_2015_TARGET60_PROTOCOL.md` | `reports/protocols/VN30_DAILY_2015_TARGET60_PROTOCOL.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_DAILY_2015_TARGET60_RESULT_SUMMARY.md` | `reports/results/VN30_DAILY_2015_TARGET60_RESULT_SUMMARY.md` | active_result | Current result, audit, diagnosis, inventory, or analysis file. | yes |
| `reports/VN30_DAILY_2015_TARGET60_V2_CLAIM_REGISTER.md` | `reports/claims/VN30_DAILY_2015_TARGET60_V2_CLAIM_REGISTER.md` | active_claim | Claim register, claim note, or claim-boundary file. | yes |
| `reports/VN30_DAILY_2015_TARGET60_V2_PROTOCOL.md` | `reports/protocols/VN30_DAILY_2015_TARGET60_V2_PROTOCOL.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_DAILY_2015_TARGET60_V2_RESULT_SUMMARY.md` | `reports/results/VN30_DAILY_2015_TARGET60_V2_RESULT_SUMMARY.md` | active_result | Current result, audit, diagnosis, inventory, or analysis file. | yes |
| `reports/VN30_HOURLY_2015_BASELINE60_RESULT_LOCK.md` | `reports/results/VN30_HOURLY_2015_BASELINE60_RESULT_LOCK.md` | active_result | Current result, audit, diagnosis, inventory, or analysis file. | yes |
| `reports/VN30_HOURLY_2015_BENCHMARK_CLAIM_REGISTER.md` | `reports/claims/VN30_HOURLY_2015_BENCHMARK_CLAIM_REGISTER.md` | active_claim | Claim register, claim note, or claim-boundary file. | yes |
| `reports/VN30_HOURLY_2015_CANONICAL_EVALUATOR_DECISION.md` | `reports/protocols/VN30_HOURLY_2015_CANONICAL_EVALUATOR_DECISION.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_HOURLY_2015_DATA_READINESS_PLAN.md` | `reports/protocols/VN30_HOURLY_2015_DATA_READINESS_PLAN.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_HOURLY_2015_JAN2025_BENCHMARK_RESULT_SUMMARY.md` | `reports/results/VN30_HOURLY_2015_JAN2025_BENCHMARK_RESULT_SUMMARY.md` | active_result | Current result, audit, diagnosis, inventory, or analysis file. | yes |
| `reports/VN30_HOURLY_2015_JAN2025_UNIVERSE_DECISION.md` | `reports/protocols/VN30_HOURLY_2015_JAN2025_UNIVERSE_DECISION.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_HOURLY_2015_POST_BENCHMARK_DIAGNOSTICS_SUMMARY.md` | `reports/results/VN30_HOURLY_2015_POST_BENCHMARK_DIAGNOSTICS_SUMMARY.md` | active_result | Current result, audit, diagnosis, inventory, or analysis file. | yes |
| `reports/VN30_HOURLY_2015_TOPK_75_VERIFICATION_DECISION.md` | `reports/protocols/VN30_HOURLY_2015_TOPK_75_VERIFICATION_DECISION.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_HOURLY_2015_TOPK_75_VERIFICATION_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_2015_TOPK_75_VERIFICATION_PROTOCOL.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_HOURLY_2015_TOPK_RANKING_CLAIM_REGISTER.md` | `reports/claims/VN30_HOURLY_2015_TOPK_RANKING_CLAIM_REGISTER.md` | active_claim | Claim register, claim note, or claim-boundary file. | yes |
| `reports/VN30_HOURLY_2015_TOPK_RANKING_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_2015_TOPK_RANKING_PROTOCOL.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_HOURLY_2015_TOPK_RANKING_RESULT_SUMMARY.md` | `reports/results/VN30_HOURLY_2015_TOPK_RANKING_RESULT_SUMMARY.md` | active_result | Current result, audit, diagnosis, inventory, or analysis file. | yes |
| `reports/VN30_HOURLY_2015_VALIDATION_FINAL_MISMATCH_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_2015_VALIDATION_FINAL_MISMATCH_PROTOCOL.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_HOURLY_BASELINE60_CLAIM_NOTE.md` | `reports/claims/VN30_HOURLY_BASELINE60_CLAIM_NOTE.md` | active_claim | Claim register, claim note, or claim-boundary file. | yes |
| `reports/VN30_HOURLY_DATA_FORENSICS_DIAGNOSIS.md` | `reports/results/VN30_HOURLY_DATA_FORENSICS_DIAGNOSIS.md` | active_result | Current result, audit, diagnosis, inventory, or analysis file. | yes |
| `reports/VN30_HOURLY_DUAL_TRACK_MODEL_COMPARISON_CLAIM_REGISTER.md` | `reports/claims/VN30_HOURLY_DUAL_TRACK_MODEL_COMPARISON_CLAIM_REGISTER.md` | active_claim | Claim register, claim note, or claim-boundary file. | yes |
| `reports/VN30_HOURLY_DUAL_TRACK_MODEL_COMPARISON_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_DUAL_TRACK_MODEL_COMPARISON_PROTOCOL.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_HOURLY_DUAL_TRACK_MODEL_COMPARISON_RESULT.md` | `reports/results/VN30_HOURLY_DUAL_TRACK_MODEL_COMPARISON_RESULT.md` | active_result | Current result, audit, diagnosis, inventory, or analysis file. | yes |
| `reports/VN30_HOURLY_EXPANDED_MODEL_POOL_STACKING_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_EXPANDED_MODEL_POOL_STACKING_PROTOCOL.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_HOURLY_EXPANDED_MODEL_POOL_STACKING_V1_CLAIM_REGISTER.md` | `reports/claims/VN30_HOURLY_EXPANDED_MODEL_POOL_STACKING_V1_CLAIM_REGISTER.md` | active_claim | Claim register, claim note, or claim-boundary file. | yes |
| `reports/VN30_HOURLY_EXPANDED_MODEL_POOL_STACKING_V1_RESULT.md` | `reports/results/VN30_HOURLY_EXPANDED_MODEL_POOL_STACKING_V1_RESULT.md` | active_result | Current result, audit, diagnosis, inventory, or analysis file. | yes |
| `reports/VN30_HOURLY_MARKET_INDEX_CONTEXT_PLAN.md` | `reports/protocols/VN30_HOURLY_MARKET_INDEX_CONTEXT_PLAN.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_HOURLY_PAPER_DOCX_MISSING_FOR_FIGURE_INSERTION.md` | `reports/paper/VN30_HOURLY_PAPER_DOCX_MISSING_FOR_FIGURE_INSERTION.md` | paper_support | VN30 hourly paper/source support artifact. | yes |
| `reports/VN30_HOURLY_PAPER_FIGURE_DATA_SOURCE_INVENTORY.md` | `reports/paper/VN30_HOURLY_PAPER_FIGURE_DATA_SOURCE_INVENTORY.md` | paper_support | VN30 hourly paper/source support artifact. | yes |
| `reports/VN30_HOURLY_PAPER_LITERATURE_DATA_TODO.md` | `reports/paper/VN30_HOURLY_PAPER_LITERATURE_DATA_TODO.md` | paper_support | VN30 hourly paper/source support artifact. | yes |
| `reports/VN30_HOURLY_PAPER_MISSING_METRICS_TODO.md` | `reports/paper/VN30_HOURLY_PAPER_MISSING_METRICS_TODO.md` | paper_support | VN30 hourly paper/source support artifact. | yes |
| `reports/VN30_HOURLY_PAPER_ROW_LEVEL_FIGURE_TODO.md` | `reports/paper/VN30_HOURLY_PAPER_ROW_LEVEL_FIGURE_TODO.md` | paper_support | VN30 hourly paper/source support artifact. | yes |
| `reports/VN30_HOURLY_PAPER_TABLE_FIGURE_CAPTIONS_EN.md` | `reports/paper/VN30_HOURLY_PAPER_TABLE_FIGURE_CAPTIONS_EN.md` | paper_support | VN30 hourly paper/source support artifact. | yes |
| `reports/VN30_HOURLY_PAPER_TABLE_FIGURE_CAPTIONS_VI.md` | `reports/paper/VN30_HOURLY_PAPER_TABLE_FIGURE_CAPTIONS_VI.md` | paper_support | VN30 hourly paper/source support artifact. | yes |
| `reports/VN30_HOURLY_PAPER_WITH_FIGURES_LAYOUT_QA.md` | `reports/paper/VN30_HOURLY_PAPER_WITH_FIGURES_LAYOUT_QA.md` | paper_support | VN30 hourly paper/source support artifact. | yes |
| `reports/VN30_HOURLY_RF_H60_BASELINE_REPRODUCTION_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_RF_H60_BASELINE_REPRODUCTION_PROTOCOL.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_HOURLY_RF_H60_BASELINE_REPRODUCTION_RESULT.md` | `reports/results/VN30_HOURLY_RF_H60_BASELINE_REPRODUCTION_RESULT.md` | active_result | Current result, audit, diagnosis, inventory, or analysis file. | yes |
| `reports/VN30_HOURLY_SELECTED_CANDIDATE_ARTIFACT_DISCOVERY.md` | `reports/results/VN30_HOURLY_SELECTED_CANDIDATE_ARTIFACT_DISCOVERY.md` | active_result | Current result, audit, diagnosis, inventory, or analysis file. | yes |
| `reports/VN30_HOURLY_SELECTED_CANDIDATE_CLAIM_BOUNDARY.md` | `reports/claims/VN30_HOURLY_SELECTED_CANDIDATE_CLAIM_BOUNDARY.md` | active_claim | Claim register, claim note, or claim-boundary file. | yes |
| `reports/VN30_HOURLY_SELECTED_CANDIDATE_ROLLING_STABILITY_RESULT.md` | `reports/results/VN30_HOURLY_SELECTED_CANDIDATE_ROLLING_STABILITY_RESULT.md` | active_result | Current result, audit, diagnosis, inventory, or analysis file. | yes |
| `reports/VN30_HOURLY_SELECTED_CANDIDATE_ROW_LEVEL_RERUN_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_SELECTED_CANDIDATE_ROW_LEVEL_RERUN_PROTOCOL.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_AUDIT_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_AUDIT_PROTOCOL.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_AUDIT_RESULT.md` | `reports/results/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_AUDIT_RESULT.md` | active_result | Current result, audit, diagnosis, inventory, or analysis file. | yes |
| `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_CLAIM_BOUNDARY.md` | `reports/claims/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_CLAIM_BOUNDARY.md` | active_claim | Claim register, claim note, or claim-boundary file. | yes |
| `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_ROBUSTNESS_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_ROBUSTNESS_PROTOCOL.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_ROBUSTNESS_RESULT.md` | `reports/results/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_ROBUSTNESS_RESULT.md` | active_result | Current result, audit, diagnosis, inventory, or analysis file. | yes |
| `reports/VN30_HOURLY_TRACK_A_BASELINE60_IMPROVEMENT_V1_CLAIM_REGISTER.md` | `reports/claims/VN30_HOURLY_TRACK_A_BASELINE60_IMPROVEMENT_V1_CLAIM_REGISTER.md` | active_claim | Claim register, claim note, or claim-boundary file. | yes |
| `reports/VN30_HOURLY_TRACK_A_BASELINE60_IMPROVEMENT_V1_RESULT.md` | `reports/results/VN30_HOURLY_TRACK_A_BASELINE60_IMPROVEMENT_V1_RESULT.md` | active_result | Current result, audit, diagnosis, inventory, or analysis file. | yes |
| `reports/VN30_HOURLY_TRACK_A_DIAGNOSTIC_65_ROW_AUDIT_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_TRACK_A_DIAGNOSTIC_65_ROW_AUDIT_PROTOCOL.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_HOURLY_TRACK_A_DIAGNOSTIC_65_ROW_CLAIM_BOUNDARY.md` | `reports/claims/VN30_HOURLY_TRACK_A_DIAGNOSTIC_65_ROW_CLAIM_BOUNDARY.md` | active_claim | Claim register, claim note, or claim-boundary file. | yes |
| `reports/VN30_HOURLY_TRACK_A_REGIME_FEATURE_IMPROVEMENT_V2_CLAIM_REGISTER.md` | `reports/claims/VN30_HOURLY_TRACK_A_REGIME_FEATURE_IMPROVEMENT_V2_CLAIM_REGISTER.md` | active_claim | Claim register, claim note, or claim-boundary file. | yes |
| `reports/VN30_HOURLY_TRACK_A_REGIME_FEATURE_IMPROVEMENT_V2_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_TRACK_A_REGIME_FEATURE_IMPROVEMENT_V2_PROTOCOL.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_HOURLY_TRACK_A_REGIME_FEATURE_IMPROVEMENT_V2_RESULT.md` | `reports/results/VN30_HOURLY_TRACK_A_REGIME_FEATURE_IMPROVEMENT_V2_RESULT.md` | active_result | Current result, audit, diagnosis, inventory, or analysis file. | yes |
| `reports/VN30_HOURLY_TRACK_A_SELECTION_MISMATCH_ANALYSIS.md` | `reports/results/VN30_HOURLY_TRACK_A_SELECTION_MISMATCH_ANALYSIS.md` | active_result | Current result, audit, diagnosis, inventory, or analysis file. | yes |
| `reports/VN30_HOURLY_TRACK_A_TARGET62_VALIDATION_SAFE_CLAIM_REGISTER.md` | `reports/claims/VN30_HOURLY_TRACK_A_TARGET62_VALIDATION_SAFE_CLAIM_REGISTER.md` | active_claim | Claim register, claim note, or claim-boundary file. | yes |
| `reports/VN30_HOURLY_TRACK_A_TARGET62_VALIDATION_SAFE_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_TRACK_A_TARGET62_VALIDATION_SAFE_PROTOCOL.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_HOURLY_TRACK_A_TARGET62_VALIDATION_SAFE_RESULT.md` | `reports/results/VN30_HOURLY_TRACK_A_TARGET62_VALIDATION_SAFE_RESULT.md` | active_result | Current result, audit, diagnosis, inventory, or analysis file. | yes |
| `reports/VN30_HOURLY_TRUE_STACKING_ALL_ALGORITHMS_CLAIM_REGISTER.md` | `reports/claims/VN30_HOURLY_TRUE_STACKING_ALL_ALGORITHMS_CLAIM_REGISTER.md` | active_claim | Claim register, claim note, or claim-boundary file. | yes |
| `reports/VN30_HOURLY_TRUE_STACKING_ALL_ALGORITHMS_PROTOCOL.md` | `reports/protocols/VN30_HOURLY_TRUE_STACKING_ALL_ALGORITHMS_PROTOCOL.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_HOURLY_TRUE_STACKING_ALL_ALGORITHMS_RESULT.md` | `reports/results/VN30_HOURLY_TRUE_STACKING_ALL_ALGORITHMS_RESULT.md` | active_result | Current result, audit, diagnosis, inventory, or analysis file. | yes |
| `reports/VN30_INDEX_BENCHMARK_CLAIM_REGISTER.md` | `reports/claims/VN30_INDEX_BENCHMARK_CLAIM_REGISTER.md` | active_claim | Claim register, claim note, or claim-boundary file. | yes |
| `reports/VN30_INDEX_BENCHMARK_RESULT_SUMMARY.md` | `reports/results/VN30_INDEX_BENCHMARK_RESULT_SUMMARY.md` | active_result | Current result, audit, diagnosis, inventory, or analysis file. | yes |
| `reports/VN30_RESEARCH_CLAIM_REGISTER.md` | `reports/claims/VN30_RESEARCH_CLAIM_REGISTER.md` | active_claim | Claim register, claim note, or claim-boundary file. | yes |
| `reports/VN30_STOCK_INDEX_JOINT_PANEL_DATA_SOURCE_RECOVERY_PROTOCOL.md` | `reports/protocols/VN30_STOCK_INDEX_JOINT_PANEL_DATA_SOURCE_RECOVERY_PROTOCOL.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_STOCK_INDEX_JOINT_PANEL_NEXT_ACTION_AFTER_UNIVERSE_FIX.md` | `reports/protocols/VN30_STOCK_INDEX_JOINT_PANEL_NEXT_ACTION_AFTER_UNIVERSE_FIX.md` | active_protocol | Current next-action protocol note. | yes |
| `reports/VN30_STOCK_INDEX_JOINT_PANEL_READINESS_REPAIR_DECISION.md` | `reports/protocols/VN30_STOCK_INDEX_JOINT_PANEL_READINESS_REPAIR_DECISION.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_STOCK_INDEX_JOINT_PANEL_TRAINING_PROTOCOL.md` | `reports/protocols/VN30_STOCK_INDEX_JOINT_PANEL_TRAINING_PROTOCOL.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VN30_STOCK_INDEX_JOINT_PANEL_TRAINING_V1_CLAIM_REGISTER.md` | `reports/claims/VN30_STOCK_INDEX_JOINT_PANEL_TRAINING_V1_CLAIM_REGISTER.md` | active_claim | Claim register, claim note, or claim-boundary file. | yes |
| `reports/VN30_STOCK_INDEX_JOINT_PANEL_TRAINING_V1_RESULT.md` | `reports/results/VN30_STOCK_INDEX_JOINT_PANEL_TRAINING_V1_RESULT.md` | active_result | Current result, audit, diagnosis, inventory, or analysis file. | yes |
| `reports/VN30_STOCK_INDEX_JOINT_PANEL_UNIVERSE_CORRECTION.md` | `reports/protocols/VN30_STOCK_INDEX_JOINT_PANEL_UNIVERSE_CORRECTION.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VNSTOCK_AGENT_DATA_GUIDE.md` | `reports/protocols/VNSTOCK_AGENT_DATA_GUIDE.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VNSTOCK_AGENT_DATA_GUIDE_SUMMARY.md` | `reports/protocols/VNSTOCK_AGENT_DATA_GUIDE_SUMMARY.md` | active_protocol | Current protocol, design, guide, or decision file. | yes |
| `reports/VNSTOCK_DATA_INTERPRETER_FIX_PLAN.md` | `reports/cleanup/VNSTOCK_DATA_INTERPRETER_FIX_PLAN.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/VNSTOCK_DATA_REPO_SOURCE_INVESTIGATION.md` | `reports/results/VNSTOCK_DATA_REPO_SOURCE_INVESTIGATION.md` | active_result | Data-source investigation result used by active data-forensics evidence. | yes |
| `reports/VNSTOCK_PROVIDER_STANDARDIZATION_CHANGELOG.md` | `reports/cleanup/VNSTOCK_PROVIDER_STANDARDIZATION_CHANGELOG.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/VNSTOCK_PROVIDER_STANDARDIZATION_GUIDE.md` | `reports/cleanup/VNSTOCK_PROVIDER_STANDARDIZATION_GUIDE.md` | cleanup_governance | Repository cleanup, audit remediation, push, or governance report. | yes |
| `reports/VSEF_1000_SEED_SMOKE_EXECUTIVE_SUMMARY.md` | `reports/superseded/VSEF_1000_SEED_SMOKE_EXECUTIVE_SUMMARY.md` | historical_or_superseded | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes |
| `reports/VSEF_1000_SEED_SMOKE_STABILITY_REPORT.md` | `reports/superseded/VSEF_1000_SEED_SMOKE_STABILITY_REPORT.md` | historical_or_superseded | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes |
| `reports/VSEF_PHASE3_ALLOCATOR_ROUTER_REAL_OUTPUT_SMOKE_REPORT.md` | `reports/superseded/VSEF_PHASE3_ALLOCATOR_ROUTER_REAL_OUTPUT_SMOKE_REPORT.md` | historical_or_superseded | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes |
| `reports/VSEF_QUANT_CORE_PHASE3_GAP_CLOSURE_PLAN.md` | `reports/superseded/VSEF_QUANT_CORE_PHASE3_GAP_CLOSURE_PLAN.md` | historical_or_superseded | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes |
| `reports/WALK_FORWARD_BENCHMARKING_ENSEMBLE_BASELINE_MODELS_VN_STOCK_MARKET.md` | `reports/superseded/WALK_FORWARD_BENCHMARKING_ENSEMBLE_BASELINE_MODELS_VN_STOCK_MARKET.md` | historical_or_superseded | Preserved older VN100/VSEF/NCKH or legacy benchmark report. | yes |
