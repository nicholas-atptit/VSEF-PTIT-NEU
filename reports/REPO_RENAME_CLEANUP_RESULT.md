# Repository Rename Cleanup Result

- Branch: `research/vn100-evidence-hardening-v1`
- Files renamed count: 22
- Folders renamed count: 0
- Files kept unchanged count: 13 inventory rows
- Protected files touched: no
- Data/output/archive renamed: no
- References updated: yes
- Active evidence index updated: yes
- README/docs updated: yes
- Repository structure guide created: yes
- Benchmark run: no
- Data fetch: no
- Model training: no
- Paper/DOCX generated: no
- Tags created: no
- Tags pushed: no
- Main touched: no

## Validation Results

- `python scripts/check_repo_hygiene.py`: passed.
- `python scripts/check_runtime_preflight.py`: passed with warnings only; summary `ok=40 warn=20 fail=0`.
- `C:\Users\luong\.venv\Scripts\python.exe scripts\check_runtime_preflight.py`: passed with warnings only; summary `ok=44 warn=16 fail=0`.
- `C:\Users\luong\.venv\Scripts\python.exe scripts\check_provider_usage_policy.py`: passed.
- `C:\Users\luong\.venv\Scripts\python.exe -m pytest tests/data/test_provider_usage_policy.py -q`: passed, 2 tests.
- `C:\Users\luong\.venv\Scripts\python.exe -m pytest tests/data/test_vn_price_gateway_contract.py -q`: passed, 7 tests.
- `C:\Users\luong\.venv\Scripts\python.exe -m pytest tests/ml/test_directional_accuracy_metrics.py -q`: passed, 12 tests.
- `powershell -ExecutionPolicy Bypass -File scripts/dev_tasks.ps1 -Task validate-all`: passed.
- `py_compile` for requested renamed scripts: passed for all four requested paths.

Observed warnings were environment or cache warnings only: optional imports missing in one interpreter, local database/Redis/Kafka/Chroma/Ollama services not reachable, `data\market_proxy.csv` missing, pytest cache write warning under `.pytest_cache`, and `requests` character-detection dependency warning. No validation command failed.

## Rename Mapping

| old_path | new_path | reason | references_updated |
| --- | --- | --- | --- |
| `reports/PAPER_FIGURE_DATA_SOURCE_INVENTORY.md` | `reports/VN30_HOURLY_PAPER_FIGURE_DATA_SOURCE_INVENTORY.md` | Add explicit VN30 hourly paper-source scope. | yes |
| `reports/PAPER_TABLE_FIGURE_CAPTIONS_EN.md` | `reports/VN30_HOURLY_PAPER_TABLE_FIGURE_CAPTIONS_EN.md` | Add explicit VN30 hourly scope and language suffix. | yes |
| `reports/PAPER_TABLE_FIGURE_CAPTIONS_VI.md` | `reports/VN30_HOURLY_PAPER_TABLE_FIGURE_CAPTIONS_VI.md` | Add explicit VN30 hourly scope and language suffix. | yes |
| `reports/PAPER_MISSING_METRICS_TODO.md` | `reports/VN30_HOURLY_PAPER_MISSING_METRICS_TODO.md` | Remove vague paper-only name. | yes |
| `reports/PAPER_ROW_LEVEL_FIGURE_TODO.md` | `reports/VN30_HOURLY_PAPER_ROW_LEVEL_FIGURE_TODO.md` | Remove vague paper-only name. | yes |
| `reports/PAPER_LITERATURE_DATA_TODO.md` | `reports/VN30_HOURLY_PAPER_LITERATURE_DATA_TODO.md` | Remove vague paper-only name. | yes |
| `reports/PAPER_DOCX_MISSING_FOR_FIGURE_INSERTION.md` | `reports/VN30_HOURLY_PAPER_DOCX_MISSING_FOR_FIGURE_INSERTION.md` | Add explicit VN30 hourly paper/DOCX source scope. | yes |
| `reports/PAPER_WITH_FIGURES_LAYOUT_QA.md` | `reports/VN30_HOURLY_PAPER_WITH_FIGURES_LAYOUT_QA.md` | Add explicit VN30 hourly paper-source scope. | yes |
| `reports/VN30_HOURLY_TARGET62_PAPER_READY_CLAIM_BOUNDARY.md` | `reports/VN30_HOURLY_SELECTED_CANDIDATE_CLAIM_BOUNDARY.md` | Replace target62/paper-ready wording with selected-candidate scope. | yes |
| `reports/VN30_HOURLY_TARGET62_STABILITY_CLAIM_BOUNDARY.md` | `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_CLAIM_BOUNDARY.md` | Clarify selected-candidate stability boundary. | yes |
| `reports/VN30_HOURLY_TARGET62_STABILITY_ROBUSTNESS_PROTOCOL.md` | `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_ROBUSTNESS_PROTOCOL.md` | Preserve protocol while avoiding misleading target62 framing. | yes |
| `reports/VN30_HOURLY_TARGET62_STABILITY_ROBUSTNESS_RESULT.md` | `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_ROBUSTNESS_RESULT.md` | Preserve result while avoiding misleading target62 framing. | yes |
| `reports/VN30_HOURLY_TARGET62_PAPER_READY_STABILITY_AUDIT_PROTOCOL.md` | `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_AUDIT_PROTOCOL.md` | Use selected-candidate naming. | yes |
| `reports/VN30_HOURLY_TARGET62_PAPER_READY_STABILITY_AUDIT_RESULT.md` | `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_AUDIT_RESULT.md` | Use selected-candidate naming. | yes |
| `reports/VN30_DAILY_2015_BENCHMARK_RESULT_SUMMARY.md` | `reports/VN30_DAILY_2015_RESULT_SUMMARY.md` | Shorten daily result summary name. | yes |
| `reports/INDEX_DIRECTIONAL_BENCHMARK_RESULT_SUMMARY.md` | `reports/VN30_INDEX_BENCHMARK_RESULT_SUMMARY.md` | Align index benchmark with VN30 research identity. | yes |
| `reports/INDEX_DIRECTIONAL_BENCHMARK_CLAIM_REGISTER.md` | `reports/VN30_INDEX_BENCHMARK_CLAIM_REGISTER.md` | Align claim register with VN30 research identity. | yes |
| `scripts/research/rerun_vn30_hourly_selected_l2_logistic_h40_row_predictions.py` | `scripts/research/rerun_vn30_hourly_selected_candidate_row_predictions.py` | Use action-based selected-candidate name without model-specific clutter. | yes |
| `scripts/research/build_paper_empirical_tables.py` | `scripts/research/build_vn30_hourly_paper_empirical_tables.py` | Add VN30 hourly paper-source scope. | yes |
| `scripts/research/build_paper_empirical_figures.py` | `scripts/research/build_vn30_hourly_paper_empirical_figures.py` | Add VN30 hourly paper-source scope. | yes |
| `scripts/research/audit_vn30_hourly_target62_paper_ready_stability.py` | `scripts/research/audit_vn30_hourly_selected_candidate_stability_summary.py` | Replace target62/paper-ready wording with selected-candidate stability scope. | yes |
| `scripts/research/audit_vn30_hourly_target62_stability_robustness.py` | `scripts/research/audit_vn30_hourly_selected_candidate_stability_robustness.py` | Replace target62 wording with selected-candidate robustness scope. | yes |

## Kept Unchanged

- Generated folders under `reports/generated/` were not renamed because they are evidence artifacts and may be referenced by reports/manifests.
- Raw data under `data/`, benchmark outputs under `outputs/`, and snapshots under `archive/generated_data_snapshots/` were not renamed.
- Legacy scripts under `scripts/legacy/research/` were inventoried but not renamed to preserve archival references.
