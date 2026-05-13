# Repo vnstock Provider Path Verification

- sys.executable: `C:\Users\luong\.venv\Scripts\python.exe`
- cwd: `K:\Repos\VSEF-PTIT-NEU`
- local shadowing detected: no
- vnstock_data import success: yes
- vnstock import success: yes
- repo adapter file exists: yes
- repo adapter import success: yes
- VN30 scripts bypass repo adapter: yes

## sys.path First 10

1. `K:\Repos\VSEF-PTIT-NEU\scripts\research`
2. `c:\Python\python313.zip`
3. `c:\Python\DLLs`
4. `c:\Python\Lib`
5. `c:\Python`
6. `C:\Users\luong\.venv`
7. `C:\Users\luong\.venv\Lib\site-packages`

## Recommended Provider Path

A. repo adapter first

B. vnstock_data direct second

C. legacy vnstock fallback third

## VN30 Hourly Script Observations

| script | uses repo adapter | uses direct provider |
|---|---:|---:|
| `scripts/research/audit_vn30_hourly_available_window.py` | no | no |
| `scripts/research/audit_vn30_hourly_coverage_2005_2026.py` | no | no |
| `scripts/research/build_vn30_hourly_available_window_paper_artifact_pack.py` | no | no |
| `scripts/research/build_vn30_hourly_listing_aware_paper_artifact_pack.py` | no | no |
| `scripts/research/build_vn30_hourly_paper_artifact_pack_2005_2026.py` | no | no |
| `scripts/research/build_vn30_hourly_vnstock_paper_artifact_pack.py` | no | yes |
| `scripts/research/diagnose_vn30_vnstock_hourly_fetch_failures.py` | no | no |
| `scripts/research/fetch_vn30_hourly_from_vnstock_2005_2026.py` | no | yes |
| `scripts/research/fetch_vn30_hourly_listing_aware_from_vnstock.py` | no | yes |
| `scripts/research/run_vn30_hourly_available_window_benchmark.py` | no | no |
| `scripts/research/run_vn30_hourly_available_window_confidence_sweep.py` | no | no |
| `scripts/research/run_vn30_hourly_available_window_cost_slippage_validation.py` | no | no |
| `scripts/research/run_vn30_hourly_available_window_exante_regime_validation.py` | no | no |
| `scripts/research/run_vn30_hourly_benchmark_2005_2026.py` | no | no |
| `scripts/research/run_vn30_hourly_benchmark_2005_2026_from_fetched.py` | no | yes |
| `scripts/research/run_vn30_hourly_confidence_sweep_2005_2026.py` | no | no |
| `scripts/research/run_vn30_hourly_cost_slippage_validation_2005_2026.py` | no | no |
| `scripts/research/run_vn30_hourly_exante_regime_validation_2005_2026.py` | no | no |
| `scripts/research/run_vn30_hourly_listing_aware_benchmark.py` | no | yes |
| `scripts/research/run_vn30_hourly_listing_aware_confidence_sweep.py` | no | no |
| `scripts/research/run_vn30_hourly_listing_aware_cost_slippage_validation.py` | no | no |
| `scripts/research/run_vn30_hourly_listing_aware_exante_regime_validation.py` | no | no |
| `scripts/research/run_vn30_hourly_vnstock_confidence_sweep.py` | no | yes |
| `scripts/research/run_vn30_hourly_vnstock_cost_slippage_validation.py` | no | yes |
| `scripts/research/run_vn30_hourly_vnstock_exante_regime_validation.py` | no | yes |
| `scripts/research/validate_external_vn30_hourly_dataset.py` | no | no |
| `scripts/research/validate_fetched_vn30_hourly_2005_2026.py` | no | no |
| `scripts/research/validate_vn30_hourly_listing_aware_dataset.py` | no | yes |
| `scripts/research/vn30_hourly_available_window_common.py` | no | no |
| `scripts/research/vn30_hourly_common.py` | no | yes |
| `scripts/research/vn30_hourly_listing_aware_common.py` | no | yes |
| `scripts/research/vn30_hourly_vnstock_common.py` | no | yes |

## Local Shadow Paths

- none
