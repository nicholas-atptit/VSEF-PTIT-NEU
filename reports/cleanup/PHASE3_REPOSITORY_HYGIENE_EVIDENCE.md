# Phase 3 Repository Hygiene Evidence

Date: 2026-05-10
Branch: `audit-remediation-governed-runtime`
Scope: repository hygiene cleanup only; no runtime logic, API governance, allocator logic, VN100 governance, feature catalogue governance, or model/training behavior changes were made.

## Inventory

| Category | Result | Action |
| --- | --- | --- |
| Tracked front-end dependencies | 3333 tracked files under `src/api/ui/web/node_modules` before cleanup. | Removed from git tracking with `git rm --cached -r`; lockfile retained. |
| Root generated outputs | `outputs/experiments/.gitkeep` was tracked. | Removed from git tracking with `git rm --cached`. |
| Generated archive/root residue | 12 tracked generated files under `archive/root`. | Fully removed as generated logs, temp scripts, and local database residue. |
| Malformed filename residue | 1 tracked malformed shell-residue filename at repository root. | Fully removed as machine-local shell residue. |
| Python caches and egg metadata | No tracked `__pycache__`, Python bytecode, or egg-info paths after cleanup. | Enforced by hygiene script. |
| Source model modules | `src/ml/models` is source code, not a generated artifact. | Hygiene script root-anchors generated model directory checks. |

## Local Path Classification

| Classification | Files | Disposition |
| --- | --- | --- |
| Active source/config/helper script | `scripts/prepare_vip_list.py`, `scripts/run_experiment.py`, `scripts/legacy/scrub_links.py` | Cleaned to use repository-relative paths or constructed scanner tokens. |
| Active documentation/runbook | `docs/governance/PHASE3_ROUTER_V1.md`, `docs/governance/VSEF_FOREIGN_FLOW_DISABLE_MODE.md`, `docs/usage/VSEF_FOREIGN_FLOW_AUDIT_COMMANDS.md` | Cleaned to use generic command examples. |
| Generated artifact | `archive/root`, `outputs/experiments/.gitkeep`, `src/api/ui/web/node_modules` | Removed from tracking or fully removed according to Phase 3 policy. |
| Historical audit/report evidence | Phase 0 reports, archived docs, experiment reports, and historical report packs | Retained with explicit hygiene-script allowlist because they are evidence snapshots, not active runtime source/config. |
| Intentional scanner/test pattern | `scripts/legacy/scrub_links.py`, `scripts/check_repo_hygiene.py` | Literal scanner tokens are constructed from string fragments to avoid self-violations. |

## Hygiene Allowlist

Allowlist entries are limited to historical evidence paths that are not active runtime source/config:

- `reports/cleanup/CODE_AUDIT_REPORT.md`
- `reports/cleanup/CODE_AUDIT_REMEDIATION_PLAN.md`
- `docs/archive/`
- `docs/audits/`
- `docs/experiments/`
- `docs/reports/`
- `reports/forecasting_core/`
- `reports/repeated_seed_1000_smoke_report_pack/`
- `reports/risk_aware/`
- `reports/superseded/VSEF_1000_SEED_SMOKE_STABILITY_REPORT.md`
- `reports/superseded/stress_test_report.md`
- `reports/superseded/system_benchmark.md`

Justification: these files are retained as historical evidence snapshots. The allowlist does not apply to active source, helper scripts, runbooks, or generated dependency/output directories.

## Verification

| Command | Exit | Evidence |
| --- | --- | --- |
| `python scripts/check_repo_hygiene.py` | 0 | `Repository hygiene check passed.` |
| `npm.cmd ci` | 0 | Installed 193 packages from the tracked lockfile; npm reported 3 moderate audit vulnerabilities. |
| `npm.cmd run build` | 0 | Vite production build completed successfully. |
| `python -m pytest tests -q` | 0 | `807 passed, 5 skipped, 33 warnings in 262.15s`. |
| `git ls-files src/api/ui/web/node_modules` | 0 | Tracked count after cleanup: 0. |
| `git ls-files outputs artifacts models tmp src/api/ui/web/node_modules archive/root` | 0 | No tracked forbidden root generated paths remain. |

## Notes

- `.gitignore` uses root-anchored generated directory rules for `/outputs/`, `/artifacts/`, `/models/`, and `/tmp/` so source modules under `src/ml/models` remain trackable.
- `node_modules` rebuild evidence was captured with `npm.cmd` because PowerShell script execution blocks `npm.ps1` in this environment.
- The npm audit vulnerability report is dependency-policy evidence only; remediation is out of Phase 3 scope.
- `reports/results/RESEARCH_TOPIC_AND_CODE_AUDIT_REPORT.md` remains untracked and was not touched.
- Phase 0 and Phase 2 governed evidence files were not intentionally modified. The remediation checklist was intentionally updated for Phase 3 tracking.
