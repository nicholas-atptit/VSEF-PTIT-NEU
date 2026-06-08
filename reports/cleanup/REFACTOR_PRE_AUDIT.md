# Repository Refactor Pre-Audit

Date: 2026-06-08

Branch: `refactor/repository-code-cleanup-v1`

## Scope and constraints

This audit precedes implementation edits. The refactor is limited to reusable,
offline research infrastructure and documentation alignment. Existing generated
evidence, raw/cache data, paper packages, result wording, and research outputs
are protected. No model training, live provider call, benchmark rerun, result
rewrite, or deprecated-file deletion is authorized by this audit.

## Current structure

- `src/` contains reusable production-era and research-era modules, but reusable
  forecast research logic remains split across `src/ml/`, `src/evaluation/`,
  `src/forecast/`, and very large scripts.
- `scripts/research/` contains more than 100 one-off audit, fetch, validation,
  benchmark, and diagnostic runners. These should remain orchestration entry
  points unless extraction can preserve their output contracts.
- `scripts/research/run_vn30_model_universe_direction_price_benchmark.py` exists
  and is 4,167 lines.
- `scripts/research/run_vn30_qml_forecasting.py` exists and is 6,427 lines.
- `scripts/research/run_vn_forecast_engine_v1.py` is missing.
- `scripts/research/run_vn30_index_group_price_range_forecast_lab.py` is missing.
- `configs/` already contains feature, policy, experiment, universe, and
  universes directories. A second root `config/` also exists and remains a
  documented structural debt because moving it would create broad import risk.
- `docs/` already contains architecture, governance, and usage material but its
  primary structure/runbook documents predate the VN Forecast Research Lab
  identity and required claim boundary.

## Duplicated and messy logic

- JSON, CSV, Markdown writing, finite-float conversion, repository-relative
  paths, and directory creation are repeated in QML, model-universe, and joint
  panel scripts.
- Strict train/validation/final timestamp constants and split checks are repeated
  across QML and daily/hourly research scripts.
- Direction metrics are implemented in `src/ml/metrics.py`, canonical-eval
  scripts, QML scripts, model-universe scripts, and regime scripts with different
  output names and incomplete repaired metrics.
- Return/price metrics are split between `src/ml/metrics.py`,
  `src/evaluation/walkforward.py`, trainers, and research runners.
- Ranking metrics and interval/range metrics lack a canonical reusable research
  API.
- Baseline prediction logic is repeated in the model-universe and joint-panel
  scripts; range baselines lack a canonical module.
- Feature-building logic is repeated in
  `scripts/research/vn30_stock_index_joint_panel_features.py`, QML helpers, and
  multiple hourly common modules.
- The model-universe runner imports many functions directly from the QML runner,
  creating fragile runner-to-runner coupling.

## Hard-coded and fragile paths/imports

- Large research runners bootstrap `sys.path` using
  `Path(__file__).resolve().parents[2]`.
- Research outputs are hard-coded as module constants under `reports/generated`,
  `reports/results`, and `reports/claims`.
- Several reusable modules use working-directory-relative `data/` and `reports/`
  paths.
- Existing scripts import other scripts as libraries. Replacing all such imports
  in this pass would be high risk because the scripts write protected evidence.

## One-off runners to retain

- QML V1-V8 diagnostics and paper-evidence runners.
- Model Universe V1-V7 benchmark/audit runners.
- Regime transferability, legacy rules, and targeted diagnostic runners.
- Provider probe/fetch/refetch scripts. They remain available but are not run by
  this refactor.
- Paper figure/table builders and project-review evidence builders.

## Logic to extract

- Strict feature/target timestamp split policy.
- Claim-boundary wording and labels.
- Protected artifact/evidence registry and deletion guards.
- Canonical direction, return/price, interval/range, and ranking metrics.
- Direction, return/price, and range baselines.
- Point-in-time-safe common feature builders.
- Forecast panel schema/validation.
- Validation-only model selection helpers.
- Repository path, serialization, timestamp, and artifact-manifest utilities.

## Missing tests and docs

- No focused tests currently cover strict feature-and-target split discipline.
- No focused tests cover the requested canonical interval and ranking metrics.
- No focused tests cover forecast-panel schema or validation-only selectors.
- No canonical runbook documents offline-only execution and missing runner
  status.
- Claim and artifact policies exist mainly as prose/evidence files rather than
  reusable code.

## Preservation boundary

The following are active/protected and must not be deleted or rewritten by this
refactor: `reports/project_review/`, `reports/paper/qml_kernel_feature_vn30/`,
`reports/generated/vn30_qml_forecasting/`,
`reports/generated/vn30_model_universe_direction_price/`,
`reports/generated/vn30_index_group_range_forecast/`,
`reports/generated/vn_forecast_engine_v1/`, matching result/claim/protocol
families, `data/`, `outputs/`, `archive/`, `paper_evidence_export/`, and
`paper_evidence_raw_full_export/`.

The active evidence index at `reports/_index/ACTIVE_EVIDENCE_INDEX.md`, evidence
matrices, and claim registers are also protected.

## Refactor decision

Create additive reusable modules and focused tests. Keep large runners in place
and compile-check them. Do not move files in this pass because their path and
import contracts are embedded in protected reports and evidence. Any safe import
adoption will be limited to utility calls that cannot alter results; remaining
runner coupling will be documented as debt.
