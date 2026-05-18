# Phase 2 Mock Isolation and Provenance Evidence

Date: 2026-05-10
Branch: `audit-remediation-governed-runtime`
Scope: runtime mode enforcement, explicit mock/fallback isolation, provenance propagation, audit/research fail-fast behavior.

## Contract Implemented

Runtime modes are centralized in `src/core/runtime_mode.py`.

| Mode | Mock behavior |
| --- | --- |
| `demo` | Explicit mock and mock fallback allowed; outputs must declare provenance. |
| `research` | Explicit mock allowed only when requested; silent mock fallback raises. |
| `audit` | Explicit mock and mock fallback both raise. |

Additive provenance shape used on touched outputs:

```json
{
  "data_provenance": {
    "source": "synthetic_mock_data",
    "uses_mock_data": true,
    "fallback_triggered": false,
    "runtime_mode": "demo"
  }
}
```

Normal provider switching and cache fallback are not treated as mock unless synthetic data is generated.

## Files Changed

| File | Purpose |
| --- | --- |
| `src/core/runtime_mode.py` | Canonical runtime mode enum, mock policy, and provenance builder. |
| `src/ml/data_loader.py` | Mock data now carries explicit frame-level provenance. |
| `src/api/routes.py` | API v1 prediction/training/paper paths enforce runtime mode and propagate provenance. |
| `src/api/routes_v2.py` | Demo/static v2 outputs declare mock provenance and fail in audit mode. |
| `src/api/schemas.py` | Additive provenance fields on touched v1 response models. |
| `src/api/schemas_v2.py` | Additive provenance fields on touched v2 response models. |
| `src/ml/backtest/paper.py` | Paper-trading mock fallback is mode-gated and provenanced. |
| `src/ml/trainer.py` | Training results/manifests preserve input data provenance. |
| `tests/test_api.py` | API mode-gating and provenance assertions. |
| `tests/test_backtest.py` | Paper-trading mock fallback mode tests. |
| `tests/test_data_loader_vn100.py` | Mock frame provenance assertion. |
| `tests/ml/test_trainer_pipeline.py` | Trainer result/manifest provenance assertions. |

## Verification Evidence

Targeted tests:

| Command | Result |
| --- | --- |
| `python -m pytest tests/test_data_loader_vn100.py::TestBackwardCompatibility -q` | `4 passed, 1 warning in 0.35s` |
| `python -m pytest tests/test_backtest.py::TestPaperTrading -q` | `8 passed, 3 warnings in 2.57s` |
| `python -m pytest tests/test_api.py::TestRuntimeModeGovernance -q` | `6 passed, 3 warnings in 2.46s` |
| `python -m pytest tests/ml/test_trainer_pipeline.py::test_cart_training_and_manifest_report -q` | `1 passed, 1 warning in 1.51s` |
| `python -m pytest tests/test_api.py -q` | `17 passed, 3 warnings in 11.98s` |
| `python -m pytest tests/test_data_loader_vn100.py -q` | `41 passed, 1 warning in 10.67s` |
| `python -m pytest tests/ml/test_trainer_pipeline.py tests/test_backtest.py tests/test_api.py -q` | `33 passed, 3 warnings in 19.87s` |

Full suite:

| Command | Result |
| --- | --- |
| `python -m pytest tests -q` | `807 passed, 5 skipped, 33 warnings in 305.62s (0:05:05)` |

Warnings remaining:

- `vnstock` / `vnai` package upgrade notices.
- LightGBM feature-name warnings in existing model tests.
- `loky` physical-core detection warning.
- Existing class-coverage and max-iteration warnings.
- Pytest cache warning: `.pytest_cache` write denied by local filesystem permissions.

## Governance Checks

| Check | Result |
| --- | --- |
| Mock usage explicit | Verified by API, data-loader, and paper-trading tests. |
| Audit mode blocks mock | Verified for API v1, API v2 demo endpoint, and paper OHLCV loader. |
| Research mode blocks silent mock fallback | Verified for API v1 and paper OHLCV loader. |
| Research explicit mock allowed | Verified by governed OHLCV loader test. |
| Demo fallback provenance visible | Verified by governed OHLCV loader and paper-trading tests. |
| Provider switching treated as non-mock | Preserved; DB-to-vnstock and cache fallbacks are not flagged as mock. |
| Phase 0 audit report unchanged | Verified with `git diff --name-only -- reports/cleanup/CODE_AUDIT_REPORT.md reports/cleanup/CODE_AUDIT_REMEDIATION_PLAN.md`. |
| Out-of-scope research report untouched | `reports/results/RESEARCH_TOPIC_AND_CODE_AUDIT_REPORT.md` remains untracked and was not edited. |

## Unresolved Risks

- API authority semantics, BUY/SELL wording, allocator behavior, VN100 governance, feature catalogue governance, and risk/regime manifest hardening remain out of scope for this phase.
- `routes_v2.py` demo/static prediction outputs are isolated and provenanced, not redesigned.
- `routes_v2.py::run_debate` still contains mock portfolio sizing and is deferred because allocator/API authority semantics are explicitly out of scope.

