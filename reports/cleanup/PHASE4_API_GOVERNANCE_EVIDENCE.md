# Phase 4 API Governance Evidence

Date: 2026-05-11
Scope: API governance enforcement only

## Summary

Phase 4 aligns active API metadata, response schemas, route payloads, static v2 demo outputs, and chat prompt boundaries with the diagnostic-only governance boundary.

The governed API now presents forecast, risk, scenario, and route diagnostics as research outputs. It does not expose public BUY/SELL labels, active account-routing payloads, or sizing instructions on the governed routes updated in this phase.

## Route Inventory

| Route | Status after Phase 4 |
| --- | --- |
| `/` | Diagnostic Research API metadata/root response with research-only, non-executing, non-advisory, no-account-routing boundaries. |
| `/api/v1/health` | Diagnostic research phase label. |
| `/api/v1/predict` | Active diagnostic payload with `diagnostic_signal`, `route_decision`, `decision_lane`, `risk_flag`, `review_required`, `diagnostic_summary`, `candidate_status`, and provenance. |
| `/api/v1/analyze` | Reuses normalized diagnostic payload and sanitizes qualitative summary text before public return. |
| `/api/v1/chat` | Prompt boundary states research diagnostics only, no financial advice, no BUY/SELL authority, no trade execution instructions, no broker/order authority. |
| `/api/v1/execute` | Deprecated legacy gate. Returns `LegacyRouteDiagnosticResponse`; no active account-routing payload. |
| `/api/v1/paper-trade` | Deprecated legacy/demo gate. Returns `LegacyRouteDiagnosticResponse`; no active account-routing payload. |
| `/api/v1/order-book` | Market-depth data route with `data_role=market_depth_only`; legacy path retained for compatibility. |
| `/predict/technical` | Demo static technical payload remains provenanced and audit-blocked. |
| `/predict/sentiment` | Demo static sentiment payload remains provenanced and audit-blocked. |
| `/predict/fused` | Demo diagnostic payload; no hardcoded BUY/SELL public value; provenance retained and audit-blocked. |
| `/debate` | Deprecated legacy gate. Returns `LegacyRouteDiagnosticResponse`; no active debate/card output. |

## Schema Inventory

| Schema | Change |
| --- | --- |
| `src/api/schemas_v2.py::FusionDecision` | Replaced `action`/`rationale` public fields with `diagnostic_signal`, `route_decision`, `decision_lane`, `diagnostic_summary`, and `review_required`. |
| `src/api/schemas_v2.py::RiskOverlay` | Replaced `position_size_suggestion`/`veto_flag` public fields with `allocation_candidate_weight`, `risk_flag`, `review_required`, and `risk_control_note`. |
| `src/api/schemas_v2.py::TerminalPayload` | Added `candidate_status`, `review_required`, `diagnostic_plan`, and `non_authoritative_summary`. |
| `src/api/schemas.py::PredictionResponse` | Reworked as a diagnostic prediction/analysis response rather than the old quantitative action-plan contract. |
| `src/api/schemas.py::LegacyRouteDiagnosticResponse` | Added for retained legacy/demo route gates. |
| `src/api/schemas.py` internal compatibility models | `ActionPlan`, `OrderPayload`, and related matrix/risk models remain only for existing non-public engine compatibility and tests; they are not used as governed API response models. |

## Terminology Mapping

| Before | After |
| --- | --- |
| BUY / EXECUTE_BUY | `upward_bias`, `monitor_upward_candidate` |
| SELL / EXECUTE_SELL | `downward_bias`, `monitor_downward_candidate` |
| STRONG_BUY / STRONG_SELL | `high_upward_bias`, `high_downward_bias` |
| recommendation | `diagnostic_summary` / `diagnostic note` |
| execution_decision | `route_decision` |
| action | `diagnostic_signal` |
| order_payload / final_order | legacy route gate with no account-routing payload |
| position_size_suggestion | `allocation_candidate_weight` |
| stop-loss / take-profit public instruction | internal range/risk references only; not public route instructions |
| paper trading route output | deprecated legacy diagnostic gate |
| debate decision card | deprecated legacy diagnostic gate |

## Files Changed

- `src/api/main.py`
- `src/api/routes.py`
- `src/api/routes_v2.py`
- `src/api/schemas.py`
- `src/api/schemas_v2.py`
- `tests/test_api.py`
- `reports/cleanup/AUDIT_REMEDIATION_CHECKLIST.md`
- `reports/cleanup/PHASE4_API_GOVERNANCE_EVIDENCE.md`

## Tests Added Or Updated

- Root metadata diagnostic-only assertions.
- Recursive forbidden authority term checks for governed API JSON payloads.
- v1 `/predict` diagnostic route field checks.
- v1 `/analyze` qualitative summary sanitization check.
- v1 `/chat` prompt-boundary check.
- v2 `/predict/fused` non-authoritative demo payload check.
- v2 audit-mode block check for fused demo output.
- Legacy gate checks for `/api/v1/execute`, `/api/v1/paper-trade`, and `/debate`.
- Existing runtime-mode provenance tests retained and passing.

## Verification

| Command | Result |
| --- | --- |
| `python -m pytest tests/test_api.py -q` | `26 passed, 3 warnings in 31.84s` |
| `python -m pytest tests -q` | `815 passed, 5 skipped, 33 warnings in 337.27s` |
| `python scripts/check_repo_hygiene.py` | `Repository hygiene check passed.` |

## Git Status

Pre-commit implementation status before report/checklist authoring:

```text
 M src/api/main.py
 M src/api/routes.py
 M src/api/routes_v2.py
 M src/api/schemas.py
 M src/api/schemas_v2.py
 M tests/test_api.py
?? reports/results/RESEARCH_TOPIC_AND_CODE_AUDIT_REPORT.md
```

Final post-commit status observed after the Phase 4 commit:

```text
?? reports/results/RESEARCH_TOPIC_AND_CODE_AUDIT_REPORT.md
```

## Unresolved Or Deferred Risks

- Internal engine/test compatibility models still carry legacy labels such as `ActionPlan.recommendation` and `OrderPayload`; these were not edited because engine/risk/model code was outside Phase 4 scope. They are not exposed as governed API response models after this phase.
- Non-API surfaces under `src/api/ui` still contain legacy UI terminology and were intentionally untouched because frontend removal work was out of scope.
- The retained legacy paths `/api/v1/execute`, `/api/v1/paper-trade`, and `/debate` remain as URLs for compatibility, but their governed outputs are gated diagnostic responses.
