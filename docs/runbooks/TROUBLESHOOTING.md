# Troubleshooting
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Runbook |
| Created / authored | Tuesday, 2026-05-05 00:00:00 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-05-05 00:00:00 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Timestamp source | Local deterministic decision-chain documentation refactor |
| Status | Active |

## PowerShell Line Continuation

PowerShell uses the backtick character for line continuation:

```powershell
python scripts/run_quant_core.py `
  --preset smoke `
  --run-mode research_core
```

CMD uses `^`, but PowerShell does not. If a command fails unexpectedly after a
line break, rerun it as one line or replace `^` with PowerShell backticks.

## MLE Convergence Warnings

MLE convergence warnings from statistical models are warnings unless a Python
`Traceback` appears. Treat them as diagnostic context and inspect
`model_execution_log.csv`, `skipped_models`, and model-health outputs before
changing code.

## Pytest Cache Permission Warnings

`.pytest_cache` permission warnings are non-fatal when the test summary reports
passing tests. They mean pytest could not update its local cache path; they do
not mean the diagnostic chain failed.

## Generated Artifacts

Do not commit generated artifacts.

Keep local outputs under ignored paths such as:

- `artifacts/`
- `outputs/`
- `tmp/`

Before committing, check:

```powershell
git status --short
```

Do not stage generated smoke outputs or unrelated report artifacts.

## Authority Boundary

If an output or log appears to imply BUY, SELL, live execution, production
trading readiness, or learned meta-model authority, treat that as documentation
or legacy-language drift unless the active governance docs explicitly say
otherwise. The current deterministic decision chain is diagnostic-only.
