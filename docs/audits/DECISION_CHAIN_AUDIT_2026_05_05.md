# Decision Chain Audit 2026-05-05
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Audit note |
| Created / authored | Tuesday, 2026-05-05 00:00:00 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-05-05 00:00:00 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Timestamp source | Local deterministic decision-chain documentation refactor |
| Status | Active |

## Verdict

Audit verdict before docs and CLI fixes:

`READY WITH FIXES`

The audit found that the deterministic decision-diagnostic chain was close to
ready for continued development, but stale documentation and CLI drift created
schema and authority-boundary risk.

## Fixes Implemented

- Router docs/code drift fixed.
- Router CLI dead flags removed.
- Quant Core governance chain updated.
- Canonical router artifacts clarified.
- Legacy router aliases made optional only.

Canonical Phase 3 Router v1 artifacts are:

- `router_decisions.csv`
- `router_summary.csv`
- `router_manifest.json`

Canonical Phase 3 Router v1 route decisions are:

- `route_allocation_candidate`
- `hold`
- `reject`
- `no_candidate`

## Current Implemented Chain

```text
Quant Core
-> Scenario Evaluation Engine v1
-> Risk Governance Layer v1
-> Decision Lane v2
-> Portfolio Allocator v1
-> Phase 3 Router v1
```

Each layer remains diagnostic-only.

## Remaining Validation

Full end-to-end smoke is still required if it has not been run locally:

```powershell
python scripts/run_quant_core.py --preset smoke --run-mode research_core --enable-scenario-engine --enable-risk-governance --enable-portfolio-allocator --enable-phase3-router --output-dir artifacts/quant_core_router_smoke
```

Generated smoke artifacts must not be committed.

## Authority Boundary

This audit does not grant BUY or SELL recommendation authority, live execution
authority, production trading readiness, or learned meta-model authority.
