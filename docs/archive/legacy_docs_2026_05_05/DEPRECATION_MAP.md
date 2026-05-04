# Legacy Docs 2026-05-05 Deprecation Map
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Deprecation map |
| Created / authored | Tuesday, 2026-05-05 00:00:00 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-05-05 00:00:00 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Timestamp source | Local documentation inventory and legacy migration |
| Status | Active |

## Rule

Active canonical docs override archived docs. Archived docs are retained for
historical context only.

## Moved Files

| Old path | New archive path | Replacement source-of-truth doc | Reason |
| --- | --- | --- | --- |
| `docs/governance/QUANT_CORE_SURFACE_AUDIT.md` | `docs/archive/legacy_docs_2026_05_05/QUANT_CORE_SURFACE_AUDIT.md` | `docs/SYSTEM_OVERVIEW.md`; `docs/DECISION_DIAGNOSTIC_CHAIN.md`; `docs/governance/PIPELINE_CONTRACTS.md`; `docs/governance/QUANT_CORE_GOVERNANCE.md`; `docs/governance/QUANT_CORE_OUTPUT_SCHEMA.md` | Historical surface audit says Phase 3 routing/meta-model logic was out of scope for its phase. That is obsolete for the current implemented deterministic diagnostic chain, which includes Portfolio Allocator v1 and Phase 3 Router v1. |

## Files Retained In Place With Explicit Legacy Sections

| Path | Replacement source-of-truth doc | Reason |
| --- | --- | --- |
| `docs/governance/PHASE3_ROUTER_V1.md` | `docs/governance/PHASE3_ROUTER_OUTPUT_SCHEMA.md` | Retained as an active reference because it explicitly marks old router filenames and route labels as legacy/non-canonical. |
| `docs/governance/PHASE3_ROUTER_OUTPUT_SCHEMA.md` | Itself | Source-of-truth schema. Mentions old router filenames only in the explicit legacy-alias policy. |

## Authority Boundary

Deprecation mapping does not grant BUY or SELL recommendation authority, live
execution authority, production trading readiness, or learned meta-model
authority.
