# Decision Physics Next Steps
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Roadmap note |
| Created / authored | Tuesday, 2026-05-05 00:00:00 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-05-05 00:00:00 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Timestamp source | Local deterministic decision-chain documentation refactor |
| Status | Active |

## Boundary

Decision Physics work must preserve the current diagnostic-only chain:

```text
Quant Core
-> Scenario Evaluation Engine v1
-> Risk Governance Layer v1
-> Decision Lane v2
-> Portfolio Allocator v1
-> Phase 3 Router v1
```

Do not add BUY or SELL recommendation authority, live execution authority,
production trading readiness claims, or learned meta-router behavior without a
separate governed implementation and validation plan.

## Next Work

1. Run full end-to-end smoke.
2. Add generated-artifact integration tests.
3. Define authoritative downstream fields.
4. Strengthen scenario calibration provenance.
5. Define Social Listening risk-only integration.
6. Run medium and `decision_core` validation.
7. Defer any learned meta-router until enough validated out-of-sample history
   exists.

## Notes

Generated-artifact integration tests should verify filenames, manifest paths,
required fields, allowed decision values, diagnostic-only authority fields, and
absence of legacy aliases from canonical manifests.

Authoritative downstream fields should distinguish:

- packet identity fields
- context join keys
- diagnostic candidate fields
- allocation fields
- route decision fields
- authority-boundary flags

Scenario calibration provenance should record calibration lookback, binning,
realized outcome availability, missing calibration share, and degradation
status.
