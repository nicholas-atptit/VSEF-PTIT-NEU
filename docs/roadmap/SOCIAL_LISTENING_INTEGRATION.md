# Social Listening Integration
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Roadmap note |
| Created / authored | Tuesday, 2026-05-05 00:00:00 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-05-05 00:00:00 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Timestamp source | Local deterministic decision-chain documentation refactor |
| Status | Proposed |

## Current Status

Social Listening is not implemented in the active deterministic
decision-diagnostic chain.

The active chain remains:

```text
Quant Core
-> Scenario Evaluation Engine v1
-> Risk Governance Layer v1
-> Decision Lane v2
-> Portfolio Allocator v1
-> Phase 3 Router v1
```

## Future Boundary

Future Social Listening integration must be risk-only or uncertainty-only. It
may add cautionary context to Risk Governance, Scenario Evaluation, or a future
uncertainty layer. It must not create candidates directly and must not increase
confidence directly.

## Allowed Social Listening Signals

Future Social Listening may detect:

- sentiment anomaly
- polarity imbalance
- narrative concentration
- source quality risk
- velocity spike
- conflict with Quant

These signals should be represented as risk or uncertainty features with
provenance, timestamps, source quality metadata, and conservative missing-data
behavior.

## Disallowed Social Listening Behavior

Future Social Listening must not:

- create BUY or SELL recommendations
- create candidates directly
- increase confidence directly
- override Risk Governance to make a candidate more favorable
- route or allocate by itself
- claim production trading readiness

## Candidate Integration Pattern

If implemented later, Social Listening should enter the active chain as one or
more diagnostic fields such as:

- `sentiment_anomaly_score`
- `polarity_imbalance_score`
- `narrative_concentration_score`
- `source_quality_risk_score`
- `velocity_spike_score`
- `quant_social_conflict_flag`

Those fields should feed only risk or uncertainty components. They should be
bounded, deterministic, auditable, and treated as missing or neutral when source
coverage is insufficient.
