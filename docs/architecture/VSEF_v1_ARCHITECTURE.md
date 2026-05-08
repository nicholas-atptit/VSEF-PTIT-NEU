# VSEF v1 Architecture Freeze

## Document Metadata

| Field | Value |
| --- | --- |
| Document name | VSEF v1 Architecture Freeze |
| Phase | 0 |
| Status | Frozen for v1 |
| Last updated date | 2026-05-09 |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Owner | Project team / maintainer |
| Document type | Architecture governance |

## Purpose

VSEF v1 is a governed stock forecasting and analysis system. Its frozen scope is
focused on daily OHLCV data ingestion, feature engineering, multiple forecasting
models, forecast comparison, model consensus, risk summary, strategy diagnostics
where already supported, and governed artifact outputs.

This document freezes the VSEF v1 architecture for review, implementation
discipline, and supervisor defense. It does not introduce new runtime behavior.
If a repository file, script, or older document suggests functionality beyond
this frozen scope, that functionality must be classified as excluded, deferred to
v1.5/v2, or legacy/archive candidate before it can be represented as part of
VSEF v1.

## Frozen v1 Architecture Layers

VSEF v1 contains only these architecture layers:

1. Data Layer
2. Feature Engineering Layer
3. Forecasting Model Layer
4. Ensemble / Consensus Layer
5. Evaluation Layer
6. Risk Summary Layer
7. Strategy Diagnostics Layer
8. Artifact / Manifest Layer
9. Documentation / Governance Layer

No new agents, APIs, dashboards, portfolio engines, broker integrations, or
autonomous decision layers are part of the frozen VSEF v1 architecture.

## Component Table

| Layer | Component | v1 Status | Evidence / expected artifacts | Notes |
| --- | --- | --- | --- | --- |
| Data Layer | `vnstock_data` daily OHLCV input | Included | `fetch_summary.csv`, source provenance, data ingestion logs where available | Official v1 provider is `vnstock_data`; other providers are outside v1 unless separately governed. |
| Data Layer | Standard OHLCV schema | Included | Data tables or CSVs with `date`, `ticker`, `open`, `high`, `low`, `close`, `volume` | Schema is governed in `docs/governance/DATA_POLICY.md`. |
| Feature Engineering Layer | Daily feature generation from OHLCV | Included | `training_summary.csv`, feature pipeline logs, model manifests | Feature evidence should be verified before claiming production readiness. |
| Forecasting Model Layer | SARIMAX | Included | Forecast rows, model execution logs, model manifests | Supported v1 model; dependency availability must be recorded if a run skips it. |
| Forecasting Model Layer | ETS | Included | Forecast rows, model execution logs, model manifests | Supported v1 model; dependency availability must be recorded if a run skips it. |
| Forecasting Model Layer | XGBoost | Included | Forecast rows, model execution logs, model manifests | Supported v1 model; optional package availability must be recorded. |
| Forecasting Model Layer | LightGBM | Included | Forecast rows, model execution logs, model manifests | Supported v1 model; optional package availability must be recorded. |
| Forecasting Model Layer | LSTM | Included | Forecast rows, training summary, model artifacts, model manifests | Supported v1 sequence model. Implementation evidence should be verified before claiming production readiness. |
| Forecasting Model Layer | BiLSTM | Included | Forecast rows, training summary, model artifacts, model manifests | Supported v1 sequence model. Implementation evidence should be verified before claiming production readiness. |
| Ensemble / Consensus Layer | Stacking | Included | Stacking forecast outputs, stacking comparison artifacts, model manifests | Stacking is an ensemble layer over model outputs, not independent investment authority. |
| Ensemble / Consensus Layer | Model consensus | Included | `model_consensus_summary.csv` | Consensus is diagnostic evidence for comparison and review. |
| Evaluation Layer | Time-series-safe evaluation | Included | `forecast_summary.csv`, evaluation summaries, model manifests | Random train-test split is not acceptable for time-series forecasting claims. |
| Evaluation Layer | Walk-forward evaluation | Included | Walk-forward run artifacts, `run_config.json`, manifests | Preferred validation method where available. |
| Risk Summary Layer | Risk summary diagnostics | Included | `risk_summary.csv` | Risk output is diagnostic only and does not authorize trades. |
| Strategy Diagnostics Layer | Existing strategy metrics diagnostics | Included where already supported | `strategy_metrics.csv` | No new strategy engine is introduced by Phase 0. |
| Artifact / Manifest Layer | Forecast and model artifacts | Included | `forecast_summary.csv`, `model_health_summary.csv`, model manifests | Claims must be backed by artifact evidence. |
| Artifact / Manifest Layer | Analysis packets | Included | `analysis_packets.jsonl` | Packets are auditable diagnostic context. |
| Artifact / Manifest Layer | Decision-lane candidates | Included as diagnostic candidates only | `decision_lane_candidates.csv` | Candidates are for review, ranking, explanation, and governance only. |
| Documentation / Governance Layer | Phase 0 governance documents | Included | Architecture, model registry, data policy, evaluation protocol, project tracker | These documents freeze v1 scope and route future ideas into backlog. |

## Explicit v1 Inclusions

The frozen VSEF v1 scope includes:

- `vnstock_data` daily OHLCV input
- Standard OHLCV schema
- SARIMAX
- ETS
- XGBoost
- LightGBM
- LSTM
- BiLSTM
- Stacking
- Time-series-safe evaluation
- Walk-forward evaluation
- Forecast summary artifacts
- Model consensus artifacts
- Model health diagnostics
- Risk summary artifacts
- Strategy metrics artifacts
- Analysis packets
- Decision-lane candidates as diagnostic candidates only

## Explicit v1 Exclusions

VSEF v1 does not include:

- Real BUY / SELL / HOLD investment advice
- Fully implemented portfolio allocator
- Capital allocation engine
- Broker execution
- Live trading
- Reinforcement learning
- Real-time intraday trading
- Alternative data expansion beyond current documented scope
- LLM-based autonomous decision-making
- New agents outside the frozen architecture
- Multi-asset institutional portfolio optimization

Existing repository surfaces that appear to go beyond this list must not be
presented as VSEF v1 production scope. They should be handled as follows unless
a future governance change explicitly revises the freeze:

| Existing surface | Phase 0 classification | Governance note |
| --- | --- | --- |
| Portfolio allocator code or documentation | Deferred to v1.5 / next implementation phase | Not part of Phase 0 and not part of frozen v1 investment authority. |
| Phase 3 router code or documentation | Deferred to v1.5/v2 diagnostic chain review | May remain as historical or experimental diagnostic work, but not frozen v1 scope. |
| Autonomous agent or LLM decision scripts | Excluded from v1 | LLMs may not make autonomous trading decisions in v1. |
| Intraday or hourly data services | Excluded from v1 | v1 uses daily OHLCV data only. |
| Alternative data expansion beyond `vnstock_data` daily OHLCV | Deferred to v1.5/v2 | Must be governed before inclusion. |
| Non-registry model experiments outside the supported v1 list | Legacy/archive candidate or v1.5/v2 backlog | Do not claim as supported v1 models. |

## Decision-Lane Clarification

Decision-lane candidates are diagnostic candidates. They are not investment
recommendations, not order instructions, and not proof of profitable trades.

In VSEF v1, decision-lane candidates may be used for review, ranking,
explanation, audit traceability, and governance. They must not be presented as
guaranteed profitable trades or as real BUY / SELL / HOLD advice.

## Architecture Explanation in 2 Minutes

VSEF v1 is a governed daily stock forecasting framework. It starts with
`vnstock_data` daily OHLCV data, validates the standard schema, builds features,
and runs the frozen supported model set: SARIMAX, ETS, XGBoost, LightGBM, LSTM,
BiLSTM, and a stacking ensemble layer. Forecasts are evaluated with
time-series-safe methods, preferably walk-forward validation where available, so
future data is not leaked into training or model selection. The system compares
models, creates consensus and model-health diagnostics, records risk summaries
and existing strategy metrics where available, and writes auditable artifacts
such as forecast summaries, model consensus summaries, manifests, analysis
packets, and decision-lane candidate files. The decision lane is diagnostic
only: it supports review and explanation, but it does not give investment
advice, allocate capital, execute trades, or authorize autonomous decisions.

## Governance Change Request Rule

Any change to frozen VSEF v1 scope must be handled as a governance change
request before it is represented as accepted v1 scope. The request must document:

- Proposed change
- Reason
- Affected documents
- Affected runtime surfaces
- Evidence required
- Approval status

Unapproved changes remain excluded, deferred, legacy/archive candidates, or
out of scope.

## Acceptance Criteria

- [x] Architecture layers are frozen.
- [x] v1 included components are listed.
- [x] v1 excluded components are listed.
- [x] Decision-lane meaning is clarified.
- [x] The document can be used in a presentation or supervisor defense.

## Related Governance Documents

- [VSEF v1 Model Registry](../governance/MODEL_REGISTRY.md)
- [VSEF v1 Data Policy](../governance/DATA_POLICY.md)
- [VSEF v1 Evaluation Protocol](../governance/EVALUATION_PROTOCOL.md)
- [VSEF v1 Project Tracker](../governance/PROJECT_TRACKER.md)
