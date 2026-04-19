# Retrieval and Indexing Strategy

This document details the chunking, rendering, and filtering policies used to prepare the Quant Core for RAG.

## Chunking Policy

We adopt a **no-chunking** default for current structured records.

| Entity Type | Policy | Reason |
| :--- | :--- | :--- |
| **HistoricalCaseRecord** | No-chunk | Already atomic and meaningful. |
| **ForecastResearchPacket** | No-chunk | Breaking structured records reduces retrieval context. |
| **AnalystMemoDraft** | Conditional | Reserved for potential long-form narrative expansion. |

## Text Rendering Policy

Retrieval text is rendered deterministically using narrativized templates. This transforms structured machine data into human-style context for encoders (embeddings).

- **Cases**: Title + Summary + Narrative (including regime and volatility) + Tags.
- **Packets**: Title + Prediction Summary + Model Agreement Context + Environmental Synopsis.

**Rules**:
- NO raw JSON dumps in the retrieval body.
- Key numeric signals must be formatted (e.g., `.4f`).
- Threshold statuses must be explicit (e.g., "threshold passed").

## Filtering Strategy

We maintain a flat CSV sidecar, `retrieval_filter_metadata.csv`, which enables hybrid search patterns.

### Canonical Filter Fields

- `ticker`, `ticker_group`
- `regime_label`, `volatility_bucket`, `agreement_bucket`
- `horizon`, `target_type`
- `run_mode`, `cost_mode`
- `realized_outcome_label`

## Provenance and ID Stability

- **Retrieval IDs**: `rdoc_{source_type}_{source_id}`.
- **Provenance Chain**: Every document carries the `source_run_id` and `source_manifest_path`, ensuring every bit of retrieved text can be audited back to the exact Git commit used for execution.
