# Analysis Feed Schema
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Architecture note |
| Created / authored | Tuesday, 2026-04-28 22:34:05 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-04-28 22:34:05 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `8800ce6e3780c7978856737e70cb5e3b999eacee` |
| Timestamp source | current metadata standardization run |
| Status | Active |

This document defines the canonical entities and ID rules for the Analysis Feed Layer.

## ID Generation Rules

Stable IDs are required for cross-linking and deduplication.

| Entity | ID Format | Example |
| :--- | :--- | :--- |
| **Packet** | `pkt_{run_id}_{ticker}_{date}_h{horizon}_{target}_{mode}` | `pkt_a1b2c3d4_ACB_2026-01-01_h10_forward_return_full_forecast` |
| **Case** | `case_{packet_id}` | `case_pkt_a1b2c3d4_ACB_...` |
| **Memo** | `memo_{packet_id}` | `memo_pkt_a1b2c3d4_ACB_...` |

## Primary Entities

### 1. ForecastResearchPacket
The primary quantitative record.

| Field | Type | Description |
| :--- | :--- | :--- |
| `packet_id` | string | Unique stable ID. |
| `source_run_id` | string | ID of the source Quant Core run. |
| `primary_prediction` | float | Primary model prediction value. |
| `agreement_bucket` | string | `low`, `medium`, `high`. |
| `volatility_bucket` | string | `low`, `moderate`, `high`. |
| `regime_summary` | dict | Contains `regime_label` and probabilities. |
| `realized_outcome_label` | string | `gain`, `loss`, `flat` (if available). |

### 2. HistoricalCaseRecord
Optimized for RAG indexing and similarity research.

| Field | Type | Description |
| :--- | :--- | :--- |
| `case_id` | string | Unique stable ID. |
| `summary_text` | string | Deterministic, structured summary string. |
| `tags` | list[string] | Searchable tags (e.g., `regime:bull`, `volatility:high`). |

### 3. AnalystMemoDraft
Handoff object for human/LLM review.

| Field | Type | Description |
| :--- | :--- | :--- |
| `memo_id` | string | Unique stable ID. |
| `primary_signal_summary` | string | Concise deterministic signal description. |
| `suggested_action` | string | `Review for entry`, `Monitor`, etc. |
| `bullish_points` | list[string] | Placeholder for future workflows. |

### 4. RetrievalMetadata
Flat schema optimized for CSV/SQL filtering.

- Combines IDs, ticker data, and buckets into a single row for efficient indexed search.

## Field Provenance

Every entity includes:
- `schema_version`: Version of the Pydantic model.
- `generated_at`: ISO timestamp of generation.
- `source_manifest_path`: Path to the source quant-core manifest.
