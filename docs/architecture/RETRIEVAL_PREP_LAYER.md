# RAG Preparation Layer
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

The **RAG Preparation Layer** is the final data normalization stage before ingestion into vector databases or RAG frameworks. It sits on top of the **Analysis Feed Layer**.

## Positioning

```mermaid
graph TD
    QC[Quant Core] -->|Artifacts| AFL[Analysis Feed Layer]
    AFL -->|Feed| RPL[RAG Preparation Layer]
    RPL -->|Documents| VDB[Vector DB / RAG Engine]
```

## Retrieval Pipeline

1.  **Selection**: Classifies feed artifacts into primary and secondary retrieval sources.
2.  **Rendering**: Deteministically turns structured data into narrativized synopses for embedding.
3.  **Metadata Extraction**: Extracts canonical filter fields (ticker, regime, etc.) for hybrid search.
4.  **Provenance**: Preserves the full chain from Quant Core commit to retrieval document.

## Canonical Sources

| Source | Role | Description |
| :--- | :--- | :--- |
| **Historical Cases** | Primary | Narrative research cases. Best for broad semantic lookup. |
| **Research Packets** | Secondary | Technical quant data. Best for deep evidence retrieval. |

## Implementation Details

- **Schema**: [src/core/retrieval_schema.py](../../src/core/retrieval_schema.py)
- **Indexing Strategy**: [RETRIEVAL_INDEXING_STRATEGY.md](RETRIEVAL_INDEXING_STRATEGY.md)
- **Runner**: [scripts/run_retrieval_prep.py](../../scripts/run_retrieval_prep.py)

## Execution

```bash
python scripts/run_retrieval_prep.py \
  --mode build_from_existing_analysis_feed \
  --analysis-feed-dir artifacts/analysis_feed \
  --output-dir artifacts/retrieval_prep
```
