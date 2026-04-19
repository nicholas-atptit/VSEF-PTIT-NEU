# Retrieval Backend strategy

This document outlines the pluggable backend strategy for the Quant Core retrieval ecosystem.

## Backend Selection Criteria

| Backend | status | Use Case |
| :--- | :--- | :--- |
| **Local File** | Validated | Developer workstations, CI/CD, deterministic smoke testing. |
| **Qdrant** | Scaffold | Large-scale semantic search, distributed production workloads. |
| **pgvector** | Future | Relational-heavy retrieval where stock metadata and vectors reside in Postgres. |

## Local Adapter (Reference)

The `FileRetrievalAdapter` serves as the ground truth for adapter contract compliance. It guarantees that the integration between `retrieval_prep` and the storage layer is operational.

## Qdrant Scaffold

The `QdrantRetrievalAdapter` in `src/retrieval/qdrant_adapter.py` is currently a **scaffold**.
- **Reason**: `qdrant-client` is missing from the current environment.
- **Path to Activation**:
  1. Install `qdrant-client`.
  2. Implement the `ingest_documents` batch logic (provided in the scaffold comments).
  3. Define the vector collection parameters (e.g., Distance Metric: Cosine, Dimension: 1536).

## Embedding Provider Strategy

Embedding generation is abstracted via the `EmbeddingProvider` interface in `src/retrieval/embedding_prep.py`.
- **Current**: Mock provider (deterministic placeholder).
- **Future**: SentenceTransformers (Local) or OpenAI/Azure (Remote).

## Provenance Chain

One of the most critical requirements is preserving the provenance chain during ingestion. The adapter ensures that every document in the store maintains a link to `source_run_id` (the specific Git commit of the quant-core run). This enables a user to trace any search result back to the model code that produced the evidence.
