"""Runner for the Retrieval Ingestion Layer."""

from __future__ import annotations

import argparse
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from src.retrieval.file_adapter import FileRetrievalAdapter
from src.retrieval.qdrant_adapter import QdrantRetrievalAdapter
from src.retrieval.types import RetrievalIndexRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Retrieval Ingestion Layer.")
    parser.add_argument("--mode", choices=["smoke", "build_from_existing_retrieval_prep"], required=True)
    parser.add_argument("--backend", choices=["file", "qdrant"], default="file")
    parser.add_argument("--retrieval-prep-dir", type=str, required=True, help="Path to retrieval-prep artifact directory")
    parser.add_argument("--output-dir", default="artifacts/retrieval_ingest", help="Where to write ingest artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prep_dir = Path(args.retrieval_prep_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Selection
    print(f"Loading retrieval-prep artifacts from {prep_dir}...")
    manifest_path = prep_dir / "retrieval_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        prep_manifest = json.load(f)
    
    # Init Backend
    if args.backend == "file":
        adapter = FileRetrievalAdapter(output_dir)
    else:
        adapter = QdrantRetrievalAdapter()
    
    print(f"Using backend: {args.backend}")
    print(f"Backend Health: {adapter.health_check()}")
    
    # 2. Load documents and filters
    doc_file = prep_dir / prep_manifest["artifact_paths"]["retrieval_documents"]
    filter_file = prep_dir / prep_manifest["artifact_paths"]["retrieval_filter_metadata"]
    
    print(f"Reading filters from {filter_file}...")
    filters_df = pd.read_csv(filter_file)
    filters_map = filters_df.set_index("retrieval_doc_id").to_dict(orient="index")

    records: List[RetrievalIndexRecord] = []
    
    print(f"Ingesting documents from {doc_file}...")
    with open(doc_file, "r", encoding="utf-8") as f:
        count = 0
        for line in f:
            doc_raw = json.loads(line)
            doc_id = doc_raw["retrieval_doc_id"]
            
            # Enrich with filters
            meta = filters_map.get(doc_id, {})
            
            record = RetrievalIndexRecord(
                retrieval_doc_id=doc_id,
                source_type=doc_raw["source_type"],
                text=doc_raw["text"],
                summary=doc_raw["summary"],
                metadata=meta,
                source_run_id=prep_manifest["source_run_id"],
                source_manifest_path=prep_manifest["source_manifest_path"],
                source_artifact_dir=prep_manifest["source_artifact_dir"]
            )
            records.append(record)
            count += 1
            if args.mode == "smoke" and count >= 10:
                break

    # 3. Execution
    print(f"Submitting {len(records)} documents for ingestion...")
    result = adapter.ingest_documents(records)
    print(f"Ingestion result: {result}")
    
    # 4. Required Outputs
    ingest_id = f"ingest_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    manifest = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ingest_id": ingest_id,
        "backend": args.backend,
        "source_retrieval_prep_dir": str(prep_dir),
        "document_count": len(records),
        "status": "complete",
        "backend_health": adapter.health_check()
    }
    
    with open(output_dir / "ingest_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    if args.backend == "file":
        local_manifest = {
            "backend": "local_file",
            "store_path": str(output_dir / "local_index_store.jsonl"),
            "document_count": len(records),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        with open(output_dir / "local_index_manifest.json", "w", encoding="utf-8") as f:
            json.dump(local_manifest, f, indent=2)

    with open(output_dir / "backend_status.json", "w", encoding="utf-8") as f:
        json.dump(adapter.health_check(), f, indent=2)
        
    # Summary
    with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"""# Retrieval Ingestion Summary

- **Ingest ID**: `{ingest_id}`
- **Backend**: `{args.backend}`
- **Documents Ingested**: {len(records)}
- **Source Run ID**: `{prep_manifest['source_run_id']}`

## Validation Status
- **Backend Health**: `{adapter.health_check()['status']}`
""")

    # Indexed Summary CSV
    summary_df = pd.DataFrame([
        {"retrieval_doc_id": r.retrieval_doc_id, "source_type": r.source_type} for r in records
    ])
    summary_df.to_csv(output_dir / "indexed_document_summary.csv", index=False)

    print(f"Retrieval Ingestion Complete. Manifest: {output_dir}/ingest_manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
