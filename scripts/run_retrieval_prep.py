"""Runner for the RAG Preparation Layer."""

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
from src.reporting.retrieval_prep import (
    load_analysis_feed_manifest,
    render_case_document,
    render_packet_document,
    generate_retrieval_filter_metadata,
    write_retrieval_prep_outputs
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RAG Preparation Layer.")
    parser.add_argument("--mode", choices=["smoke", "build_from_existing_analysis_feed"], required=True)
    parser.add_argument("--analysis-feed-dir", type=str, required=True, help="Path to validated analysis-feed artifact directory")
    parser.add_argument("--output-dir", default="artifacts/retrieval_prep", help="Where to write retrieval artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feed_dir = Path(args.analysis_feed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading analysis-feed manifest from {feed_dir}...")
    manifest_src = load_analysis_feed_manifest(feed_dir)
    
    # Common provenance fields
    provenance = {
        "source_run_id": manifest_src["source_run_id"],
        "source_manifest_path": manifest_src["source_manifest_path"],
        "source_artifact_dir": manifest_src["source_artifact_dir"]
    }
    
    docs: List[Any] = []
    chunks: List[Any] = []
    filters: List[Any] = []
    
    # 1. Process Historical Case Records (Primary)
    case_file = feed_dir / manifest_src["artifact_paths"]["historical_case_records"]
    print(f"Processing cases from {case_file}...")
    with open(case_file, "r", encoding="utf-8") as f:
        count = 0
        for line in f:
            raw = json.loads(line)
            doc = render_case_document(raw, provenance)
            docs.append(doc)
            filters.append(generate_retrieval_filter_metadata(doc, raw))
            count += 1
            if args.mode == "smoke" and count >= 5:
                break
    primary_count = count
    
    # 2. Process Forecast Research Packets (Secondary)
    pkt_file = feed_dir / manifest_src["artifact_paths"]["forecast_research_packets"]
    print(f"Processing packets from {pkt_file}...")
    with open(pkt_file, "r", encoding="utf-8") as f:
        count = 0
        for line in f:
            raw = json.loads(line)
            doc = render_packet_document(raw, provenance)
            docs.append(doc)
            filters.append(generate_retrieval_filter_metadata(doc, raw))
            count += 1
            if args.mode == "smoke" and count >= 5:
                break
    secondary_count = count

    print(f"Total retrieval documents rendered: {len(docs)} ({primary_count} primary, {secondary_count} secondary)")
    
    # Chunking policy: none for structured records in this phase
    print("Chunking policy: no-chunk (atomic records only).")
    
    print(f"Writing outputs to {output_dir}...")
    artifact_paths = write_retrieval_prep_outputs(output_dir, docs, chunks, filters)
    
    # Final Manifest
    retrieval_id = f"retrieval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    retrieval_manifest = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_run_id": provenance["source_run_id"],
        "source_manifest_path": provenance["source_manifest_path"],
        "source_artifact_dir": provenance["source_artifact_dir"],
        "retrieval_id": retrieval_id,
        "source_analysis_feed_manifest": str(feed_dir / "feed_manifest.json"),
        "document_count": len(docs),
        "chunk_count": len(chunks),
        "primary_source_count": primary_count,
        "secondary_source_count": secondary_count,
        "artifact_paths": artifact_paths
    }
    
    with open(output_dir / "retrieval_manifest.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_manifest, f, indent=2)
        
    # Summary Markdown
    with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"""# Retrieval Preparation Summary
        
- **Retrieval ID**: `{retrieval_id}`
- **Source Feed**: `{retrieval_manifest['source_analysis_feed_manifest']}`
- **Total Documents**: {len(docs)}
- **Primary (Cases)**: {primary_count}
- **Secondary (Packets)**: {secondary_count}
- **Chunking**: Inactive (atomic records)

## Artifacts
{chr(10).join([f"- **{k}**: `{v}`" for k, v in artifact_paths.items()])}
""")
        
    # Schema Summary
    summary_data = [
        {"entity": "RetrievalDocument", "count": len(docs), "description": "Narrativized retrieval units"},
        {"entity": "RetrievalChunk", "count": len(chunks), "description": "Support for future segmented docs"},
        {"entity": "RetrievalFilterMetadata", "count": len(filters), "description": "Canonical search filters"}
    ]
    pd.DataFrame(summary_data).to_csv(output_dir / "retrieval_schema_summary.csv", index=False)

    print(f"RAG Preparation Layer Complete. Retrieval Manifest: {output_dir}/retrieval_manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
