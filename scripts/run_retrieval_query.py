"""Runner for the Retrieval Query Layer."""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.file_adapter import FileRetrievalAdapter
from src.retrieval.filters import RetrievalFilters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the Retrieval Index.")
    parser.add_argument("--backend", choices=["file", "qdrant"], default="file")
    parser.add_argument("--index-dir", default="artifacts/retrieval_ingest", help="Directory for local index")
    parser.add_argument("--query", type=str, default="", help="Text search query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--filter", action="append", help="Filter in format k:v (e.g. ticker:ACB)")
    parser.add_argument("--fetch-id", type=str, help="Fetch a specific document by ID")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index_dir = Path(args.index_dir)
    
    # Init Backend
    if args.backend == "file":
        adapter = FileRetrievalAdapter(index_dir)
    else:
        from src.retrieval.qdrant_adapter import QdrantRetrievalAdapter
        adapter = QdrantRetrievalAdapter()
        
    print(f"Backend Status: {adapter.health_check()}")

    # 1. Fetch by ID path
    if args.fetch_id:
        print(f"Fetching document {args.fetch_id}...")
        docs = adapter.fetch_by_id([args.fetch_id])
        if not docs:
            print("Not found.")
        else:
            print(json.dumps(docs[0].model_dump(), indent=2))
        return 0

    # 2. Query path
    filters = None
    if args.filter:
        f_map = {}
        for f in args.filter:
            if ":" in f:
                k, v = f.split(":", 1)
                # Try to convert to int if numeric
                if v.isdigit():
                    v = int(v)
                f_map[k] = v
        filters = RetrievalFilters(**f_map)

    print(f"Querying: '{args.query}' (filters: {filters.to_dict() if filters else 'None'}, top_k: {args.top_k})...")
    results = adapter.query(args.query, filters=filters, top_k=args.top_k)
    
    print(f"Found {len(results)} results:")
    for i, res in enumerate(results):
        print(f"\n[{i+1}] ID: {res.retrieval_doc_id} (Score: {res.score:.2f})")
        print(f"    Source: {res.source_type} | Source ID: {res.source_case_id or res.source_packet_id}")
        print(f"    Summary: {res.summary}")

    # Output smoke artifact
    smoke_output = {
        "query": args.query,
        "filters": filters.to_dict() if filters else None,
        "result_count": len(results),
        "results": [r.model_dump() for r in results]
    }
    with open(index_dir / "retrieval_query_smoke.json", "w", encoding="utf-8") as f:
        json.dump(smoke_output, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
