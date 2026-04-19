"""Runner for the Analysis Feed Layer."""

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
from src.reporting.analysis_feed import (
    load_quant_core_manifest,
    normalize_to_research_packets,
    generate_case_records,
    generate_memo_drafts,
    generate_retrieval_metadata,
    write_analysis_feed
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Analysis Feed Layer.")
    parser.add_argument("--mode", choices=["smoke", "build_from_existing_quant_core"], required=True)
    parser.add_argument("--quant-core-dir", type=str, help="Path to validated quant-core artifact directory")
    parser.add_argument("--output-dir", default="artifacts/analysis_feed", help="Where to write feed artifacts")
    return parser.parse_args()


def render_feed_summary_markdown(manifest: dict) -> str:
    """Render a concise summary of the generated feed."""
    return f"""# Analysis Feed Summary
    
## Feed Identity
- **Feed ID**: `{manifest['feed_id']}`
- **Generated At**: {manifest['generated_at']}
- **Schema Version**: {manifest['schema_version']}

## Source Provenance
- **Source Run ID**: `{manifest['source_run_id']}`
- **Source Manifest**: `{manifest['source_quant_core_manifest']}`

## Content Stats
- **Research Packets**: {manifest['packet_count']}
- **Case Records**: {manifest['case_count']}
- **Analyst Memos**: {manifest['memo_count']}

## Artifacts
{chr(10).join([f"- **{k}**: `{v}`" for k, v in manifest['artifact_paths'].items()])}
"""


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == "smoke":
        print("Running in SMOKE mode...")
        # Create dummy quant core artifacts or use existing if provided
        # For simplicity, we assume we need a source dir
        if not args.quant_core_dir:
            print("Error: --quant-core-dir is required even in smoke mode to find source schema context.")
            return 1

    source_dir = Path(args.quant_core_dir)
    print(f"Loading source manifest from {source_dir}...")
    manifest_src = load_quant_core_manifest(source_dir)
    
    print("Normalizing Forecast Research Packets...")
    packets = normalize_to_research_packets(source_dir, manifest_src)
    
    if args.mode == "smoke":
        packets = packets[:10] # Tiny slice
        print(f"Smoke mode: sliced to {len(packets)} packets")
        
    print("Generating Historical Case Records...")
    cases = generate_case_records(packets)
    
    print("Generating Analyst Memo Drafts...")
    memos = generate_memo_drafts(packets)
    
    print("Building Retrieval Metadata...")
    metadata = generate_retrieval_metadata(packets, cases, memos)
    
    print(f"Writing Analysis Feed to {output_dir}...")
    artifact_paths = write_analysis_feed(output_dir, manifest_src, packets, cases, memos, metadata)
    
    # Final Manifest
    feed_id = f"feed_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    feed_manifest = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_run_id": manifest_src.get("core_run_id") or manifest_src["git"]["commit_hash"][:8],
        "source_manifest_path": str(source_dir / "run_manifest.json"),
        "source_artifact_dir": str(source_dir),
        "feed_id": feed_id,
        "source_quant_core_manifest": str(source_dir / "run_manifest.json"),
        "packet_count": len(packets),
        "case_count": len(cases),
        "memo_count": len(memos),
        "artifact_paths": artifact_paths
    }
    
    manifest_path = output_dir / "feed_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(feed_manifest, f, indent=2)
        
    # Summary Markdown
    summary_path = output_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(render_feed_summary_markdown(feed_manifest))
        
    # Schema Summary CSV
    schema_summary = [
        {"entity": "ForecastResearchPacket", "count": len(packets), "description": "Normalized quant-core outputs"},
        {"entity": "HistoricalCaseRecord", "count": len(cases), "description": "RAG-ready deterministic cases"},
        {"entity": "AnalystMemoDraft", "count": len(memos), "description": "Placeholders for future analyst review"},
        {"entity": "RetrievalMetadata", "count": len(metadata), "description": "Flat search/filter index"}
    ]
    pd.DataFrame(schema_summary).to_csv(output_dir / "feed_schema_summary.csv", index=False)
    
    print(f"Analysis Feed Layer Complete. Feed Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
