"""Retrieval preparation and deterministic rendering layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.core.retrieval_schema import (
    RetrievalDocument,
    RetrievalChunk,
    RetrievalFilterMetadata
)


def load_analysis_feed_manifest(analysis_feed_dir: str | Path) -> Dict[str, Any]:
    """Load the source analysis-feed manifest."""
    manifest_path = Path(analysis_feed_dir) / "feed_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Analysis-feed manifest not found at {manifest_path}")
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_case_document(case_data: Dict[str, Any], provenance: Dict[str, str]) -> RetrievalDocument:
    """Render a HistoricalCaseRecord into a RetrievalDocument."""
    
    case_id = case_data["case_id"]
    packet_id = case_data["source_packet_id"]
    ticker = case_data["ticker"]
    
    title = f"Research Case: {ticker} ({case_data['regime_label']} regime)"
    
    # Deterministic Narrative
    narrative = (
        f"Research case for {ticker} at horizon {case_data['horizon']}. "
        f"Context: {case_data['summary_text']} "
        f"Target Type: {case_data['target_type']}. "
        f"Tags: {', '.join(case_data.get('tags', []))}. "
    )
    if case_data.get("realized_outcome_label"):
        narrative += f"Final Outcome: {case_data['realized_outcome_label']}."

    return RetrievalDocument(
        **provenance,
        retrieval_doc_id=f"rdoc_case_{case_id}",
        source_type="case",
        source_entity_id=case_id,
        source_packet_id=packet_id,
        source_case_id=case_id,
        title=title,
        summary=case_data["summary_text"],
        text=narrative,
        metadata={
            "ticker": ticker,
            "regime": case_data["regime_label"],
            "volatility": case_data["volatility_bucket"],
            "agreement": case_data["agreement_bucket"],
            "outcome": case_data.get("realized_outcome_label")
        }
    )


def render_packet_document(packet_data: Dict[str, Any], provenance: Dict[str, str]) -> RetrievalDocument:
    """Render a ForecastResearchPacket into a RetrievalDocument."""
    
    packet_id = packet_data["packet_id"]
    ticker = packet_data["ticker"]
    date_str = packet_data["timestamp"]
    
    title = f"Quant Forecast: {ticker} on {date_str} (H{packet_data['horizon']})"
    
    # Synoptic rendering
    narrative = (
        f"Technical quant forecast for {ticker} generated on {date_str}. "
        f"Horizon: {packet_data['horizon']} days. "
        f"Primary Prediction: {packet_data.get('primary_prediction', 0.0):.4f} "
        f"via {packet_data['primary_model_name']} ({packet_data['primary_model_role']}). "
        f"Agreement: {packet_data['agreement_bucket']} (Score: {packet_data.get('model_agreement_score', 0.0):.2f}). "
        f"Regime: {packet_data.get('regime_summary', {}).get('regime_label', 'unknown')}. "
        f"Volatility: {packet_data['volatility_bucket']}. "
    )
    if packet_data.get("realized_outcome_label"):
        narrative += f"Realized Outcome: {packet_data['realized_outcome_label']} ({packet_data.get('realized_return', 0.0):.4f})."

    return RetrievalDocument(
        **provenance,
        retrieval_doc_id=f"rdoc_packet_{packet_id}",
        source_type="packet",
        source_entity_id=packet_id,
        source_packet_id=packet_id,
        title=title,
        summary=f"Quant forecast for {ticker} (H{packet_data['horizon']}) with {packet_data['agreement_bucket']} agreement.",
        text=narrative,
        metadata={
            "ticker": ticker,
            "horizon": packet_data["horizon"],
            "target": packet_data["target_type"],
            "prediction": packet_data.get("primary_prediction"),
            "agreement": packet_data["agreement_bucket"]
        }
    )


def generate_retrieval_filter_metadata(doc: RetrievalDocument, source_data: Dict[str, Any]) -> RetrievalFilterMetadata:
    """Extract canonical filter fields for the document."""
    
    # We unify fields from both doc metadata and source_data
    return RetrievalFilterMetadata(
        retrieval_doc_id=doc.retrieval_doc_id,
        source_type=doc.source_type,
        ticker=source_data["ticker"],
        ticker_group=source_data.get("ticker_group") or source_data.get("group_name"),
        horizon=source_data["horizon"],
        target_type=source_data["target_type"],
        regime_label=source_data.get("regime_label") or source_data.get("regime_summary", {}).get("regime_label"),
        volatility_bucket=source_data.get("volatility_bucket", "unknown"),
        signal_strength_bucket=source_data.get("signal_strength_bucket", "unknown"),
        agreement_bucket=source_data.get("agreement_bucket", "unknown"),
        model_role_context=source_data.get("model_role_context") or source_data.get("primary_model_role", ""),
        run_mode=source_data["run_mode"],
        cost_mode=source_data.get("cost_mode"),
        realized_outcome_label=source_data.get("realized_outcome_label"),
        date=str(pd.Timestamp(source_data["timestamp"]).date()) if "timestamp" in source_data else ""
    )


def write_retrieval_prep_outputs(
    output_dir: str | Path,
    docs: List[RetrievalDocument],
    chunks: List[RetrievalChunk],
    filters: List[RetrievalFilterMetadata]
) -> Dict[str, str]:
    """Write retrieval preparation artifacts."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Documents JSONL
    doc_file = out_path / "retrieval_documents.jsonl"
    with open(doc_file, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(d.model_dump_json() + "\n")
    paths["retrieval_documents"] = str(doc_file.relative_to(out_path))
    
    # Chunks JSONL
    chunk_file = out_path / "retrieval_chunks.jsonl"
    with open(chunk_file, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")
    paths["retrieval_chunks"] = str(chunk_file.relative_to(out_path))
    
    # Filter Metadata CSV
    f_df = pd.DataFrame([f.model_dump() for f in filters])
    filter_file = out_path / "retrieval_filter_metadata.csv"
    f_df.to_csv(filter_file, index=False)
    paths["retrieval_filter_metadata"] = str(filter_file.relative_to(out_path))
    
    # Source Map CSV
    source_map = [
        {"retrieval_doc_id": d.retrieval_doc_id, "source_type": d.source_type, "source_entity_id": d.source_entity_id}
        for d in docs
    ]
    pd.DataFrame(source_map).to_csv(out_path / "retrieval_source_map.csv", index=False)
    paths["retrieval_source_map"] = "retrieval_source_map.csv"
    
    return paths
