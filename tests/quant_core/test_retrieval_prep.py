"""Focused tests for the RAG Preparation Layer."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import pandas as pd
from src.core.retrieval_schema import RetrievalDocument, RetrievalFilterMetadata
from src.reporting.retrieval_prep import (
    render_case_document,
    render_packet_document,
    generate_retrieval_filter_metadata
)


@pytest.fixture
def mock_provenance():
    return {
        "source_run_id": "src_run_123",
        "source_manifest_path": "/path/to/manifest.json",
        "source_artifact_dir": "/path/to/artifacts"
    }


def test_render_case_document_narrative(mock_provenance):
    case_data = {
        "case_id": "case_1",
        "source_packet_id": "pkt_1",
        "ticker": "ACB",
        "horizon": 10,
        "target_type": "forward_return",
        "regime_label": "bull",
        "volatility_bucket": "moderate",
        "agreement_bucket": "high",
        "run_mode": "full_forecast",
        "timestamp": "2026-01-01T09:00:00Z",
        "summary_text": "Ticker ACB, bull regime...",
        "tags": ["ticker:ACB", "regime:bull"],
        "realized_outcome_label": "gain"
    }
    
    doc = render_case_document(case_data, mock_provenance)
    assert isinstance(doc, RetrievalDocument)
    assert doc.retrieval_doc_id == "rdoc_case_case_1"
    assert "Research Case: ACB" in doc.title
    assert "gain" in doc.text
    assert doc.metadata["regime"] == "bull"


def test_render_packet_document_narrative(mock_provenance):
    packet_data = {
        "packet_id": "pkt_1",
        "ticker": "VNM",
        "timestamp": "2026-01-01T09:00:00Z",
        "horizon": 10,
        "target_type": "forward_return",
        "primary_model_name": "SARIMAX",
        "primary_model_role": "statistical",
        "primary_prediction": 0.0123,
        "agreement_bucket": "medium",
        "model_agreement_score": 0.65,
        "volatility_bucket": "low",
        "regime_summary": {"regime_label": "sideways"},
        "realized_outcome_label": None
    }
    
    doc = render_packet_document(packet_data, mock_provenance)
    assert doc.retrieval_doc_id == "rdoc_packet_pkt_1"
    assert "Quant Forecast: VNM" in doc.title
    assert "0.0123" in doc.text
    assert "medium" in doc.text


def test_retrieval_filter_metadata_extraction(mock_provenance):
    packet_data = {
        "packet_id": "pkt_1",
        "ticker": "VNM",
        "timestamp": "2026-01-01T09:00:00Z",
        "horizon": 10,
        "target_type": "forward_return",
        "run_mode": "full_forecast",
        "primary_model_name": "SARIMAX",
        "primary_model_role": "statistical",
        "volatility_bucket": "low",
        "regime_summary": {"regime_label": "sideways"},
        "agreement_bucket": "medium"
    }
    
    doc = render_packet_document(packet_data, mock_provenance)
    filters = generate_retrieval_filter_metadata(doc, packet_data)
    
    assert isinstance(filters, RetrievalFilterMetadata)
    assert filters.ticker == "VNM"
    assert filters.regime_label == "sideways"
    assert filters.run_mode == "full_forecast"
    assert filters.date == "2026-01-01"


def test_idempotency_rendering(mock_provenance):
    case_data = {
        "case_id": "case_1", "source_packet_id": "pkt_1", "ticker": "ACB", "horizon": 10,
        "target_type": "forward_return", "regime_label": "bull", "volatility_bucket": "moderate",
        "agreement_bucket": "high", "summary_text": "Ticker ACB..."
    }
    
    doc1 = render_case_document(case_data, mock_provenance)
    doc2 = render_case_document(case_data, mock_provenance)
    
    # Ignore generated_at
    d1 = doc1.model_dump(); d1.pop("generated_at")
    d2 = doc2.model_dump(); d2.pop("generated_at")
    assert d1 == d2
