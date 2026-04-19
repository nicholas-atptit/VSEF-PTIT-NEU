"""Focused tests for the Analysis Feed Layer."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import pandas as pd
from src.core.analysis_schema import (
    ForecastResearchPacket,
    HistoricalCaseRecord,
    AnalystMemoDraft
)
from src.reporting.analysis_feed import (
    normalize_to_research_packets,
    generate_case_records,
    generate_memo_drafts,
    generate_retrieval_metadata
)


@pytest.fixture
def mock_manifest():
    return {
        "core_run_id": "test_run_123",
        "git": {"commit_hash": "abcdef123456"},
        "artifact_paths": {
            "analysis_packets": "analysis_packets.jsonl"
        }
    }


@pytest.fixture
def mock_quant_core_dir(tmp_path):
    # Create a mock analysis_packets.jsonl
    d = tmp_path / "quant_core"
    d.mkdir()
    
    packet_file = d / "analysis_packets.jsonl"
    manifest_file = d / "run_manifest.json"
    
    packet_data = [
        {
            "timestamp": "2026-01-01T09:00:00Z",
            "ticker": "ACB",
            "horizon": 10,
            "target_type": "forward_return",
            "run_mode": "full_forecast",
            "primary_model_name": "SARIMAX",
            "primary_model_role": "statistical",
            "primary_prediction": 0.015,
            "model_by_model_predictions": [],
            "model_ranks": {},
            "ensemble_summary": None,
            "model_agreement_score": 0.8,
            "agreement_bucket": "high",
            "risk_summary": {"vol_forecast": 0.02},
            "regime_summary": {"regime_label": "bull"},
            "policy_summary": {"mean_position_size": 1.0},
            "volatility_bucket": "moderate",
            "signal_strength_bucket": "high",
            "realized_y_true": 0.025
        },
        {
            "timestamp": "2026-01-01T09:00:00Z",
            "ticker": "VNM",
            "horizon": 10,
            "target_type": "forward_return",
            "run_mode": "full_forecast",
            "primary_model_name": "SARIMAX",
            "primary_model_role": "statistical",
            "primary_prediction": -0.005,
            "model_by_model_predictions": [],
            "model_ranks": {},
            "ensemble_summary": None,
            "model_agreement_score": 0.4,
            "agreement_bucket": "low",
            "risk_summary": {"vol_forecast": 0.01},
            "regime_summary": {"regime_label": "sideways"},
            "policy_summary": {"mean_position_size": 0.0},
            "volatility_bucket": "low",
            "signal_strength_bucket": "low",
            "realized_y_true": None
        }
    ]
    
    with open(packet_file, "w") as f:
        for p in packet_data:
            f.write(json.dumps(p) + "\n")
            
    manifest_data = {
        "core_run_id": "test_run_123",
        "git": {"commit_hash": "abcdef123456"},
        "artifact_paths": {
            "analysis_packets": str(packet_file)
        }
    }
    with open(manifest_file, "w") as f:
        json.dump(manifest_data, f)
        
    return d


def test_packet_schema_and_provenance(mock_quant_core_dir, mock_manifest):
    packets = normalize_to_research_packets(mock_quant_core_dir, mock_manifest)
    assert len(packets) == 2
    pkt = packets[0]
    assert isinstance(pkt, ForecastResearchPacket)
    assert pkt.source_run_id == "test_run_123"
    assert pkt.ticker == "ACB"
    assert pkt.packet_id.startswith("pkt_test_run_123_ACB_2026-01-01")
    assert pkt.realized_outcome_label == "gain"
    
    # Check VNM (missing outcome)
    pkt_vnm = packets[1]
    assert pkt_vnm.realized_outcome_label is None


def test_cross_link_integrity(mock_quant_core_dir, mock_manifest):
    packets = normalize_to_research_packets(mock_quant_core_dir, mock_manifest)
    cases = generate_case_records(packets)
    memos = generate_memo_drafts(packets)
    
    assert len(cases) == len(packets)
    assert len(memos) == len(packets)
    
    for pkt, case, memo in zip(packets, cases, memos):
        assert case.source_packet_id == pkt.packet_id
        assert memo.related_packet_id == pkt.packet_id
        assert case.case_id == f"case_{pkt.packet_id}"
        assert memo.memo_id == f"memo_{pkt.packet_id}"


def test_deterministic_summary(mock_quant_core_dir, mock_manifest):
    packets = normalize_to_research_packets(mock_quant_core_dir, mock_manifest)
    cases = generate_case_records(packets)
    
    summary = cases[0].summary_text
    assert "Ticker ACB" in summary
    assert "bull regime" in summary
    assert "high agreement" in summary
    assert "threshold passed" in summary


def test_retrieval_metadata_alignment(mock_quant_core_dir, mock_manifest):
    packets = normalize_to_research_packets(mock_quant_core_dir, mock_manifest)
    cases = generate_case_records(packets)
    memos = generate_memo_drafts(packets)
    metadata = generate_retrieval_metadata(packets, cases, memos)
    
    assert len(metadata) == len(packets)
    meta = metadata[0]
    assert meta.related_packet_id == packets[0].packet_id
    assert meta.ticker == "ACB"


def test_idempotency(mock_quant_core_dir, mock_manifest):
    packets1 = normalize_to_research_packets(mock_quant_core_dir, mock_manifest)
    packets2 = normalize_to_research_packets(mock_quant_core_dir, mock_manifest)
    
    for p1, p2 in zip(packets1, packets2):
        assert p1.packet_id == p2.packet_id
        d1 = p1.model_dump()
        d2 = p2.model_dump()
        d1.pop("generated_at", None)
        d2.pop("generated_at", None)
        assert d1 == d2
