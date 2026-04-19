"""Tests for the Local File-Backed Retrieval Adapter."""

from __future__ import annotations

import pytest
from src.retrieval.file_adapter import FileRetrievalAdapter
from src.retrieval.types import RetrievalIndexRecord
from src.retrieval.filters import RetrievalFilters


@pytest.fixture
def adapter(tmp_path):
    return FileRetrievalAdapter(tmp_path)


def test_ingest_and_fetch(adapter):
    record = RetrievalIndexRecord(
        retrieval_doc_id="test_1",
        source_type="case",
        text="Sample text",
        summary="Sample summary",
        metadata={"ticker": "AAA"},
        source_run_id="run_1",
        source_manifest_path="m.json",
        source_artifact_dir="dir"
    )
    
    adapter.ingest_documents([record])
    fetched = adapter.fetch_by_id(["test_1"])
    
    assert len(fetched) == 1
    assert fetched[0].retrieval_doc_id == "test_1"


def test_upsert_behavior(adapter):
    record1 = RetrievalIndexRecord(
        retrieval_doc_id="test_1", source_type="case", text="T1", summary="S1",
        source_run_id="r1", source_manifest_path="m", source_artifact_dir="d"
    )
    adapter.ingest_documents([record1])
    
    # Update
    record2 = record1.model_copy(update={"text": "Updated text"})
    adapter.upsert_documents([record2])
    
    fetched = adapter.fetch_by_id(["test_1"])
    assert fetched[0].text == "Updated text"


def test_query_filtering(adapter):
    r1 = RetrievalIndexRecord(
        retrieval_doc_id="t1", source_type="case", text="Apple", summary="S1",
        metadata={"ticker": "AAPL", "regime_label": "bull"},
        source_run_id="r1", source_manifest_path="m", source_artifact_dir="d"
    )
    r2 = RetrievalIndexRecord(
        retrieval_doc_id="t2", source_type="case", text="Banana", summary="S2",
        metadata={"ticker": "BNN", "regime_label": "bear"},
        source_run_id="r1", source_manifest_path="m", source_artifact_dir="d"
    )
    adapter.ingest_documents([r1, r2])
    
    # Text search
    results = adapter.query("apple")
    assert len(results) == 1
    assert results[0].retrieval_doc_id == "t1"
    
    # Filter search
    filters = RetrievalFilters(regime_label="bear")
    results = adapter.query("", filters=filters)
    assert len(results) == 1
    assert results[0].retrieval_doc_id == "t2"


def test_delete_behavior(adapter):
    r1 = RetrievalIndexRecord(
        retrieval_doc_id="t1", source_type="case", text="T1", summary="S1",
        source_run_id="r1", source_manifest_path="m", source_artifact_dir="d"
    )
    adapter.ingest_documents([r1])
    adapter.delete_documents(["t1"])
    assert len(adapter.fetch_by_id(["t1"])) == 0
