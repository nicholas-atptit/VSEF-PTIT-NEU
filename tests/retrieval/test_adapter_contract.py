"""Tests for the Retrieval Adapter contract and compliance."""

from __future__ import annotations

import pytest
from src.retrieval.file_adapter import FileRetrievalAdapter
from src.retrieval.base import RetrievalAdapter
from src.retrieval.types import RetrievalIndexRecord


@pytest.fixture
def temp_index(tmp_path):
    return tmp_path


def test_adapter_inheritance():
    """Verify that FileRetrievalAdapter inherits from RetrievalAdapter."""
    adapter = FileRetrievalAdapter("tmp")
    assert isinstance(adapter, RetrievalAdapter)


def test_record_validation():
    """Verify RetrievalIndexRecord validation."""
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
    assert record.retrieval_doc_id == "test_1"
    assert record.metadata["ticker"] == "AAA"
