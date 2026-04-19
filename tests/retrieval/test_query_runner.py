"""Tests for the Retrieval Query Runner."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import pytest
import json


def test_query_runner_smoke():
    """Verify that run_retrieval_query.py executes successfully."""
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "run_retrieval_query.py"
    index_dir = repo_root / "artifacts" / "retrieval_ingest_smoke"
    
    if not index_dir.exists():
        pytest.skip("retrieval_ingest_smoke artifacts must exist for this test.")

    cmd = [
        sys.executable,
        str(script_path),
        "--backend", "file",
        "--index-dir", str(index_dir),
        "--query", "ACB",
        "--top-k", "3"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Query runner failed: {result.stderr}"
    
    smoke_file = index_dir / "retrieval_query_smoke.json"
    assert smoke_file.exists()
    with open(smoke_file, "r") as f:
        data = json.load(f)
        assert "results" in data
        # results might be 0 if 'ACB' isn't in the 10 smoke docs, 
        # but the runner itself should succeed.
