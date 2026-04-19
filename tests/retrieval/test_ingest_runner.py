"""Tests for the Retrieval Ingestion Runner."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import pytest


def test_ingest_runner_smoke():
    """Verify that run_retrieval_ingest.py executes successfully in smoke mode."""
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "run_retrieval_ingest.py"
    prep_dir = repo_root / "artifacts" / "retrieval_prep"
    output_dir = repo_root / "artifacts" / "test_ingest_smoke"
    
    if not prep_dir.exists():
        pytest.skip("retrieval_prep artifacts must exist for this test.")

    cmd = [
        sys.executable,
        str(script_path),
        "--mode", "smoke",
        "--backend", "file",
        "--retrieval-prep-dir", str(prep_dir),
        "--output-dir", str(output_dir)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Ingest runner failed: {result.stderr}"
    assert (output_dir / "ingest_manifest.json").exists()
    assert (output_dir / "local_index_store.jsonl").exists()
