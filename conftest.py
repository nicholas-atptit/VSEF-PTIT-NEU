from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from config.settings import PROJECT_ROOT


@pytest.fixture
def tmp_path():
    """Provide a repo-local temporary directory for deterministic offline test runs."""
    base_dir = PROJECT_ROOT / "tmp" / "pytest_tmpdirs"
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"case_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
