import pandas as pd
import pytest

from src.governance.artifact_policy import assert_deletion_allowed, is_protected_evidence_path
from src.governance.claim_boundary import claim_statement
from src.governance.split_policy import assign_split


def test_strict_split_requires_feature_and_target_inside_same_period():
    frame = pd.DataFrame(
        {
            "feature_timestamp": ["2023-12-30", "2023-12-31", "2024-06-01", "2024-12-31", "2025-01-01"],
            "target_timestamp": ["2023-12-31", "2024-01-02", "2024-06-02", "2025-01-02", "2025-01-02"],
        }
    )
    result = assign_split(frame)
    assert result["split"].tolist() == ["train", pd.NA, "validation", pd.NA, "final"]


def test_protected_evidence_cannot_be_deleted():
    assert is_protected_evidence_path("reports/generated/vn30_qml_forecasting/qml_manifest.json")
    with pytest.raises(PermissionError):
        assert_deletion_allowed("data/raw/example.csv")


def test_claim_statement_preserves_blocked_claims():
    statement = claim_statement()
    assert "no BUY/SELL" in statement
    assert "no live deployment" in statement
    assert "final rows scoring-only" in statement
