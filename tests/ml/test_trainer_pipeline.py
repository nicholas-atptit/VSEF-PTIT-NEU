from __future__ import annotations

import json

import pandas as pd
import pytest

from src.ml.data_loader import generate_mock_data
from src.ml.trainer import DualModelTrainer


def test_cart_training_and_manifest_report(tmp_path) -> None:
    trainer = DualModelTrainer(model_dir=tmp_path / "models")
    df = generate_mock_data(ticker="MOCK", num_days=900)

    result = trainer.train(
        ticker="MOCK",
        df=df,
        algorithms=["cart"],
        horizons=["short"],
        max_depth=3,
    )

    assert result["ticker"] == "MOCK"
    assert result["primary_algorithm"] == "cart"
    assert result["report_rows"]
    assert result["data_provenance"]["uses_mock_data"] is True
    assert result["data_provenance"]["runtime_mode"] == "demo"

    row = result["report_rows"][0]
    assert row["algorithm"] == "cart"
    assert row["artifact_type"] == "joblib"
    assert row["data_start"] <= row["data_end"]
    assert row["raw_rows"] > 0
    assert "train_seconds" in row

    manifest_path = tmp_path / "models" / "MOCK" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["primary_algorithm"] == "cart"
    assert "short" in manifest["horizons"]
    assert manifest["data_provenance"]["uses_mock_data"] is True
    assert manifest["data_provenance"]["runtime_mode"] == "demo"

    features = trainer.compute_features_for_ticker("MOCK", df)
    prediction = trainer.predict("MOCK", features, horizon="short")
    assert prediction["algorithm"] == "cart"
    assert "trend_probabilities" in prediction
    assert "expected_range" in prediction


def test_sequence_inference_requires_sufficient_history(tmp_path) -> None:
    pytest.importorskip("torch")

    trainer = DualModelTrainer(model_dir=tmp_path / "models")
    df = generate_mock_data(ticker="SEQ", num_days=900)
    trainer.train(
        ticker="SEQ",
        df=df,
        algorithms=["lstm"],
        horizons=["short"],
        sequence_length=10,
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
        batch_size=8,
        epochs=1,
        patience=1,
    )

    features = trainer.compute_features_for_ticker("SEQ", df).tail(5)
    with pytest.raises(ValueError, match="Insufficient history"):
        trainer.predict("SEQ", features, horizon="short")
