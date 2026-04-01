"""Smoke tests for label integration in the training pipeline.

Verifies:
    - CLI mode resolution to LabelTrainingConfig
    - Integration of custom labels into build_daily_features
    - Basic end-to-end training flow for custom labels
"""

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock

# Import from the script (path adjustment needed as it's not in a package)
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.train_ml_tickers import build_daily_features, _train_custom_label
from src.ml.labels.training_adapter import resolve_label_config, SUPPORTED_LABEL_MODES
from src.ml.data_loader import generate_mock_data


@pytest.fixture
def mock_ohlcv():
    """Generate 200 days of mock data."""
    df = generate_mock_data(ticker="TEST", num_days=200)
    # Ensure raw close is present
    df['close_raw'] = df['close']
    return df


def test_resolve_all_modes():
    """Verify all supported modes resolve correctly."""
    for mode in SUPPORTED_LABEL_MODES:
        config = resolve_label_config(mode)
        assert config.mode == mode
        assert config.generator is not None
        assert config.label_column in config.generator.label_columns


def test_build_features_with_custom_label(mock_ohlcv):
    """Verify build_daily_features applies the custom label."""
    mode = "binary_1d"
    config = resolve_label_config(mode)
    
    # 1. Without custom label (legacy path)
    df_legacy = build_daily_features(mock_ohlcv.copy())
    assert "target_trend_short" in df_legacy.columns
    assert config.label_column not in df_legacy.columns
    
    # 2. With custom label
    df_custom = build_daily_features(mock_ohlcv.copy(), label_config=config)
    assert config.label_column in df_custom.columns
    # Legacy targets should NOT be present in this path to save memory/time
    assert "target_trend_short" not in df_custom.columns


@pytest.mark.parametrize("mode", ["binary_1d", "regression_5d"])
def test_train_custom_label_smoke(mock_ohlcv, mode, tmp_path):
    """Minimal end-to-end training smoke test for both task types."""
    config = resolve_label_config(mode)
    df = build_daily_features(mock_ohlcv, label_config=config)
    
    # Identify some feature columns
    feature_cols = [c for c in df.columns if c.startswith('d_')][:10]
    
    # Run minimal training
    ticker = "TEST"
    metrics = _train_custom_label(
        daily_df=df,
        ticker=ticker,
        ticker_dir=tmp_path,
        feature_cols=feature_cols,
        label_config=config
    )
    
    assert metrics is not None
    if config.task_type == "classification":
        assert f"{mode}_acc" in metrics
        assert (tmp_path / f"trend_classifier_{mode}.joblib").exists()
    else:
        assert f"{mode}_mae" in metrics
        assert (tmp_path / f"regressor_{mode}.joblib").exists()
        
    assert (tmp_path / f"feature_cols_{mode}.joblib").exists()
