import pytest
import json
import os
import pandas as pd
from src.reporting.ranked_predictions import RankedPredictionGenerator

@pytest.fixture
def mock_predictions():
    return {
        "AAA": {
            "technical": {
                "current_price": 10.0,
                "horizons": [{
                    "horizon": "1w",
                    "trend_probs": {"up": 0.8, "down": 0.1, "sideways": 0.1},
                    "expected_range": {"bottom_10th": 9.5, "median_50th": 11.0, "ceiling_90th": 12.0},
                    "volatility_score": 0.05,
                    "confidence": 0.9
                }]
            },
            "fusion": {"action": "BUY"}
        },
        "BBB": {
            "technical": {
                "current_price": 20.0,
                "horizons": [{
                    "horizon": "1w",
                    "trend_probs": {"up": 0.4, "down": 0.4, "sideways": 0.2},
                    "expected_range": {"bottom_10th": 18.0, "median_50th": 20.0, "ceiling_90th": 22.0},
                    "volatility_score": 0.15,
                    "confidence": 0.8
                }]
            },
            "fusion": {"action": "HOLD"}
        },
        "CCC": {
            "technical": {
                "current_price": 30.0,
                "horizons": [{
                    "horizon": "1w",
                    "trend_probs": {"up": 0.1, "down": 0.8, "sideways": 0.1},
                    "expected_range": {"bottom_10th": 25.0, "median_50th": 28.0, "ceiling_90th": 30.0},
                    "volatility_score": 0.02,
                    "confidence": 0.95
                }]
            },
            "fusion": {"action": "SELL"}
        }
    }

def test_flattening(mock_predictions):
    gen = RankedPredictionGenerator(output_dir="tmp_reports")
    gen.predictions = mock_predictions
    df = gen.flatten_predictions()
    
    assert len(df) == 3
    assert df[df["ticker"] == "AAA"]["prob_up"].iloc[0] == 0.8
    assert df[df["ticker"] == "CCC"]["prob_down"].iloc[0] == 0.8
    # AAA expected return: (11.0 / 10.0) - 1 = 0.1
    assert pytest.approx(df[df["ticker"] == "AAA"]["expected_return"].iloc[0]) == 0.1

def test_ranking_logic(mock_predictions):
    gen = RankedPredictionGenerator(output_dir="tmp_reports")
    gen.predictions = mock_predictions
    ranks = gen.generate_ranks(top_n=2)
    
    # Top long 1d (AAA should be #1)
    assert ranks["top_long_1d"].iloc[0]["ticker"] == "AAA"
    
    # High risk (BBB should be #1)
    assert ranks["high_risk_volatility_names"].iloc[0]["ticker"] == "BBB"
    
    # Top expected return (AAA should be #1)
    assert ranks["top_expected_return"].iloc[0]["ticker"] == "AAA"

def test_export(mock_predictions, tmp_path):
    # Use tmp_path fixture for safety
    report_dir = tmp_path / "reports"
    gen = RankedPredictionGenerator(output_dir=str(report_dir))
    gen.predictions = mock_predictions
    
    files = gen.export_reports(top_n=2)
    assert len(files) > 0
    for f in files:
        assert os.path.exists(f)
