"""Verification script for Phase 2: Technical Forecasting Upgrades."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.trainer import DualModelTrainer
from src.ml.signal_generator import SignalGenerator

def test_trainer_horizons():
    trainer = DualModelTrainer()
    ticker = "SSI"
    
    print(f"--- Testing DualModelTrainer for {ticker} ---")
    
    # Create a dummy features row (assuming 20 features based on trainer_v3)
    # We'll just fetch the actual feature_cols if possible
    try:
        trainer._ensure_models_loaded(ticker)
        feature_cols = trainer._models[ticker].get("feature_cols", [])
        print(f"Loaded {len(feature_cols)} feature columns.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    dummy_row = pd.Series({col: 0.1 for col in feature_cols})
    
    horizons = ["short", "mid", "long"]
    for h in horizons:
        try:
            pred = trainer.predict(ticker, dummy_row, horizon=h)
            print(f"\nHorizon: {h.upper()}")
            print(f"  - Action: {pred.get('trend_probabilities', {}).get('confidence', 'N/A')} confidence")
            print(f"  - Suffix Used (Fallback Test): {pred.get('horizon')}")
            print(f"  - Feature Version: {pred.get('feature_set_version')}")
        except Exception as e:
            print(f"  - Error predicting {h}: {e}")

def test_signal_generator():
    sg = SignalGenerator()
    ticker = "SSI"
    current_close = 35.0
    
    mock_output = {
        "trend_probabilities": {"up": 0.75, "sideways": 0.10, "down": 0.15, "confidence": 0.88},
        "expected_range": {"bottom_10th": 34.0, "median_50th": 36.5, "ceiling_90th": 39.0},
        "horizon": "mid",
        "feature_set_version": "v5.0"
    }
    
    print("\n--- Testing SignalGenerator Advanced Zones ---")
    signal = sg.generate(ticker, current_close, mock_output)
    
    q_sig = signal["quantitative_signals"]
    action_plan = q_sig["action_plan"]
    
    print(f"Ticker: {signal['ticker']}")
    print(f"Action: {action_plan['recommendation']}")
    print(f"Entry Zone: {action_plan['entry_zone']}")
    print(f"Exit Zones: {action_plan['exit_zones']}")
    print(f"Rationale: {action_plan['rationale']}")
    print(f"Horizon: {q_sig['horizon']}")

if __name__ == "__main__":
    test_trainer_horizons()
    test_signal_generator()
