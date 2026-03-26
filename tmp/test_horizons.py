import asyncio
import pandas as pd
from pathlib import Path
import sys

# Add current dir to path for imports
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.ml.trainer import DualModelTrainer

async def main():
    trainer = DualModelTrainer()
    
    ticker = "VGI"
    # Mock data
    df = pd.DataFrame({
        "time": pd.date_range(start="2024-01-01", periods=100, freq="D"),
        "open": [10.0] * 100,
        "high": [11.0] * 100,
        "low": [9.0] * 100,
        "close": [10.5] * 100,
        "volume": [1000] * 100
    })
    
    # Use the trainer's built-in feature pipeline
    features_df = trainer.compute_features_for_ticker(ticker, df)
    if features_df is None:
        print("Feature computation failed")
        return
        
    last_row = features_df.iloc[-1]
    trainer._ensure_models_loaded(ticker)
    
    horizons = ["1d", "1w", "1m", "6m"]
    for h in horizons:
        print(f"\n--- Testing Horizon: {h} ---")
        try:
            pred = trainer.predict(ticker, last_row, horizon=h)
            if pred:
                print(f"SUCCESS: {pred['trend_probabilities']}")
                print(f"   Range: {pred['expected_range']}")
            else:
                print(f"FAILED: Returned empty or None")
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(main())
