import asyncio
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.ml.data_loader import VN100DataLoader
from src.ml.feature_engineering import FeatureEngineer

async def main():
    loader = VN100DataLoader(prefer_source="csv")
    df = loader.build_inference_dataset(tickers=["ABS"], lookback_days=300)
    
    fe = FeatureEngineer()
    # FeatureEngineer expects 'date' (renamed from 'time' in trainer)
    df = df.rename(columns={"time": "date", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
    feat_df = fe.transform(df)
    
    print("Columns produced by FeatureEngineer:")
    print(sorted(feat_df.columns))
    
    if "return_20d" in feat_df.columns:
        print("✅ return_20d is PRESENT")
    else:
        print("❌ return_20d is MISSING")

if __name__ == "__main__":
    asyncio.run(main())
