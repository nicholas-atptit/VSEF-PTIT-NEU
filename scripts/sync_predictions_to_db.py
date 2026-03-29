import asyncio
import os
import pandas as pd
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

async def sync_predictions():
    print("🚀 Syncing ML Predictions from Models to PostgreSQL...")
    url = os.getenv("TIMESCALE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/algo_trading")
    engine = create_async_engine(url)
    
    report_path = Path("reports/evaluation_report.csv")
    if not report_path.exists():
        # Try the new phase 42 report
        report_path = Path("reports/performance_universal_v4.csv")
        
    if not report_path.exists():
        print("⚠️ No evaluation report found. Skipping sync.")
        return

    try:
        df = pd.read_csv(report_path)
        print(f"📊 Found {len(df)} trained tickers in report.")
        
        async with engine.begin() as conn:
            # Create table if not exists (Basic schema)
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS agent_predictions (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10) NOT NULL,
                    prediction_label VARCHAR(20),
                    confidence FLOAT,
                    target_price FLOAT,
                    horizon VARCHAR(10),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            for _, row in df.iterrows():
                ticker = row['ticker']
                # Map metrics to labels for the UI
                # We'll use 'long_acc' or 'elite_acc' as confidence proxy
                confidence = row.get('elite_acc', 0.0)
                if confidence == 0:
                    confidence = row.get('acc', 0.5)
                
                # Simple label logic
                label = "UP" if row.get('acc', 0.5) > 0.55 else "NEUTRAL"
                
                # Insert into DB
                await conn.execute(
                    text("INSERT INTO agent_predictions (ticker, prediction_label, confidence, horizon) VALUES (:t, :l, :c, 'short')"),
                    {"t": ticker, "l": label, "c": float(confidence)}
                )
            
        print("✅ Sync completed successfully.")
        
    except Exception as e:
        print(f"❌ Sync error: {e}")
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(sync_predictions())
