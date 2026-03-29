import asyncio
import pandas as pd
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

async def generate_features():
    print("🚀 Extracting Sentiment Intelligence from Database...")
    url = os.getenv("TIMESCALE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/algo_trading")
    engine = create_async_engine(url)
    
    try:
        async with engine.connect() as conn:
            # Query all analyzed news
            query = text("SELECT ticker, sentiment_score, timestamp FROM news_intelligence")
            res = await conn.execute(query)
            rows = res.fetchall()
            
            if not rows:
                print("⚠️ No news intelligence data found in DB.")
                return

            df = pd.DataFrame(rows, columns=['ticker', 'sentiment', 'timestamp'])
            
            # Convert timestamp to date
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            
            # Aggregate per ticker per day
            daily = df.groupby(['ticker', 'date']).agg({
                'sentiment': ['mean', 'count', 'std']
            }).reset_index()
            
            # Flatten columns
            daily.columns = ['ticker', 'date', 'sentiment_avg', 'news_volume', 'sentiment_std']
            daily['sentiment_std'] = daily['sentiment_std'].fillna(0.0)
            
            # Save to CSV for ML trainer
            os.makedirs("data", exist_ok=True)
            output_path = "data/sentiment_features.csv"
            daily.to_csv(output_path, index=False)
            print(f"✅ Generated {len(daily)} sentiment records and saved to {output_path}")
            
    except Exception as e:
        print(f"❌ Error generating sentiment features: {e}")
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(generate_features())
