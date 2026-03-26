import asyncio
import sys
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.database.connection import get_session
from sqlalchemy import text

async def main():
    try:
        async with get_session() as session:
            # 1. List all tables in 'public' schema
            print("--- PUBLIC TABLES ---")
            res = await session.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
            tables = [r[0] for r in res.fetchall()]
            for t in sorted(tables):
                print(f"  - {t}")
            
            # 2. Check specific common names
            targets = ['company_profile', 'company_profiles', 'ticker_list', 'tickers']
            print("\n--- TARGET CHECKS ---")
            for t in targets:
                try:
                    await session.execute(text(f"SELECT ticker FROM {t} LIMIT 1"))
                    print(f"  ✅ {t}: EXISTS")
                except Exception:
                    print(f"  ❌ {t}: NOT FOUND")
                    
    except Exception as e:
        print(f"Critical Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
