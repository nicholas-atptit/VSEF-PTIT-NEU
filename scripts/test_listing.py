from vnstock import Vnstock
import pandas as pd

def test_listing():
    v = Vnstock()
    print("🚀 Testing All Listing Info...")
    try:
        # Try different listing methods
        print("1. v.market.listing()...")
        try:
            df = v.market.listing()
            print(f"✅ Success: {len(df)} rows")
            print(df.head(2))
            return
        except Exception as e: print(f"❌ Fail: {e}")

        print("2. v.stock(symbol='HPG').listing.info()...")
        try:
            df = v.stock(symbol='HPG').listing.info()
            print(f"✅ Success: {df}")
        except Exception as e: print(f"❌ Fail: {e}")

        print("3. v.market.all_symbols()...")
        try:
            df = v.all_symbols()
            print(f"✅ Success: {len(df)} rows")
        except Exception as e: print(f"❌ Fail: {e}")

    except Exception as e:
        print(f"❌ Global Error: {e}")

if __name__ == "__main__":
    test_listing()
