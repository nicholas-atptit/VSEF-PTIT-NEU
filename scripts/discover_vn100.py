from vnstock import Vnstock
import pandas as pd

try:
    # Try common 3.x patterns
    vn = Vnstock()
    # 1. Try listing
    try:
        df = vn.market.index_components(symbol='VN100')
        print("index_components method found")
        print(df.head())
    except Exception as e:
        print(f"index_components failed: {e}")

    # 2. Try legacy listing_components if it exists in namespace
    try:
        from vnstock import listing_components
        df = listing_components("VN100")
        print("listing_components function found")
        print(df.head())
    except Exception as e:
        print(f"listing_components failed: {e}")

except Exception as e:
    print(f"Vnstock init failed: {e}")
