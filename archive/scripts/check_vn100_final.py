from vnstock import Vnstock
import pandas as pd

try:
    vn = Vnstock()
    # Try getting VN100 components
    try:
        s = vn.stock(symbol="VN100", source="VCI")
        df = s.listing.components()
        print("VN100 components found via s.listing.components()")
        print(df.head())
        print(f"Total tickers: {len(df)}")
    except Exception as e:
        print(f"VN100 components failed: {e}")

except Exception as e:
    print(f"Vnstock init failed: {e}")
