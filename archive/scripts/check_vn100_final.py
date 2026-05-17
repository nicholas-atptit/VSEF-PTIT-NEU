from vnstock_data import Listing
import pandas as pd

try:
    listing = Listing(source="VCI")
    try:
        df = listing.symbols_by_group("VN100")
        print("VN100 components found via Listing.symbols_by_group()")
        print(df.head())
        print(f"Total tickers: {len(df)}")
    except Exception as e:
        print(f"VN100 components failed: {e}")

except Exception as e:
    print(f"vnstock_data Listing init failed: {e}")
