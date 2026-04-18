"""Discover VN100 constituents using vnstock_data.Listing (canonical provider)."""
from vnstock_data import Listing
import pandas as pd

try:
    df = Listing(source="vnd").all_symbols()
    print(f"Total symbols from vnstock_data Listing: {len(df)}")
    print(df.head(10))

    # Try index-specific filter if available
    try:
        df_vn100 = Listing(source="vnd").symbols_by_group("VN100")
        if df_vn100 is not None and not df_vn100.empty:
            print(f"\nVN100 constituents: {len(df_vn100)}")
            print(df_vn100.head())
    except Exception as e:
        print(f"symbols_by_group('VN100') not supported: {e}")

except Exception as e:
    print(f"vnstock_data Listing failed: {e}")
