from vnstock_data import Listing
import pandas as pd

try:
    listing = Listing(source="VCI")
    print("vnstock_data Listing initialized")
    
    try:
        df = listing.symbols_by_group("VN100")
        print("Listing.symbols_by_group found")
        print(df.head())
    except Exception as e:
        print(f"symbols_by_group failed: {e}")

    try:
        df = listing.all_symbols()
        print("Listing.all_symbols found")
        print(df.head())
    except Exception as e:
        print(f"all_symbols failed: {e}")

except Exception as e:
    print(f"vnstock_data Listing init failed: {e}")
