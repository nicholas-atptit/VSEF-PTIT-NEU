from vnstock_data import Listing
import pandas as pd

try:
    listing = Listing(source="VCI")
    print("vnstock_data Listing initialized")
    
    attrs = [a for a in dir(listing) if not a.startswith('_')]
    print(f"Listing attributes: {attrs}")
    
    try:
        df = listing.symbols_by_group("VN100")
        print("Listing.symbols_by_group('VN100') succeeded")
        print(df.head())
    except Exception as e:
        print(f"Listing VN100 failed: {e}")

except Exception as e:
    print(f"vnstock_data Listing init failed: {e}")
