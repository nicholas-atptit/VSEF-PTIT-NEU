from vnstock_data import Listing


def explore():
    listing = Listing(source="VCI")
    print(f"Exploring vnstock_data Listing:\n{dir(listing)}")

    try:
        info = listing.all_symbols()
        print(f"\nListing.all_symbols() type: {type(info)}")
        print(info)
    except Exception as e:
        print(f"\nListing.all_symbols() failed: {e}")


if __name__ == "__main__":
    explore()
