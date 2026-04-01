from vnstock import Vnstock

def explore():
    v = Vnstock()
    s = v.stock(symbol='HPG')
    print(f"🚀 Exploring v.stock('HPG'):\n{dir(s)}")
    print(f"\n🚀 Exploring v.stock('HPG').listing:\n{dir(s.listing)}")
    
    try:
        info = s.listing.info()
        print(f"\n🚀 s.listing.info() type: {type(info)}")
        print(info)
    except Exception as e:
        print(f"\n❌ s.listing.info() failed: {e}")

if __name__ == "__main__":
    explore()
