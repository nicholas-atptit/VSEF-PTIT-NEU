import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

print("Testing AI Agent imports...")
try:
    from src.ml.signal_generator import SignalGenerator
    print("✅ SignalGenerator imported successfully.")
except Exception as e:
    print(f"❌ SignalGenerator failed: {e}")
    import traceback
    traceback.print_exc()

try:
    from src.ml.llm.news_intel import NewsIntelEngine
    print("✅ NewsIntelEngine imported successfully.")
except Exception as e:
    print(f"❌ NewsIntelEngine failed: {e}")
    import traceback
    traceback.print_exc()

try:
    from src.context.news_crawler import NewsCrawler
    print("✅ NewsCrawler imported successfully.")
except Exception as e:
    print(f"❌ NewsCrawler failed: {e}")
    import traceback
    traceback.print_exc()
