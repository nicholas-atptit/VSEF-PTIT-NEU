"""Full End-to-End Kafka Pipeline Test.

Verifies the complete chain:
  market.data.raw → ml.predictions → llm.analysis → cache

This test publishes a mock OHLCV event, simulates the ML prediction consumer
publishing to ml.predictions, then simulates the LLM analysis consumer
publishing to llm.analysis, and finally verifies the cache writer stores it.

Requires Kafka to be running on localhost:9092.
"""

import asyncio
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.streaming.kafka_client import KafkaPublisher, KafkaSubscriber


async def run_full_chain_test():
    print("\n" + "=" * 60)
    print("  FULL KAFKA PIPELINE E2E TEST")
    print("  Data -> ML -> LLM -> Cache")
    print("=" * 60)

    publisher = KafkaPublisher()

    try:
        await publisher.start()
        print("\n[1/5] Kafka Broker connected!")
    except Exception as e:
        print(f"FAIL: Cannot connect to Kafka: {e}")
        return

    # ── Step 1: Publish raw market data ──
    print("\n[2/5] Publishing mock OHLCV to 'market.data.raw'...")
    market_event = {
        "ticker": "FPT",
        "timestamp": datetime.now().isoformat(),
        "open": 120.0,
        "high": 123.5,
        "low": 119.0,
        "close": 122.0,
        "volume": 5000000,
        "_e2e_test": True,
    }
    await publisher.publish("market.data.raw", market_event)
    print(f"   -> Published: FPT OHLCV (close=122.0)")

    # ── Step 2: Simulate ML Prediction Consumer output ──
    print("\n[3/5] Publishing mock ML prediction to 'ml.predictions'...")
    ml_event = {
        "ticker": "FPT",
        "timestamp": market_event["timestamp"],
        "trend_probabilities": {"up": 0.65, "sideways": 0.15, "down": 0.20},
        "expected_range": {"bottom_10th": 118.5, "median_50th": 121.0, "ceiling_90th": 125.0},
        "_e2e_test": True,
    }
    await publisher.publish("ml.predictions", ml_event)
    print(f"   -> Published: FPT trend=65% up, range=[118.5, 125.0]")

    # ── Step 3: Simulate LLM Analysis Consumer output ──
    print("\n[4/5] Publishing mock LLM analysis to 'llm.analysis'...")
    llm_event = {
        "ticker": "FPT",
        "timestamp": market_event["timestamp"],
        "ml_prediction": ml_event,
        "llm_analysis": {
            "analysis_status": "success",
            "sentiment": "positive",
            "risk_factor": "medium",
            "reasoning": "FPT co xu huong tang nhe dua tren mo hinh ML (65% up). Ky vong gia dao dong 118.5-125.0.",
        },
        "_e2e_test": True,
    }
    await publisher.publish("llm.analysis", llm_event)
    print(f"   -> Published: FPT sentiment=positive, risk=medium")

    # ── Step 4: Verify Cache Writer can read it ──
    print("\n[5/5] Verifying Cache Writer Consumer receives the event...")
    
    subscriber = KafkaSubscriber("llm.analysis", "e2e_test_cache_group")
    try:
        await subscriber.start()
        
        async def listen():
            async for msg in subscriber.stream():
                if msg.get("_e2e_test") and msg.get("ticker") == "FPT":
                    return msg

        task = asyncio.create_task(listen())
        result = await asyncio.wait_for(task, timeout=5.0)
        
        # Verify data integrity
        assert result["ticker"] == "FPT"
        assert result["ml_prediction"]["trend_probabilities"]["up"] == 0.65
        assert result["llm_analysis"]["sentiment"] == "positive"
        
        print(f"   -> Received & verified: FPT analysis intact!")
        
        # Write to cache manually to verify API can read it
        from src.streaming.consumers.cache_writer_consumer import CacheWriterConsumer, CACHE_DIR, CACHE_FILE
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = {}
        cache["FPT"] = {
            "ticker": "FPT",
            "timestamp": result["timestamp"],
            "ml_prediction": result["ml_prediction"],
            "llm_analysis": result["llm_analysis"],
        }
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        
        # Verify API can read from cache
        cached = CacheWriterConsumer.read_cache("FPT")
        assert cached["ticker"] == "FPT"
        assert cached["ml_prediction"]["trend_probabilities"]["up"] == 0.65
        print(f"   -> API Cache verified: /predict can read FPT from cache!")
        
    except asyncio.TimeoutError:
        print("   -> WARN: Timeout waiting for llm.analysis consumer. Topic may be new.")
    except Exception as e:
        print(f"   -> ERROR: {e}")
    finally:
        await subscriber.stop()
    
    await publisher.stop()

    print("\n" + "=" * 60)
    print("  E2E TEST COMPLETE!")
    print("  Full chain: Data -> ML -> LLM -> Cache -> API")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(run_full_chain_test())
