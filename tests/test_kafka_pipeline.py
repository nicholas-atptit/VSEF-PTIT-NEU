"""Integration test for the core Kafka Pipeline (Producers and Consumers).

This script verifies that:
1. Producer can connect to broker and send messages.
2. Consumers can connect, subscribe to topics, and receive messages.
3. The JSON serialization/deserialization works correctly.

Requires Kafka to be running on localhost:9092.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

import pytest

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

pytest.importorskip("aiokafka")

from src.api.streaming.kafka_client import KafkaPublisher, KafkaSubscriber

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TEST_TOPIC = "test.kafka.pipeline"


async def run_test():
    """Run a simple produce/consume test."""
    print("\n" + "=" * 50)
    print("🚀 Bắt đầu test luồng Kafka Pipeline")
    print("=" * 50)

    publisher = KafkaPublisher()
    subscriber = KafkaSubscriber(TEST_TOPIC, group_id="test_group_1")

    # 1. Start Publisher & Subscriber
    try:
        print("\n⏳ 1. Đang kết nối tới Kafka Broker (localhost:9092)...")
        await publisher.start()
        await subscriber.start()
        print("✅ Kết nối Broker thành công!")
    except Exception as e:
        print(f"❌ Kết nối Broker thất bại: {e}")
        print("\n💡 LƯU Ý: Bác cần phải mở Docker Desktop và chạy lệnh: `docker compose up -d` trước khi test.")
        sys.exit(1)

    # 2. Produce a mock market data message
    print("\n⏳ 2. Đang đóng vai trò Data Producer (Gửi dữ liệu HPG)...")
    mock_event = {
        "ticker": "HPG",
        "timestamp": datetime.now().isoformat(),
        "open": 30.5,
        "high": 31.0,
        "low": 30.0,
        "close": 30.8,
        "volume": 15000000,
        "_test_flag": True
    }
    await publisher.publish(TEST_TOPIC, mock_event)
    print(f"✅ Đã gửi event: {mock_event}")

    # 3. Consume the message
    print("\n⏳ 3. Đang đóng vai trò Consumer (Lắng nghe dữ liệu HPG)...")
    received = False
    try:
        # Listen for max 5 seconds
        async def listen():
            async for msg in subscriber.stream():
                if msg.get("_test_flag"):
                    return msg

        task = asyncio.create_task(listen())
        msg = await asyncio.wait_for(task, timeout=5.0)
        print(f"✅ Đã nhận thành công event: {msg}")
        received = True
        
        if msg["ticker"] == "HPG" and msg["close"] == 30.8:
            print("✅ Dữ liệu Parse JSON giữ nguyên định dạng!")
            
    except asyncio.TimeoutError:
        print("❌ Timeout: Không nhận được dữ liệu trả về từ Broker. Có thể topic chưa được khởi tạo.")
    except Exception as e:
        print(f"❌ Lỗi Consumer: {e}")

    # 4. Cleanup
    await publisher.stop()
    await subscriber.stop()

    if received:
        print("\n" + "=" * 50)
        print("🎉 TEST THÀNH CÔNG: Luồng Kafka đã hoạt động mượt mà!")
        print("=" * 50 + "\n")
    else:
        print("\n" + "=" * 50)
        print("⚠️ TEST THẤT BẠI: Vui lòng kiểm tra lại log.")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(run_test())
