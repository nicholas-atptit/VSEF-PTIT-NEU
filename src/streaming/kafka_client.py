"""Base Async Kafka Client for Publishers and Consumers.

Provides wrapper classes around aiokafka to standardize messaging
across the Algo Trading infrastructure.
"""

import json
from typing import Any, AsyncGenerator

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class KafkaPublisher:
    """Async wrapper for AIOKafkaProducer."""

    def __init__(self) -> None:
        settings = get_settings()
        # Default fallback to localhost if missing in settings
        broker_url = getattr(settings, "kafka_broker_url", "localhost:9092")
        self.producer = AIOKafkaProducer(
            bootstrap_servers=broker_url,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        self._started = False

    async def start(self) -> None:
        """Start the Kafka producer."""
        if not self._started:
            await self.producer.start()
            self._started = True
            logger.info("kafka_publisher_started")

    async def stop(self) -> None:
        """Stop the Kafka producer."""
        if self._started:
            await self.producer.stop()
            self._started = False
            logger.info("kafka_publisher_stopped")

    async def publish(self, topic: str, message: dict[str, Any]) -> None:
        """Publish a JSON dictionary message to a specific topic."""
        if not self._started:
            await self.start()
            
        try:
            await self.producer.send_and_wait(topic, message)
            logger.debug(f"kafka_message_published", topic=topic)
        except Exception as e:
            logger.error(f"kafka_publish_error", topic=topic, error=str(e))


class KafkaSubscriber:
    """Async wrapper for AIOKafkaConsumer."""

    def __init__(self, topic: str, group_id: str) -> None:
        settings = get_settings()
        broker_url = getattr(settings, "kafka_broker_url", "localhost:9092")
        self.topic = topic
        self.consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=broker_url,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=False, # We commit manually after successful processing
        )
        self._started = False

    async def start(self) -> None:
        """Start the Kafka consumer."""
        if not self._started:
            await self.consumer.start()
            self._started = True
            logger.info("kafka_subscriber_started", topic=self.topic)

    async def stop(self) -> None:
        """Stop the Kafka consumer."""
        if self._started:
            await self.consumer.stop()
            self._started = False
            logger.info("kafka_subscriber_stopped", topic=self.topic)

    async def stream(self) -> AsyncGenerator[dict[str, Any], None]:
        """Stream messages continuously from the topic."""
        if not self._started:
            await self.start()

        try:
            async for msg in self.consumer:
                yield msg.value
                # Commit offset after yielding so if the processor fails, it won't commit
                await self.consumer.commit()
        except Exception as e:
            logger.error("kafka_consume_error", error=str(e))
        finally:
            await self.stop()
