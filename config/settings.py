"""Centralized configuration using Pydantic Settings.

All settings are loaded from environment variables or .env file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── DNSE API ──────────────────────────────────────────────
    dnse_api_key: str = ""
    dnse_api_secret: str = ""
    dnse_base_url: str = "https://openapi.dnse.com.vn"

    # ── Vnstock API ───────────────────────────────────────────
    vnstock_api_key: str = ""

    # Fast In-Memory Broker (Phase 1 Upgrades)
    redis_url: str = "redis://localhost:6379"

    # ── TimescaleDB ──────────────────────────────────────────
    timescale_host: str = "localhost"
    timescale_port: int = 5432
    timescale_db: str = "algo_trading"
    timescale_user: str = "postgres"
    timescale_password: str = "your_password_here"

    @property
    def timescale_url(self) -> str:
        """Construct async DB URL safely from components."""
        return f"postgresql+asyncpg://{self.timescale_user}:{self.timescale_password}@{self.timescale_host}:{self.timescale_port}/{self.timescale_db}"

    @property
    def timescale_sync_url(self) -> str:
        """Construct sync DB URL safely from components."""
        return f"postgresql+psycopg2://{self.timescale_user}:{self.timescale_password}@{self.timescale_host}:{self.timescale_port}/{self.timescale_db}"

    # ── ChromaDB ─────────────────────────────────────────────
    chroma_host: str = "localhost"
    chroma_port: int = 8000

    # ── Streaming (Module 1) ─────────────────────────────────
    ping_interval_seconds: float = 1.0
    reconnect_delay_seconds: float = 5.0
    max_reconnect_attempts: int = 10

    # -- Kafka Message Broker (Phase 6) --
    kafka_broker_url: str = "localhost:9092"

    # ── Trading Session Times (Vietnam HH:MM) ────────────────
    morning_open: str = "08:45"
    morning_close: str = "11:30"
    afternoon_open: str = "12:45"
    afternoon_close: str = "15:00"

    # ── Filter Engine (Module 2) ─────────────────────────────
    blacklist_refresh_interval_minutes: int = 5
    indicator_buffer_size: int = 200

    # RSI / MACD / SMA parameters
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    sma_periods: list[int] = [20, 50, 200]

    # ── Historical Backdate (Module 3) ───────────────────────
    backdate_rate_limit_delay: float = 0.5
    backdate_batch_size: int = 100
    backdate_start_year: int = 2014

    # ── Logging ──────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"

    # ── Embedding Model (Module 4) ───────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    bctc_data_dir: str = str(PROJECT_ROOT / "data/bctc")
    rag_top_k_per_zone: int = 5
    rag_news_lookback_days: int = 1095
    latency_sla_seconds: float = 5.0

    # ── ML Pipeline (Phase 2) ────────────────────────────────
    model_dir: str = str(PROJECT_ROOT / "models")
    trend_threshold_pct: float = 2.0
    trend_lookahead_days: int = 3
    max_risk_tolerance: float = 0.70
    confidence_stock_quantitative: float = 0.95
    confidence_general_context: float = 0.70

    # ── Label Engineering ────────────────────────────────────
    label_cls_1d_threshold: float = 0.01   # ±1 % for 1-day 3-class
    label_cls_5d_threshold: float = 0.02   # ±2 % for 5-day 3-class

    # ── LLM Pipeline (Phase 3 & 4 Upgrade) ────────────────────
    llm_provider: Literal["ollama", "openai", "groq", "gemini"] = "gemini"
    
    # Ollama (Local)
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_api_key: str = "" # Default is empty
    ollama_model_name: str = "qwen2.5:7b"
    
    # OpenAI
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model_name: str = "gpt-4o"
    
    # Groq (Llama 3/3.3)
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model_name: str = "llama-3.3-70b-versatile"
    
    # Gemini (via OpenAI-compatible adapter or direct API)
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    gemini_model_name: str = "gemini-2.0-flash"

    # ── Agentic Architecture (Phase 1 Master Plan) ──────────
    sentiment_enabled: bool = True
    fusion_weight_technical: float = 0.6
    fusion_weight_sentiment: float = 0.4
    risk_budget_total: float = 1.0  # Max portfolio exposure
    
    # Horizon Definitions
    short_horizon_days: int = 5
    mid_horizon_days: int = 20
    long_horizon_days: int = 120

    # ── TUI & Monitoring ─────────────────────────────────────
    terminal_refresh_ms: int = 500
    prediction_loop_seconds: int = 15

    # ── API (Phase 2 & 3) ────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8888

    @property
    def chroma_url(self) -> str:
        return f"http://{self.chroma_host}:{self.chroma_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance (singleton)."""
    return Settings()
