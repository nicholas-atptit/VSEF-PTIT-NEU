"""Local LLM Client via OpenAI library.

Connects to the local Ollama instance acting as an OpenAI-compatible endpoint.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from config.settings import get_settings


def get_llm_client() -> AsyncOpenAI:
    """Initialize and return the Async OpenAI client pointing to Ollama."""
    settings = get_settings()

    return AsyncOpenAI(
        base_url=settings.ollama_base_url,
        api_key=settings.ollama_api_key,
    )
