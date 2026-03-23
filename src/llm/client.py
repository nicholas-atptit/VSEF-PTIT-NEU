"""Local LLM Client via OpenAI library.

Connects to the local Ollama instance acting as an OpenAI-compatible endpoint.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from config.settings import get_settings


def get_llm_client() -> AsyncOpenAI:
    """Initialize and return the Async OpenAI client for the active provider."""
    settings = get_settings()
    provider = settings.llm_provider

    if provider == "openai":
        base_url = settings.openai_base_url
        api_key = settings.openai_api_key
    elif provider == "groq":
        base_url = settings.groq_base_url
        api_key = settings.groq_api_key
    elif provider == "gemini":
        base_url = settings.gemini_base_url
        api_key = settings.gemini_api_key
    else:  # Default to Ollama
        base_url = settings.ollama_base_url
        api_key = settings.ollama_api_key

    return AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
    )
