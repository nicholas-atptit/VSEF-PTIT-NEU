"""Local LLM Client for Ollama.

Provides a clean, minimal interface for generating text using a local Ollama instance.
Isolates the LLM infrastructure from the agents.
"""

from __future__ import annotations

import httpx
import json
from config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

class LocalLLMClient:
    """Client for interacting with local Ollama API."""

    def __init__(
        self, 
        base_url: str | None = None, 
        model: str | None = None,
        timeout: int | None = None
    ) -> None:
        settings = get_settings()
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.llm_model_explainer
        self.timeout = timeout or settings.llm_timeout

    async def generate(self, prompt: str) -> str:
        """Generate a response from the local LLM.
        
        Args:
            prompt: The input text prompt.
            
        Returns:
            The generated string, or an error message if it fails.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 500
            }
        }
        
        # Note: Ollama /v1/completions or /api/generate
        # We use /api/generate for direct Ollama API, but settings might point to /v1
        endpoint = f"{self.base_url.rstrip('/')}/api/generate"
        if "/v1" in self.base_url:
             # If using OpenAI-compatible endpoint in Ollama
             endpoint = f"{self.base_url.rstrip('/')}/completions"
             payload = {
                 "model": self.model,
                 "prompt": prompt,
                 "max_tokens": 500,
                 "temperature": 0.2
             }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                
                data = response.json()
                
                # Check for direct Ollama response format vs OpenAI-compatible format
                if "response" in data:
                    return data["response"].strip()
                elif "choices" in data:
                    return data["choices"][0]["text"].strip()
                else:
                    return "Error: Unexpected LLM response format."
                    
        except httpx.TimeoutException:
            logger.error("llm_client_timeout", model=self.model)
            return "Error: LLM request timed out."
        except Exception as e:
            logger.error("llm_client_error", error=str(e), model=self.model)
            return f"Error: Failed to connect to local LLM: {str(e)}"

# Singleton-style helper
_client = None

def get_local_client() -> LocalLLMClient:
    global _client
    if _client is None:
        _client = LocalLLMClient()
    return _client
