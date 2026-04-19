"""Pluggable embedding provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class EmbeddingProvider(ABC):
    """Base class for embedding generation (e.g. OpenAI, SentenceTransformers)."""

    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate vectors for a list of texts."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the vector dimension."""
        pass


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic mock provider for testing and smoke validation."""

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Return deterministic dummy vectors based on text length."""
        results = []
        for text in texts:
            # Deterministic but fake
            val = len(text) / 1000.0
            results.append([val] * self.dimension)
        return results

    def get_dimension(self) -> int:
        return self.dimension
