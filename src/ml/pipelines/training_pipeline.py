"""Deprecated legacy training pipeline kept only for historical reference."""

from __future__ import annotations

from typing import List

class TrainingPipeline:
    """Deprecated legacy workflow superseded by ``src.ml.trainer.DualModelTrainer``."""

    def __init__(self, symbols: List[str]) -> None:
        raise RuntimeError(
            "TrainingPipeline is a deprecated legacy path and is intentionally blocked. "
            "Use src.ml.trainer.DualModelTrainer.train(...) or "
            "src.ml.trainer.DualModelTrainer.train_explicit_split(...) instead."
        )

    def run(self) -> None:
        """Blocked because this legacy path is not a supported training architecture."""
        raise RuntimeError(
            "TrainingPipeline.run() is blocked because the legacy baseline stack is not supported. "
            "Use DualModelTrainer for all active ML training workflows."
        )
