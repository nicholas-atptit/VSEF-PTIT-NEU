"""Lightweight experiment tracking for ML training runs.

Log training metadata and metrics to a JSONL file for versioning and comparison.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import uuid

from src.utils.logging import get_logger

logger = get_logger(__name__)

class ExperimentTracker:
    """Tracks ML experiment metadata and results.
    
    Default storage: reports/experiments.jsonl
    """

    def __init__(self, storage_path: str = "reports/experiments.jsonl"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._run_id = str(uuid.uuid4())[:8]

    def log_experiment(
        self,
        ticker: str,
        label_type: str,
        model_type: str,
        feature_count: int,
        metrics: Dict[str, Any],
        train_start: str,
        train_end: str,
        model_path: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Log a single training experiment run.
        
        Args:
            ticker: Stock ticker symbol.
            label_type: Type of label used (e.g., binary, regression).
            model_type: ML model architecture.
            feature_count: Number of features used.
            metrics: Dictionary of performance metrics.
            train_start: ISO timestamp for training start.
            train_end: ISO timestamp for training end.
            model_path: Path to the saved model artifact.
            params: Hyperparameters used.
            tags: Additional metadata tags.
            
        Returns:
            str: The run ID.
        """
        experiment_data = {
            "run_id": self._run_id,
            "ticker": ticker,
            "label_type": label_type,
            "model_type": model_type,
            "feature_count": feature_count,
            "metrics": metrics,
            "train_start": train_start,
            "train_end": train_end,
            "model_path": model_path,
            "params": params or {},
            "tags": tags or {},
            "created_at": datetime.utcnow().isoformat() + "Z"
        }

        try:
            with open(self.storage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(experiment_data) + "\n")
            
            logger.info(
                "experiment_logged",
                run_id=self._run_id,
                ticker=ticker,
                label=label_type,
                metrics=metrics
            )
        except Exception as e:
            logger.error("experiment_log_failed", error=str(e), ticker=ticker)

        return self._run_id

    @property
    def run_id(self) -> str:
        return self._run_id
