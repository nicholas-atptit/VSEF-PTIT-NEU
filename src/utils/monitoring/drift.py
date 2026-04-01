"""Feature Drift Detection Engine (Phase 5).

Monitoring statistical shifts in market features to detect model degradation.
"""

import numpy as np
import pandas as pd
from typing import Any
from src.utils.logging import get_logger

logger = get_logger(__name__)

class DriftMonitor:
    def __init__(self, threshold_z: float = 3.0):
        self.threshold_z = threshold_z
        self.baseline_stats = {} # feature_name -> (mean, std)

    def update_baseline(self, df: pd.DataFrame):
        """Update statistics from a known 'good' or training distribution."""
        for col in df.select_dtypes(include=[np.number]).columns:
            self.baseline_stats[col] = (df[col].mean(), df[col].std())
        logger.info("drift_baseline_updated", features=list(self.baseline_stats.keys()))

    def check_drift(self, latest_features: pd.Series) -> dict[str, Any]:
        """Check if the latest feature vector has drifted from baseline."""
        drift_report = {
            "is_drifted": False,
            "drifted_features": [],
            "max_z": 0.0
        }
        
        for feat, val in latest_features.items():
            if feat in self.baseline_stats:
                mu, sigma = self.baseline_stats[feat]
                if sigma > 0:
                    z = abs(val - mu) / sigma
                    if z > self.threshold_z:
                        drift_report["is_drifted"] = True
                        drift_report["drifted_features"].append(feat)
                        drift_report["max_z"] = max(drift_report["max_z"], z)
        
        if drift_report["is_drifted"]:
            logger.warning("feature_drift_detected", 
                           features=drift_report["drifted_features"],
                           max_z=round(drift_report["max_z"], 2))
            
        return drift_report

def calculate_psi(expected, actual, buckets=10):
    """Calculate Population Stability Index (PSI)."""
    # Placeholder for more advanced drift monitoring
    return 0.0
