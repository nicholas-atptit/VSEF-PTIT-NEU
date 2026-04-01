"""Orchestrator for model training and feature update."""

from __future__ import annotations

from typing import List
from src.data.datasets.loader import DatasetLoader
from src.ml.training.baseline_model import BaselineModel
from src.utils.logging import get_logger

logger = get_logger(__name__)

class TrainingPipeline:
    """End-to-end training workflow: Data -> Features -> Train -> Eval."""

    def __init__(self, symbols: List[str]) -> None:
        self.loader = DatasetLoader(symbols)
        self.trainer = BaselineModel()

    def run(self) -> None:
        """Execute full training pipeline.
        
        Steps: Load -> Split -> Train -> Eval -> Save.
        """
        logger.info("starting_training_pipeline")
        try:
            # 1. Create dataset
            df = self.loader.create_features_labels()
            if df.empty:
                logger.error("training_failed_no_data")
                return
                
            # 2. Temporal split
            train, test = self.loader.temporal_split(df)
            
            # 3. Train
            # Assuming 'target_return_1d' is our label from RegressionLabels
            y_train = train['target_return_1d']
            X_train = train.drop(columns=['target_return_1d', 'timestamp', 'symbol'])
            
            self.trainer.train(X_train, y_train)
            
            # 4. Evaluate
            y_test = test['target_return_1d']
            X_test = test.drop(columns=['target_return_1d', 'timestamp', 'symbol'])
            metrics = self.trainer.evaluate(X_test, y_test)
            
            # 5. Save (using project settings)
            from config.settings import get_settings
            s = get_settings()
            import os
            model_path = os.path.join(s.model_dir, "baseline_regressor.joblib")
            self.trainer.save(model_path)
            
            logger.info("training_pipeline_finished", mae=metrics.get('mae'))
        except Exception as e:
            logger.error("training_pipeline_error", error=str(e))
