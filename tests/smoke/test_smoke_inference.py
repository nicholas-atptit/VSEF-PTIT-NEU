"""Quick smoke test for training and inference."""

from pathlib import Path
import tempfile
import pandas as pd
from src.ml.trainer import DualModelTrainer
from src.ml.data_loader import generate_mock_data
import json

# Create a temp directory for models
with tempfile.TemporaryDirectory() as tmpdir:
    # Train CART
    trainer = DualModelTrainer(model_dir=tmpdir)
    df = generate_mock_data(ticker='TEST', num_days=900)
    
    result = trainer.train(
        ticker='TEST',
        df=df,
        algorithms=['cart'],
        horizons=['short'],
        max_depth=3,
    )
    
    print(f'✅ Training successful: {result["ticker"]} with {len(result["report_rows"])} report rows')
    
    # Test inference
    features = trainer.compute_features_for_ticker('TEST', df)
    pred = trainer.predict('TEST', features, horizon='short')
    
    print(f'✅ Inference successful: algorithm={pred["algorithm"]}')
    print(f'   Prediction keys: {list(pred.keys())}')
    print(f'   Trend probabilities: {pred.get("trend_probabilities")}')
    print(f'   Expected range: {pred.get("expected_range")}')
    print('✅✅✅ All smoke tests passed!')
