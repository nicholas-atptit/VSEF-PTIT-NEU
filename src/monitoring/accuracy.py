"""Online Accuracy Tracking Engine (Phase 5).

Compares historical agent predictions against realized market price moves.
"""

import datetime as dt
import pandas as pd
from typing import Any
from typing import Dict, Any
from sqlalchemy import text
from src.database.connection import get_db
from src.utils.logging import get_logger

logger = get_logger(__name__)

class AccuracyMonitor:
    def __init__(self):
        self.db = get_db()

    def calculate_recent_accuracy(self, ticker: str, horizon: str = "1w") -> Dict[str, Any]:
        """Calculate directional accuracy for the last N realized predictions."""
        # 1. Define the 'realization' window
        # For '1w', we look for predictions made between 8 days and 6 days ago.
        now = dt.datetime.now(dt.UTC)
        lookback_days = 7 if horizon == "1w" else 30 if horizon == "1m" else 1
        
        start_date = now - dt.timedelta(days=lookback_days + 1)
        end_date = now - dt.timedelta(days=lookback_days - 1)
        
        query = text("""
            SELECT p.ticker, p.created_at as pred_time, ap.trend as predicted_trend,
                   price_start.close as price_at_pred, price_end.close as price_now
            FROM agent_runs p
            JOIN agent_predictions ap ON p.id = ap.run_id
            JOIN raw_prices price_start ON p.ticker = price_start.ticker 
                 AND ABS(EXTRACT(EPOCH FROM (p.created_at - price_start.timestamp))) < 3600
            JOIN raw_prices price_end ON p.ticker = price_end.ticker 
                 AND price_end.timestamp > :start_time AND price_end.timestamp < :end_time
            WHERE p.ticker = :ticker AND ap.horizon = :horizon
            ORDER BY p.created_at DESC
            LIMIT 20
        """)
        
        try:
            with self.db.connect() as conn:
                df = pd.read_sql(query, conn, params={
                    "ticker": ticker,
                    "horizon": horizon,
                    "start_time": now - dt.timedelta(hours=24),
                    "end_time": now
                })
            
            if df.empty:
                return {"accuracy": 0.0, "sample_size": 0}
            
            # Calculate hit rate
            def is_hit(row):
                actual_move = row['price_now'] - row['price_at_pred']
                if row['predicted_trend'] == 'UP' and actual_move > 0: return 1
                if row['predicted_trend'] == 'DOWN' and actual_move < 0: return 1
                if row['predicted_trend'] == 'SIDE' and abs(actual_move/row['price_at_pred']) < 0.01: return 1
                return 0
            
            df['hit'] = df.apply(is_hit, axis=1)
            acc = df['hit'].mean()
            
            return {
                "accuracy": round(float(acc), 2),
                "sample_size": len(df),
                "horizon": horizon
            }
            
        except Exception as e:
            logger.error("accuracy_calc_failed", ticker=ticker, error=str(e))
            return {"accuracy": 0.0, "sample_size": 0}
