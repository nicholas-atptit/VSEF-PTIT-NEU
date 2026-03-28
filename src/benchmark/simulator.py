import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import random

from src.agents.orchestrator import TradingOrchestrator
from src.portfolio.weights_schema import PortfolioTarget, AssetWeight
from src.database.decision_card_schema import DecisionCard

class HistoricalSimulator:
    """
    Simulate quá trình giao dịch Multi-Agent trên dữ liệu quá khứ.
    Hoạt động như một 'Time-Travel Engine'.
    """
    def __init__(self, use_llm_provider: str = 'ollama', batch_delay_sec: float = 2.0):
        self.orchestrator = TradingOrchestrator(use_llm_provider=use_llm_provider)
        # Delay để giảm tải RAM cho Ollama Local
        self.batch_delay_sec = batch_delay_sec

    def fetch_historical_data(self, ticker: str, date: datetime) -> Dict[str, Any]:
        """
        Mock: Quét DB / CSV để lấy nến OHLCV và News tại ngày 'date'.
        Bản thật sẽ kết nối TimescaleDB hoặc vnstock CSV_Cache ở đây.
        """
        # Giả lập dữ liệu được fetch về thành công
        return {
            "ticker": ticker,
            "simulated_date": date.isoformat(),
            "close_price": random.uniform(20.0, 50.0),
            "volume": random.randint(100000, 5000000)
        }

    async def run_simulation(
        self, tickers: List[str], start_date: str, end_date: str, horizon: str = "daily"
    ) -> List[DecisionCard]:
        """
        Duyệt qua từng ngày từ start_date -> end_date. Gọi Agentic Debate.
        Lưu output phục vụ việc Evaluator soi chiếu vào PnL thực tế sau đó.
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        results: List[DecisionCard] = []
        current_date = start
        
        while current_date <= end:
            print(f"--- Simulating date: {current_date.strftime('%Y-%m-%d')} ---")
            
            # Xử lý tuần tự thay vì song song để chống nghẽn RAM Ollama local
            for ticker in tickers:
                # 1. Lấy dữ liệu fake của ngày hnay
                inject_data = self.fetch_historical_data(ticker, current_date)
                
                # 2. Chạy Debate
                try:
                    decision_dict = await self.orchestrator.execute_debate(ticker, initial_data=inject_data)
                    card = DecisionCard(
                        meta={
                            "ticker": ticker,
                            "provider": self.orchestrator.provider,
                            "timestamp": current_date, # Gắn nhãn Time-Travel
                            "latency_sec": decision_dict.get("latency_sec", 0)
                        },
                        tech_summary=decision_dict["tech_summary"],
                        news_summary=decision_dict["news_summary"],
                        bull_thesis=decision_dict["bull_thesis"],
                        bear_thesis=decision_dict["bear_thesis"],
                        risk_veto=decision_dict["risk_veto"],
                        risk_reason=decision_dict["risk_reason"],
                        action=decision_dict["action"],
                        target_weight=decision_dict["target_weight"],
                        rationale=decision_dict["rationale"]
                    )
                    results.append(card)
                except Exception as e:
                    print(f"[Error] Simulation fail on {ticker} / {current_date}: {e}")
                    
                # 3. Nghỉ ngơi giữ sức cho RAM LLM local
                time.sleep(self.batch_delay_sec)
                
            # Sang ngày hôm sau (Hoặc tuần sau tùy horizon)
            current_date += timedelta(days=1)
            
        return results
