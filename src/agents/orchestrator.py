import time
import uuid
import asyncio
from typing import Dict, Any, List

from .technical_agent import TechnicalAgent
from .news_macro_agent import NewsMacroAgent
from .bull_researcher import BullResearcherAgent
from .bear_researcher import BearResearcherAgent
from .risk_manager_agent import RiskManagerAgent
from .portfolio_manager_agent import PortfolioManagerAgent

class TradingOrchestrator:
    """
    Quản lý luồng thực thi: Parallel run (Technical + News) -> Debate (Bull vs Bear) -> Veto (Risk) -> Allocation (Portfolio).
    Đây là Entrypoint cho Multi-Agent Graph.
    """
    def __init__(self, use_llm_provider: str = 'ollama'):
        self.provider = use_llm_provider
        self.technical = TechnicalAgent()
        self.news = NewsMacroAgent()
        self.bull = BullResearcherAgent()
        self.bear = BearResearcherAgent()
        self.risk = RiskManagerAgent()
        self.portfolio = PortfolioManagerAgent()

    async def execute_debate(self, ticker: str, initial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Thực thi toàn bộ Graph: Technical & News -> Bull & Bear -> Risk -> Portfolio.
        Returns một DecisionCard nháp (dạng dict).
        """
        initial_data = initial_data or {"ticker": ticker}
        start_time = time.time()
        
        # 1. Thu thập chứng cứ đồng thời (Parallel)
        tech_task = asyncio.create_task(self.technical.analyze(initial_data))
        news_task = asyncio.create_task(self.news.analyze(initial_data))
        tech_res, news_res = await asyncio.gather(tech_task, news_task)

        context_data = {
            "technical": tech_res,
            "news": news_res,
            "ticker": ticker
        }

        # 2. Debate (Parallel Bull/Bear phase)
        # Trong bản đồ thực tế có thể có nhiều vòng, ở đây V1 chỉ 1 vòng
        bull_task = asyncio.create_task(self.bull.analyze(context_data))
        bear_task = asyncio.create_task(self.bear.analyze(context_data))
        bull_res, bear_res = await asyncio.gather(bull_task, bear_task)

        debate_context = {
            "bull_thesis": bull_res,
            "bear_thesis": bear_res,
            "evidence": context_data
        }

        # 3. Risk Veto
        risk_res = await self.risk.analyze(debate_context)
        
        # 4. Portfolio Allocation / Final Decision
        portfolio_context = {
            "debate": debate_context,
            "risk_decision": risk_res
        }
        final_res = await self.portfolio.analyze(portfolio_context)
        
        end_time = time.time()

        # Tạo format nháp của DecisionCard
        decision_card = {
            "decision_id": str(uuid.uuid4()),
            "ticker": ticker,
            "timestamp": time.time(),
            "provider": self.provider,
            "tech_summary": tech_res,
            "news_summary": news_res,
            "bull_thesis": bull_res["thesis"],
            "bear_thesis": bear_res["thesis"],
            "risk_veto": risk_res["veto"],
            "risk_reason": risk_res["reason"],
            "action": final_res["action"],
            "target_weight": final_res["target_weight"],
            "rationale": final_res["rationale"],
            "latency_sec": round(end_time - start_time, 2)
        }
        
        return decision_card
