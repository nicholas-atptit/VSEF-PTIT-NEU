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

        # 2. Debate Round 1
        bull_task_r1 = asyncio.create_task(self.bull.analyze(context_data))
        bear_task_r1 = asyncio.create_task(self.bear.analyze(context_data))
        bull_res_r1, bear_res_r1 = await asyncio.gather(bull_task_r1, bear_task_r1)

        # 3. Debate Round 2 (Injecting history)
        context_data_bull_r2 = context_data.copy()
        context_data_bull_r2["previous_round"] = {"bear_thesis": bear_res_r1["thesis"]}
        
        context_data_bear_r2 = context_data.copy()
        context_data_bear_r2["previous_round"] = {"bull_thesis": bull_res_r1["thesis"]}

        bull_task_r2 = asyncio.create_task(self.bull.analyze(context_data_bull_r2))
        bear_task_r2 = asyncio.create_task(self.bear.analyze(context_data_bear_r2))
        bull_res, bear_res = await asyncio.gather(bull_task_r2, bear_task_r2)

        debate_context = {
            "bull_thesis": bull_res,
            "bear_thesis": bear_res,
            "evidence": context_data
        }

        # 4. Risk Veto
        risk_res = await self.risk.analyze(debate_context)
        
        # 5. Accuracy Layer Calculation
        from src.ml.accuracy.regime_detector import RegimeDetector
        from src.ml.accuracy.signal_consensus import SignalConsensus
        from src.ml.accuracy.decision_thresholds import DecisionThresholds
        
        regime_label = RegimeDetector.detect_regime(tech_res)
        threshold = DecisionThresholds.get_dynamic_threshold(regime_label)
        
        # Giả lập extract điểm số cho SignalConsensus từ response
        tech_score = tech_res.get("score", 0.5) if isinstance(tech_res, dict) else 0.5
        news_score = news_res.get("sentiment_score", 0.5) if isinstance(news_res, dict) else 0.5
        
        consensus_score = SignalConsensus.calculate_consensus(
            tech_score=tech_score,
            news_score=news_score,
            risk_veto=risk_res.get("veto", False)
        )
        
        # 6. Portfolio Allocation / Final Decision
        portfolio_context = {
            "debate": debate_context,
            "risk_decision": risk_res,
            "threshold": threshold,
            "consensus": consensus_score
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
            "evidence_ids": [f"TECH-{start_time}", f"NEWS-{start_time}"],
            "consensus_score": consensus_score,
            "regime_label": regime_label,
            "dynamic_confidence_threshold": threshold,
            "bull_thesis": bull_res["thesis"],
            "bear_thesis": bear_res["thesis"],
            "risk_veto": risk_res["veto"],
            "risk_reason": risk_res["reason"],
            "action": final_res["action"],
            "target_weight": final_res["target_weight"],
            "rationale": final_res["rationale"],
            "latency_sec": round(end_time - start_time, 2),
            "confidence": getattr(final_res, "confidence", 0.8)
        }
        
        return decision_card
