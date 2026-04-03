from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

from config.settings import get_settings
from src.ml.llm.client import get_llm_client
from .contracts import MarketSignal, AnalystDecision, RiskDecision


class ExplainerAgent:
    """Agent that generates natural language explanations for trading decisions.
    
    Uses an LLM (Ollama/OpenAI/Gemini) to synthesize technical signals and 
    risk constraints into a human-readable narrative.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.settings = get_settings()
        self.model_name = model_name or self.settings.llm_model_explainer
        self._client = get_llm_client()

    async def explain(
        self, 
        signal: MarketSignal, 
        analyst: AnalystDecision, 
        risk: RiskDecision
    ) -> str:
        """Generate a markdown explanation for a single ticker decision."""
        if not self.settings.enable_llm_explainer:
            return "LLM Explainer is disabled in settings."

        prompt = self._build_prompt(signal, analyst, risk)
        
        try:
            response = await self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a senior financial analyst and quantitative researcher. Your goal is to explain trading signals clearly and concisely."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            return response.choices[0].message.content or "No explanation generated."
        except Exception as e:
            from src.utils.logging import get_logger
            logger = get_logger(__name__)
            logger.error("explainer_agent_error", ticker=signal.ticker, error=str(e))
            return f"Failed to generate explanation: {str(e)}"

    def _build_prompt(
        self, 
        signal: MarketSignal, 
        analyst: AnalystDecision, 
        risk: RiskDecision
    ) -> str:
        """Construct the prompt from the quantitative context.
        
        Optimized for local 8B models (concise, structured).
        """
        return f"""
Bạn là chuyên gia phân tích tài chính cao cấp. Hãy giải thích quyết định giao dịch sau cho mã {signal.ticker}:

### Dữ liệu Định lượng:
- Giá hiện tại: {signal.current_price:.2f}
- Dự báo (Xác suất): Tăng={signal.trend_up_prob:.2%}, Giảm={signal.trend_down_prob:.2%}, Đi ngang={signal.trend_sideways_prob:.2%}
- Độ tin cậy: {signal.confidence:.2f}
- Biến động (Volatility): {signal.volatility:.4f}
- RSI-14: {signal.rsi_14 or 'N/A'}, SMA-20: {signal.sma_20 or 'N/A'}

### Quyết định của Hệ thống:
- Hành động từ Analyst: {analyst.action}
- Lý do: {', '.join(analyst.reasons)}
- Trạng thái Rủi ro (Risk): {"Được duyệt" if risk.approved else "BỊ TỪ CHỐI"}
- Lý do phủ quyết (Veto): {', '.join(risk.veto_reasons) if risk.veto_reasons else "Không có"}

Yêu cầu:
1. Giải thích ngắn gọn (2-3 đoạn) lý do tại sao hệ thống đưa ra quyết định này.
2. Phân tích các yếu tố kỹ thuật chính và rủi ro được xác định.
3. Sử dụng tông giọng chuyên nghiệp, khách quan.
4. Ngôn ngữ: TIẾNG VIỆT. Định dạng: Markdown.
"""

    async def explain_batch(self, signals: list[MarketSignal], analyst_decisions: list[AnalystDecision], risk_decisions: list[RiskDecision]) -> list[str]:
        """Process multiple signals in parallel."""
        tasks = [
            self.explain(s, a, r) 
            for s, a, r in zip(signals, analyst_decisions, risk_decisions)
        ]
        return await asyncio.gather(*tasks)
