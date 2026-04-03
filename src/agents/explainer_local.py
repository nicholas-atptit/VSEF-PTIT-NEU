"""Local Explainer Agent.

Generates natural language explanations using the local LLM.
Optimized for the low-resource branch.
"""

from __future__ import annotations

import asyncio
from typing import Any

from config.settings import get_settings
from src.llm.local_client import get_local_client
from .contracts import MarketSignal, RiskDecision, PortfolioProposal


class LocalExplainerAgent:
    """Agent specialized in generating local LLM explanations for trading decisions."""

    def __init__(self, model_name: str | None = None) -> None:
        self.settings = get_settings()
        self._client = get_local_client()

    async def explain(
        self, 
        signal: MarketSignal, 
        risk: RiskDecision, 
        portfolio: PortfolioProposal
    ) -> str:
        """Generate a structured, Vietnamese explanation for a trading decision.
        
        Inputs are strictly from the deterministic core.
        """
        if not self.settings.enable_llm_explainer:
            return "Local LLM Explainer is disabled in settings."

        prompt = self._build_prompt(signal, risk, portfolio)
        
        # Execute explanation with fallback
        try:
            explanation = await self._client.generate(prompt)
            # Basic validation of response
            if explanation.startswith("Error:"):
                return f"Explain-Only Fallback: Trading system decision made, but explanation generation failed. ({explanation})"
            return explanation
        except Exception as e:
            from src.utils.logging import get_logger
            logger = get_logger(__name__)
            logger.error("local_explainer_agent_error", ticker=signal.ticker, error=str(e))
            return f"Fallback: Explanation service unavailable. [{str(e)}]"

    def _build_prompt(
        self, 
        signal: MarketSignal, 
        risk: RiskDecision, 
        portfolio: PortfolioProposal
    ) -> str:
        """Construct a structured, professional prompt for the local 8B model.
        
        Explicitly requests Vietnamese output and Markdown formatting.
        """
        # Find the specific proposal for this ticker if available
        proposal = next((p for p in portfolio.positions if p.ticker == signal.ticker), None)
        proposal_weight = proposal.weight if proposal else 0.0
        
        return f"""
Bạn là một chuyên gia phân tích tài chính và quản trị rủi ro cao cấp. 
Hãy giải thích quyết định giao dịch cho mã cổ phiếu {signal.ticker} dựa trên dữ liệu định lượng sau:

### 1. Tín hiệu Thị trường (Market Signal):
- Giá hiện tại: {signal.current_price:.2f}
- Dự báo kỹ thuật: Tăng={signal.trend_up_prob:.1%}, Giảm={signal.trend_down_prob:.1%}, Đi ngang={signal.trend_sideways_prob:.1%}
- Độ tin cậy (Confidence): {signal.confidence:.2f}
- Biến động (Volatility): {signal.volatility:.4f}
- Chỉ số RSI: {signal.rsi_14 or 'N/A'}
- Đường SMA-20: {signal.sma_20 or 'N/A'}

### 2. Đánh giá Rủi ro (Risk Decision):
- Trạng thái Duyệt: {"ĐƯỢC CHẤP THUẬN" if risk.approved else "BỊ TỪ CHỐI"}
- Kích thước vị thế tối đa gợi ý: {risk.position_size_pct:.1%}
- Điểm cắt lỗ (Stop Loss): {risk.stop_loss_pct:.1%}
- Điểm chốt lời (Take Profit): {risk.take_profit_pct:.1%}
- Lý do veto (nếu có): {', '.join(risk.veto_reasons) if risk.veto_reasons else "Không có"}

### 3. Đề xuất Danh mục (Portfolio Proposal):
- Tỉ trọng phân bổ thực tế: {proposal_weight:.1%}
- Ghi chú chiến lược: {portfolio.notes[0] if portfolio.notes else "N/A"}

### YÊU CẦU:
Viết một đoạn giải thích ngắn gọn (200-300 từ) bằng TIẾNG VIỆT nhằm trả lời các câu hỏi sau:
1. Tại sao hệ thống lại đưa ra hành động này (Mua/Bán/Phứng)?
2. Các yếu tố kỹ thuật nào là động lực chính?
3. Rủi ro nào đã được hệ thống rà soát và xử lý (hoặc từ chối)?

Sử dụng định dạng Markdown, tông giọng chuyên nghiệp, khách quan và súc tích.
"""

    async def explain_batch(
        self, 
        signals: list[MarketSignal], 
        risk_decisions: list[RiskDecision], 
        portfolio: PortfolioProposal
    ) -> list[str]:
        """Process local explanations in parallel."""
        tasks = [
            self.explain(s, r, portfolio) 
            for s, r in zip(signals, risk_decisions)
        ]
        return await asyncio.gather(*tasks)
