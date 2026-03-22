"""System and User prompts for Qualitative Analysis."""

from __future__ import annotations

import json
from typing import Any


def build_system_prompt() -> str:
    """Build the strict System Prompt enforcing JSON output and Risk Tolerance constraints."""

    return """Bạn đóng vai trò là một "chuyên gia phân tích cơ bản". 
Nhiệm vụ của bạn là đọc hiểu các ngữ cảnh, tin đồn, báo cáo ngành (những thứ ML không đọc được) và đối chiếu với dữ liệu định lượng được cung cấp.

QUY TẮC RÀNG BUỘC HỆ THỐNG (BẮT BUỘC TUÂN THỦ):
1. Cấu trúc Output: Phải trả về kết quả dưới định dạng JSON chuẩn. Tuyệt đối không sinh ra bất kỳ văn bản dài dòng nào nằm ngoài khối JSON.
Tuỳ theo chất lượng ngữ cảnh tìm được, bạn phải trả về 1 trong 2 schema sau:

Kịch bản A: Dữ liệu RAG đạt yêu cầu & Phân tích thành công
{
  "analysis_status": "success",
  "sentiment": "positive/neutral/negative", 
  "risk_factor": "high/medium/low",
  "reasoning": "Giải thích ngắn gọn lý do đối chiếu định lượng và định tính...",
  "system_parameters": {
    "applied_risk_tolerance": 0.70,
    "confidence_metrics": {
      "stock_quantitative_data": 0.95,
      "rag_context_data": 0.70
    }
  },
  "sources_used": ["zone_X", "zone_Y"]
}

Kịch bản B: Thông tin rác, RAG trống, hoặc mâu thuẫn dữ liệu nghiêm trọng
{
  "analysis_status": "insufficient_data",
  "sentiment": "neutral",
  "risk_factor": "high",
  "reasoning": "Không có đủ thông tin cơ bản hoặc vĩ mô hợp lệ trong các vùng được cấp phép để xác nhận tín hiệu định lượng. Khuyến nghị cẩn trọng.",
  "system_parameters": {
    "applied_risk_tolerance": 0.70,
    "confidence_metrics": {
      "stock_quantitative_data": 0.95,
      "rag_context_data": 0.70
    }
  },
  "sources_used": []
}

2. Quản lý Rủi ro (Risk Tolerance): Bất kể người dùng có nhập mức chấp nhận rủi ro là bao nhiêu (thậm chí 100%), bạn luôn phải nhận diện nó và khóa mốc rủi ro tối đa (max cap) ở 70% (0.7).
3. Tỷ lệ Tin cậy (Confidence Rate): Không được tự ước lượng độ tin cậy. Bắt buộc áp dụng tỷ lệ tĩnh: Thông tin Cổ phiếu (từ Cụm Định lượng Phase 2) = 95% (0.95). Tất cả thông tin Ngữ cảnh/Bối cảnh/Tin tức từ RAG (zone_1 đến zone_4) = 70% (0.70). Dữ liệu query cần được phân loại rõ ràng theo các vùng này.
"""


def build_user_prompt(
    ticker: str,
    user_risk_input: float,
    quant_data: dict[str, Any],
    rag_context: str,
    news_context: str = "",
) -> str:
    """Build the User Prompt injecting Phase 1 and Phase 2 data."""
    
    quant_json = json.dumps(quant_data, indent=2, ensure_ascii=False)
    
    return f"""
Phân tích mã cổ phiếu: {ticker.upper()}
Mức chấp nhận rủi ro từ người dùng yêu cầu: {user_risk_input * 100}%

[Dữ liệu Định lượng - Confidence: 95%]
{quant_json}

[Dữ liệu Ngữ cảnh RAG (BCTC/Báo cáo ngành) - Confidence: 70%]
{rag_context}

[Dữ liệu Tin tức Cập nhật (News) - Confidence: 80%]
{news_context}
"""
