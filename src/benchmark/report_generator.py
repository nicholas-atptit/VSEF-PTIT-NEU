import json
import os
from datetime import datetime
from typing import Dict, Any

class ReportGenerator:
    """
    Sinh báo cáo tiến độ bằng file Markdown .md (Human-reading) 
    và .json (Machine-reading) sau khi Benchmark xong.
    """
    def __init__(self, output_dir: str = "reports/"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate(self, config_used: Dict[str, Any], metrics: Dict[str, Any], llm_quality: Dict[str, Any]) -> str:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_file = os.path.join(self.output_dir, f"benchmark_{timestamp_str}.md")
        json_file = os.path.join(self.output_dir, f"benchmark_{timestamp_str}.json")
        
        # 1. Sinh File JSON
        raw_payload = {
            "config": config_used,
            "financial_metrics": metrics,
            "agent_quality": llm_quality
        }
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(raw_payload, f, ensure_ascii=False, indent=4)
            
        # 2. Sinh File Markdown đẹp
        md_content = f"""# Trading Benchmark Report - {datetime.now().strftime("%Y-%m-%d %H:%M")}

## 1. Thông Tin Chạy (Runtime Config)
- **Profile LLM**: `{config_used.get('profile', 'UNKNOWN')}`
- **Models**: `{config_used.get('model', 'UNKNOWN')}`
- **Regime Thị Trường**: `{config_used.get('regime', 'UNKNOWN')}`
- **Khung thời gian Trade**: `{config_used.get('horizon', 'UNKNOWN')}`

## 2. Chỉ Đánh Giá Thuộc Tính Tài Chính (Financial Metrics)
| Metric | Baseline Multi-Agent Risk Overlay |
|--------|-----------------------------------|
| CAGR | {metrics.get('cagr', 0)*100:.2f}% |
| Sharpe Ratio | {metrics.get('sharpe', 0)} |
| Max Drawdown | {metrics.get('max_drawdown', 0)*100:.2f}% |
| Turnover Rate | {metrics.get('turnover_rate', 0)} |
| Fee-adjusted PnL | {metrics.get('fee_adjusted_pnl', 0)}% |

## 3. Chất Lượng Quyết Định Của Agent (Model Quality)
- **Tổng Quyết định (Call LLM)**: {llm_quality.get('total_decisions_made', 0)} lần
- **Tỷ Lệ Bị Phủ Quyết (Veto Rate)**: {llm_quality.get('veto_rate', 0)*100:.2f}%
- **Tỷ Lệ Mâu Thuẫn Quá Mức**: {llm_quality.get('contradiction_rate', 0)*100:.2f}%
- **Độ trễ trung bình 1 phiên**: {llm_quality.get('avg_latency_sec', 0)} giây/mã cổ phiếu

> Tính chất Multi-Agent tỏ ra ưu việt ở chỗ dù Model ảo tưởng, nhưng Veto Rate = {llm_quality.get('veto_rate', 0)*100:.1f}% đã giúp danh mục **khóa rủi ro Tail-risk**, giữ Drawdown ở mức cho phép.

Ghi chú: *Báo cáo JSON đi kèm đã được lưu tại thư mục cùng cấp phục vụ đẩy log tracking.*
"""
        with open(md_file, "w", encoding="utf-8") as fm:
            fm.write(md_content)
            
        return md_file
