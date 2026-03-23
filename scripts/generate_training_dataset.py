"""Generate high-quality instruction tuning dataset for Risk Analyst SOP.

This script creates synthetic (but grounded) training pairs in JSONL format
for fine-tuning LLMs (Llama-3, Qwen, etc.) to follow the Phase 8 SOP.
"""

import json
import os
import random
from datetime import datetime

# ── Training Format Sample (ShareGPT / OpenAI) ───────────────────────

def generate_sample_success():
    """Generates a high-quality 'success' case."""
    ticker = "FPT"
    
    # Input Data (Zone 1-3)
    instruction = "Bạn là Một HỆ THỐNG PHÂN TÍCH ĐỊNH TÍNH (Qualitative Risk Analyst Engine). CHỈ sử dụng thông tin trong <ZONE_DATA>. Không dự đoán giá."
    
    input_data = """<ZONE_DATA>
[Zone 1: Fundamental/RAG]
FPT công bố lợi nhuận quý 4 tăng trưởng 20%. Mảng xuất khẩu phần mềm sang Nhật Bản đạt kỷ lục.
[Zone 2: Latest News]
Tin tức: FPT ký kết hợp đồng 100 triệu USD với đối tác Mỹ.
[Zone 3: Quantitative/Macro Background]
Xác suất ML hiện tại: {'up': 0.75, 'sideways': 0.15, 'down': 0.1}
Dự phóng ML hiện tại: {'bottom_10th': 120, 'median_50th': 125, 'ceiling_90th': 130}
</ZONE_DATA>"""

    # Ideal Completion
    output = {
        "analysis_status": "success",
        "confidence_score": 0.95,
        "veto_flag": False,
        "overall_outlook": "positive",
        "reasoning": "Tăng trưởng lợi nhuận 20% và hợp đồng 100 triệu USD là minh chứng cho sức mạnh nội tại cực tốt.",
        "signals": {
            "bullish": [
                {"evidence": "Lợi nhuận quý 4 tăng 20%", "zone": "zone_1"},
                {"evidence": "Hợp đồng 100 triệu USD với đối tác Mỹ", "zone": "zone_2"}
            ],
            "bearish": []
        },
        "main_risks": [
            {"risk_type": "macro", "description": "Biến động tỷ giá Yên Nhật có thể ảnh hưởng mảng xuất khẩu phần mềm", "zone": "zone_1"}
        ],
        "anti_hallucination_check_passed": True
    }
    
    return {
        "instruction": instruction,
        "input": input_data,
        "output": json.dumps(output, ensure_ascii=False)
    }

def generate_sample_insufficient():
    """Generates a 'kill switch' case."""
    instruction = "Bạn là Một HỆ THỐNG PHÂN TÍCH ĐỊNH TÍNH (Qualitative Risk Analyst Engine). CHỈ sử dụng thông tin trong <ZONE_DATA>."
    
    input_data = """<ZONE_DATA>
[Zone 1: Fundamental/RAG]
Dữ liệu trống.
[Zone 2: Latest News]
Tin tức: Giá xăng tăng nhẹ.
[Zone 3: Quantitative/Macro Background]
Xác suất ML hiện tại: {'up': 0.5, 'sideways': 0.3, 'down': 0.2}
</ZONE_DATA>"""

    output = {
        "analysis_status": "insufficient_data",
        "confidence_score": 0.3,
        "veto_flag": False,
        "overall_outlook": None,
        "reasoning": "Thiếu dữ liệu doanh nghiệp (Zone 1) và chỉ có 1 evidence vĩ mô không liên quan trực tiếp mã cổ phiếu.",
        "signals": None,
        "main_risks": None
    }
    
    return {
        "instruction": instruction,
        "input": input_data,
        "output": json.dumps(output, ensure_ascii=False)
    }

def main():
    output_file = "data/training/risk_analyst_instructions.jsonl"
    os.makedirs("data/training", exist_ok=True)
    
    dataset = []
    # Generate 10 sample pairs (mixed)
    for _ in range(5):
        dataset.append(generate_sample_success())
        dataset.append(generate_sample_insufficient())
        
    random.shuffle(dataset)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"✅ Generated {len(dataset)} instruction pairs in {output_file}")
    print("Format optimized for Llama-3/Qwen fine-tuning (instruction/input/output).")

if __name__ == "__main__":
    main()
