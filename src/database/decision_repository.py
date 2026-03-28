import json
import os
from .decision_card_schema import DecisionCard

class DecisionRepository:
    """
    Theo dõi và thu thập Audit Trails (Decision Cards).
    Ở phiên bản 1, vì lý do nhanh chóng, lưu dạng JSONL vào tmp/reports/.
    Bản Production sẽ viết vào TimescaleDB (JSONB type column).
    """
    def __init__(self, output_dir: str = "tmp/reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file = os.path.join(self.output_dir, "decisions_log.jsonl")

    def save_decision(self, decision: DecisionCard):
        """
        Lưu đối tượng DecisionCard vào JSON lines file.
        """
        try:
            # Pydantic dump model
            data_dict = decision.model_dump()
            # Convert datetime to ISO string
            data_dict["meta"]["timestamp"] = data_dict["meta"]["timestamp"].isoformat()
            
            with open(self.log_file, "a", encoding="utf-8") as file:
                file.write(json.dumps(data_dict, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Lỗi khi save DecisionCard: {e}")

    def get_decisions_by_ticker(self, ticker: str):
        """Đọc ngược log file (Hoặc Query lại từ TimescaleDB) để replay."""
        results = []
        if not os.path.exists(self.log_file):
            return results
        with open(self.log_file, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                if data.get("meta", {}).get("ticker") == ticker:
                    results.append(data)
        return results
