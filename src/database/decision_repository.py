import json
import os
from datetime import datetime
from .decision_card_schema import DecisionCard

class DecisionRepository:
    """
    Theo dõi và thu thập Audit Trails (Decision Cards).
    Ở phiên bản mới, lưu Artifact JSON riêng rẽ vào reports/decision_cards/.
    Bản Production sẽ viết vào TimescaleDB.
    """
    def __init__(self, output_dir: str = "reports/decision_cards"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file = os.path.join(self.output_dir, "decisions_log.jsonl")

    def save_decision(self, decision: DecisionCard):
        """
        Lưu đối tượng DecisionCard vào JSON lines file và JSON artifact rời.
        """
        try:
            # Pydantic dump model
            data_dict = decision.model_dump()
            # Convert datetime format explicitly safely
            date_field = data_dict["meta"]["timestamp"]
            if isinstance(date_field, str):
                pass
            elif hasattr(date_field, "isoformat"):
                data_dict["meta"]["timestamp"] = date_field.isoformat()
            
            # Write to JSONL
            with open(self.log_file, "a", encoding="utf-8") as file:
                file.write(json.dumps(data_dict, ensure_ascii=False) + "\n")
                
            # Write discrete JSON Artifact
            ticker = data_dict["meta"]["ticker"]
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            artifact_file = os.path.join(self.output_dir, f"{ticker}_{timestamp_str}_{data_dict['meta']['decision_id']}.json")
            with open(artifact_file, "w", encoding="utf-8") as file:
                json.dump(data_dict, file, ensure_ascii=False, indent=2)
                
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

