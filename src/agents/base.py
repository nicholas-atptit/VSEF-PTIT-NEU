from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os

class BaseAgent(ABC):
    """
    Abstract base class cho tất cả các Agents trong hệ thống.
    Mỗi Agent là một nút (node) trong Orchestrator Graph.
    """
    def __init__(self, name: str, llm_provider: Optional[str] = None):
        self.name = name
        # Ưu tiên provider truyền vào, nếu không dùng LLM_PROVIDER từ env
        self.llm_provider = llm_provider or os.getenv('LLM_PROVIDER', 'ollama')

    @abstractmethod
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input: Dữ liệu (State, Market Context, Models).
        Output: Phân tích hoặc quyết định của Agent dưới dạng dict.
        """
        pass
