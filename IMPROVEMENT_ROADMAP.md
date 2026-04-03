# Improvement Roadmap - Low-Resource Deployment Branch

## P0: Core Functionality (Immediate)
- **Local Model Configuration**: Set up `settings.py` to use Ollama with `qwen3:8b` as the default.
- **Explain-Only Mode**: Ensure `enable_llm_explainer` is `True` and its output is correctly captured in reports.
- **Asynchronous Execution**: Verify that the orchestrator's `await` calls prevent UI blocking during inference.

## P1: Performance & UX (Short-Term)
- **Prompt Optimization**: Streamline the quantitative context passed to the 8B model to reduce token count and latency.
- **Vietnamese Language Support**: Add explicit system instructions for consistent, professional Vietnamese financial analysis.
- **Memory Management**: Implement conditional LLM loading (if possible via Ollama's `keep_alive` or similar) to save resources for the ML pipeline.

## P2: Long-Term Enhancements
- **Multi-Ticker Batching**: Optimize `explain_batch` to handle multiple tickers efficiently on low-resource hardware.
- **Quantization Support**: Add documentation for using GGUF-quantized models (e.g., Q4_K_M) for even lower resource usage.
- **Model Switching Logic**: Create a fallback mechanism if the local LLM is down (e.g., use a simple rule-based summary).

---
*Signed: Antigravity - Senior Software Architect*
