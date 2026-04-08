# Structure Final - Low-Resource Deployment Branch

## File Structure Overview

```text
h:\AI-ML-LLM in Stock_march26_PTIT_NEU/
├── config/
│   ├── settings.py           <-- [MODIFIED] Local-Only Defaults
├── src/
│   ├── agents/
│   │   ├── orchestrator.py   <-- [UNCHANGED] ML -> Risk -> Portfolio -> Explain Flow
│   │   ├── explainer.py      <-- [MODIFIED] Local LLM Optimization
│   ├── ml/
│   │   ├── llm/
│   │   │   ├── client.py     <-- [UNCHANGED] OpenAI-compatible Ollama provider
├── AUDIT_REPORT.md           <-- [NEW] This Audit
├── IMPROVEMENT_ROADMAP.md    <-- [NEW] Planned Upgrades
├── REFACTOR_LOG.md           <-- [NEW] History of modifications
└── STRUCTURE_FINAL.md        <-- [NEW] This document
```

## Folder Responsibilities

- **`config/`**: Centralized configuration via Pydantic Settings. For this branch, it defaults to Ollama and local Qwen models.
- **`src/agents/`**: Contains the multi-agent logic. The `AgentOrchestrator` is the deterministic core.
- **`src/ml/llm/`**: Houses LLM clients and prompts. The `client.py` is generic and supports local Ollama endpoints.

## Future Development Guidelines

- **New Agents**: Any new "active" agents (controlling decisions) must NOT be added to this branch's orchestrator to maintain its "Explain-Only" nature.
- **Dependency Management**: Avoid adding heavy Python dependencies (e.g., LangChain) to keep the resource footprint minimal.

---
*Signed: Antigravity - Senior Software Architect*
