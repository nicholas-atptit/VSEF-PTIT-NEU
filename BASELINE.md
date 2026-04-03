# Local Baseline: low-resource-local-qwen

## Purpose
This branch provides a low-resource, single-local-model baseline for the trading system.

## Target Use Case
- Local development
- Low-cost deployment
- Deterministic portfolio decisions
- Optional natural-language explanation using a local LLM

## Final Architecture
ML / Signal -> Analyst (if enabled in this branch) -> Risk -> Portfolio -> Local Explainer

## Deterministic Core
The following components remain deterministic and authoritative:
- Signal generation
- Risk review
- Portfolio allocation

## Optional Layer
The local LLM explanation layer is optional and runs only after portfolio decisions are finalized.

## Failure Behavior
If Ollama or the local LLM is unavailable:
- portfolio output must still be returned
- trading decisions remain unchanged
- explanation falls back gracefully

## Why This Branch Exists
This branch is the stable comparison baseline before scaling back up to more complex multi-agent or multi-model systems.

## Comparison Role for Future Branches
Future branches should be compared against this one on:
- latency
- stability
- backtest parity
- failure tolerance
- operational complexity
