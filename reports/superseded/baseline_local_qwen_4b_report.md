# Baseline Benchmark Report: local-qwen

**Date**: 2026-04-03 22:21:10
**Branch**: low-resource-local-qwen

## 1. Runtime Performance
- **Explainer Disabled**: 0.12 ms
- **Explainer Enabled (Healthy)**: 443.25 ms
- **Explainer Forced Failure**: 0.20 ms

## 2. Deterministic Portfolio Parity
| Mode Comparison | Parity Result | Deterministic Fields Match |
| :--- | :--- | :--- |
| Disabled vs Enabled | PASSED | ticker, action, weight, exposure, buffer |
| Disabled vs Forced Failure | PASSED | ticker, action, weight, exposure, buffer |

## 3. Resilience & Fallback
- **Fallback String Caught**: YES
- **Failure Sample**: `Explain-Only Fallback: Trading system decision made, but explanation generation failed. (Benchmark F...`

## 4. Operational Observations
- **Core Integrity**: Quantitative trading path remains logic-neutral.

---
Deterministic Core Parity: PASS
Explainer Fallback: PASS
Baseline Status: VERIFIED
