# 1. Executive Summary

This repository is **not** best understood as a generic “AI stock prediction” project. It is more accurately a **Vietnamese equity quantitative research platform** that combines OHLCV-based forecasting, walk-forward evaluation, statistical comparison utilities, regime/risk overlays, and deterministic allocation/routing layers.

The strongest verified academic identity is **walk-forward evaluation of regime-aware multi-model forecasting for Vietnamese equities**. That identity is supported by real code in the feature, trainer, walk-forward, benchmark, stress-test, statistical, allocation, and router modules. It is **not** strongest as a production trading system, and it is **not** strongest as a pure explainable-AI project.

The repository contains meaningful technical depth, but it also contains major caveats:
- active governance docs say the system is diagnostic-only,
- active API/service code still exposes BUY/SELL/execution semantics,
- several runtime paths are mock/demo-heavy,
- reproducibility is weakened by local environment assumptions and committed artifacts/caches,
- some research claims can be supported only as **engineering scaffolding** rather than as **finished academic contribution**.

My final judgment is:
- **Good candidate for an undergraduate thesis:** yes, with scope control.
- **Possible candidate for a master’s thesis:** yes, if narrowed and empirically hardened.
- **Ready conference paper:** not yet.
- **Ready journal paper:** no.

**Final recommended title:**

**English:** *Walk-Forward Evaluation of Regime-Aware Multi-Model Forecasting for Vietnamese Equities*

**Vietnamese:** *Đánh giá Walk-Forward cho Hệ Dự báo Đa Mô Hình Có Nhận diện Chế độ Thị trường trên Cổ phiếu Việt Nam*

That title is the most defensible because it matches what the repository actually proves in code: multi-model forecasting, time-series-aware evaluation, regime-aware analysis, and a Vietnam-specific market setting. It avoids inflated claims about execution, alpha, portfolio optimality, or novel AI.

---

# 2. Codebase Audit

## 2.1 Repository architecture

### What currently exists
- Main runtime and governance surfaces are visible in:
  - [README.md](README.md)
  - [src/api/main.py](src/api/main.py)
  - [src/api/routes.py](src/api/routes.py)
  - [src/api/routes_v2.py](src/api/routes_v2.py)
  - [src/ml/trainer.py](src/ml/trainer.py)
  - [src/evaluation/walkforward.py](src/evaluation/walkforward.py)
  - [src/evaluation/backtest.py](src/evaluation/backtest.py)
  - [src/portfolio_allocator/gating.py](src/portfolio_allocator/gating.py)
  - [src/phase3_router/routing.py](src/phase3_router/routing.py)
- The documented diagnostic chain is present in [README.md](README.md), [docs/AUTHORITY_BOUNDARY.md](docs/AUTHORITY_BOUNDARY.md), and [docs/governance/PIPELINE_CONTRACTS.md](docs/governance/PIPELINE_CONTRACTS.md).
- Experiment and artifact-oriented flows are visible in committed reports, benchmark modules, and scripts.

### Main folders and roles
- `src/data/`, `src/ml/`, `src/evaluation/`, `src/risk_governance/`, `src/portfolio_allocator/`, `src/phase3_router/`, `src/api/`, `scripts/`, `tests/`, `docs/`, `reports/`.

### Main scripts / entry points
- `scripts/run_quant_core.py` — documented quant-core chain entry from [README.md](README.md).
- `scripts/run_experiment.py` — config-driven experiment runner with local venv example already committed in code.
- `uvicorn src.api.main:app` style service path implied by [src/api/main.py](src/api/main.py).

### Data flow
- Market/provider data -> canonical adapter -> data loader/context joins -> feature engineering -> trainer/evaluator -> backtest/benchmark -> risk/allocation/router -> API/report artifacts.

### Experiment flow
- Training and comparison are organized around the trainer, benchmark runners, stress tests, risk tuning, and walk-forward robustness modules.

### Strengths
- The repository is not a toy. There is real modular structure for forecasting, evaluation, and downstream decision diagnostics.
- Governance documents are explicit and detailed.

### Weaknesses
- Runtime structure is mixed with legacy/demo/service surfaces.
- Repository root and output organization are noisy.

### Risks
- A reviewer may conclude the project is overengineered relative to its strongest research contribution.

### Must be fixed
- Separate canonical research path from legacy/demo/API surfaces in the thesis narrative.

### Research-grade?
- **Partially yes.** Architecture supports research.

### Production-grade?
- **No.** API/runtime contradictions and hygiene issues are too significant.

---

## 2.2 Evidence table

| System capability | Evidence file(s) | Verification status | Research relevance | Production relevance | Risk / caveat |
| --- | --- | --- | --- | --- | --- |
| Diagnostic-only governance chain | [README.md](README.md), [docs/AUTHORITY_BOUNDARY.md](docs/AUTHORITY_BOUNDARY.md), [docs/governance/PIPELINE_CONTRACTS.md](docs/governance/PIPELINE_CONTRACTS.md) | Documented but not fully verified | High | High | Contradicted by active API/service code using BUY/SELL/execution semantics. |
| Canonical VN data adapter with schema normalization | [src/data/adapters/vnstock_adapter.py](src/data/adapters/vnstock_adapter.py) | Verified by source code | High | High | Mock/stub fallback paths reduce provenance certainty in some flows. |
| OHLCV loading and context joins | [src/ml/data_loader.py](src/ml/data_loader.py) | Verified by source code | High | Medium | Loader contains multiple fallback modes including synthetic/mock behavior. |
| Time-series-safe feature engineering claims | [src/ml/feature_engineering.py](src/ml/feature_engineering.py) | Verified by source code | High | Medium | Main path looks careful; thesis should not overclaim every peripheral feature branch is equally mature. |
| Multi-horizon trainer and artifact-manifest flow | [src/ml/trainer.py](src/ml/trainer.py) | Verified by source code | High | Medium | Strong engineering scaffold, but not enough by itself to claim novel methodology. |
| Walk-forward evaluation with gap-aware windows | [src/evaluation/walkforward.py](src/evaluation/walkforward.py) | Verified by source code | Very high | Medium | Strongest evaluation contribution; should be central in thesis framing. |
| Cost-aware backtest with fee/slippage assumptions | [src/evaluation/backtest.py](src/evaluation/backtest.py) | Verified by source code | High | Medium | Realistic assumptions exist, but end-to-end runtime still has mock-heavy branches elsewhere. |
| Bootstrap confidence intervals | [src/ml/statistics/bootstrap_eval.py](src/ml/statistics/bootstrap_eval.py), [tests/ml/statistics/test_bootstrap_eval.py](tests/ml/statistics/test_bootstrap_eval.py) | Verified by tests | High | Low | Utility exists; not proven to govern all report claims repository-wide. |
| Diebold-Mariano forecast comparison | [src/ml/statistics/dm_test.py](src/ml/statistics/dm_test.py), [tests/ml/statistics/test_dm_test.py](tests/ml/statistics/test_dm_test.py) | Verified by tests | High | Low | Good research utility; again not proven to be systematically enforced in all benchmark narratives. |
| Benchmark runner across modes | [src/ml/benchmark/system_benchmark.py](src/ml/benchmark/system_benchmark.py), [tests/ml/test_system_benchmark.py](tests/ml/test_system_benchmark.py) | Verified by tests | Very high | Low | Strong research scaffold, but still mainly point-estimate / leaderboard style. |
| Deterministic stress testing | [src/ml/benchmark/stress_test.py](src/ml/benchmark/stress_test.py), [tests/ml/test_stress_test.py](tests/ml/test_stress_test.py) | Verified by tests | High | Low | Good research support, not itself a strong thesis title. |
| Risk-aware tuning | [src/ml/benchmark/risk_tuning.py](src/ml/benchmark/risk_tuning.py) | Verified by source code | Medium | Low | Validation-only tuning is better than naive tuning, but selection bias still matters. |
| Portfolio allocation gating | [src/portfolio_allocator/gating.py](src/portfolio_allocator/gating.py), [tests/portfolio_allocator/test_allocator_gating.py](tests/portfolio_allocator/test_allocator_gating.py) | Verified by tests | Medium | Medium | Useful deterministic layer; more engineering than central research novelty. |
| Phase 3 routing diagnostics | [src/phase3_router/routing.py](src/phase3_router/routing.py), [tests/phase3_router/test_router_decisions.py](tests/phase3_router/test_router_decisions.py) | Verified by tests | Medium | Medium | Good governed diagnostic layer, but not best primary academic identity. |
| API v1 execution-style outputs | [src/api/routes.py](src/api/routes.py), [src/api/schemas.py](src/api/schemas.py) | Contradicted by code | Low | High | Directly conflicts with diagnostic-only governance story. |
| API v2 fused BUY output | [src/api/routes_v2.py](src/api/routes_v2.py), [src/api/schemas_v2.py](src/api/schemas_v2.py) | Placeholder / mock / demo-only | Low | Low | Hardcoded/mocked semantics make this unsuitable as thesis evidence. |
| Repository hygiene and reproducibility baseline | [reports/cleanup/CODE_AUDIT_REPORT.md](reports/cleanup/CODE_AUDIT_REPORT.md), committed `__pycache__`, `outputs/`, `artifacts/`, `models/` | Verified by committed report/artifact | High | High | Must be described honestly as a weakness. |

---

## 2.3 Data pipeline

### What currently exists
- Canonical VN data integration is represented by [src/data/adapters/vnstock_adapter.py](src/data/adapters/vnstock_adapter.py).
- Loading, validation, market/sector/macro/breadth/foreign-flow joins are represented by [src/ml/data_loader.py](src/ml/data_loader.py).

### Data source
- `vnstock_data` is declared as canonical in dependencies and adapter design.
- **Verification status:** Verified by source code.

### Data cleaning / schema
- Earlier source inspection showed standardization toward `date`, `ticker`, `open`, `high`, `low`, `close`, `volume` and explicit sorting/deduplication in the adapter.
- [src/ml/trainer.py:194](src/ml/trainer.py:194)–[224](src/ml/trainer.py:224) confirms normalization, date parsing, uppercase ticker handling, numeric coercion, sort, and dedupe in the trainer path.
- **Verification status:** Verified by source code.

### Missing value handling
- The loader and feature pipeline use coercion and fill/forward-fill patterns.
- This is practical engineering, but every fill rule is not automatically academically neutral.
- **Verification status:** Verified by source code.

### Chronological sorting
- Trainer normalization explicitly sorts and deduplicates by date: [src/ml/trainer.py:224](src/ml/trainer.py:224).
- Walk-forward logic sorts timestamps before splitting: [src/evaluation/walkforward.py:65](src/evaluation/walkforward.py:65), [159](src/evaluation/walkforward.py:159)–[166](src/evaluation/walkforward.py:166).
- **Verification status:** Verified by source code.

### Ticker-level consistency
- The trainer inserts or normalizes ticker labels: [src/ml/trainer.py:219](src/ml/trainer.py:219)–[223](src/ml/trainer.py:223).
- **Verification status:** Verified by source code.

### Leakage risk
- No obvious future-join pattern was found in the core loader/trainer/walk-forward path that would automatically invalidate the whole repo.
- The bigger risk is not blatant leakage in the core path; it is **scope creep across multiple research and runtime branches**.
- **Verification status:** Inferred from architecture plus verified code inspection.

### Strong
- Time ordering and normalization are taken seriously in the core path.

### Weak
- Mock/fallback behavior weakens provenance in some workflows.

### Risky
- A thesis that claims “clean market data pipeline” without qualifying fallback behavior would overstate maturity.

### Must be fixed
- For academic use, all experiments must explicitly state whether any mock or fallback path was possible.

### Research-grade?
- **Yes, conditionally.**

### Production-grade?
- **No.**

---

## 2.4 Feature engineering

### What currently exists
- Central feature logic is in [src/ml/feature_engineering.py](src/ml/feature_engineering.py).
- The module explicitly states that deterministic features use only information available on or before time `t`: [src/ml/feature_engineering.py:1](src/ml/feature_engineering.py:1)–[6](src/ml/feature_engineering.py:6).
- Feature sets include returns, volatility proxies, ATR-style features, RSI, MACD, volume shock, and research/compatibility modes: [src/ml/feature_engineering.py:40](src/ml/feature_engineering.py:40)–[71](src/ml/feature_engineering.py:71), [127](src/ml/feature_engineering.py:127)–[169](src/ml/feature_engineering.py:169).

### Technical indicators
- RSI and Bollinger-style helpers exist explicitly: [src/ml/feature_engineering.py:199](src/ml/feature_engineering.py:199) onward.
- MACD and RSI fields appear in the VN100 feature inventory.
- **Verification status:** Verified by source code.

### Lagged / rolling / volatility / volume features
- Feature inventory includes lagged returns, rolling volatility, moving averages, ATR proxies, and volume shocks: [src/ml/feature_engineering.py:127](src/ml/feature_engineering.py:127)–[169](src/ml/feature_engineering.py:169).
- **Verification status:** Verified by source code.

### Safe for time-series forecasting?
- The core claim is plausible and supported by the module contract plus the walk-forward/trainer ecosystem.
- However, this should be phrased as “main canonical path appears time-series aware,” not “all repository features are fully leakage-proof.”
- **Verification status:** Verified by source code, with caution.

### Potential future leakage?
- No immediate smoking gun in the main feature inventory.
- The risk is broader: multiple feature modes and legacy compatibility branches can make thesis scope ambiguous.
- **Verification status:** Inferred from architecture.

### Strong
- Rich feature surface and explicit time-series framing.

### Weak
- Some branches are backward-compatibility/research-expansion oriented rather than thesis-clean.

### Risky
- Overclaiming novelty. Technical indicators + gradient boosting + recurrent models is not novel by itself.

### Must be fixed
- Thesis scope should freeze a small canonical feature set and justify it.

### Research-grade?
- **Yes, as supporting methodology.**

### Production-grade?
- **Partially.**

---

## 2.5 Model training pipeline

### What currently exists
- Unified trainer facade: [src/ml/trainer.py](src/ml/trainer.py).
- Multi-horizon support: [src/ml/trainer.py:77](src/ml/trainer.py:77)–[81](src/ml/trainer.py:81).
- Sequence models and booster families explicitly named: [src/ml/trainer.py:82](src/ml/trainer.py:82)–[86](src/ml/trainer.py:86).
- Artifact/manifest dependencies are imported at the top of the trainer: [src/ml/trainer.py:13](src/ml/trainer.py:13)–[24](src/ml/trainer.py:24).

### Models implemented
- Earlier repository exploration found CART, XGBoost, LightGBM, SARIMAX, ETS, LSTM, and BiLSTM paths around the trainer/benchmark stack.
- **Verification status:** Verified by source code.

### Training flow
- Data normalization -> context loading -> feature preparation -> task/horizon model training -> manifest writing/loading.
- **Verification status:** Verified by source code.

### Hyperparameter tuning
- Risk-side tuning exists in [src/ml/benchmark/risk_tuning.py](src/ml/benchmark/risk_tuning.py).
- This is useful, but tuning alone is not a thesis contribution.
- **Verification status:** Verified by source code.

### Model persistence
- Manifest and artifact helpers are directly imported and used by the trainer.
- **Verification status:** Verified by source code.

### Random seed handling
- Benchmark/statistics/tuning modules use explicit seeds.
- The repo takes determinism more seriously than a casual prototype.
- **Verification status:** Verified by source code and tests.

### Fairness of model comparison
- [tests/ml/test_system_benchmark.py](tests/ml/test_system_benchmark.py) explicitly checks identical train/val/test row counts across benchmark modes.
- **Verification status:** Verified by tests.

### Strong
- This is one of the strongest parts of the repo.

### Weak
- The broader surface is larger than what a single thesis can justify.

### Risky
- Reviewer criticism: “You implemented many models, but what is the actual research question?”

### Must be fixed
- The thesis must compare a controlled subset of models with a narrow target and evaluation protocol.

### Research-grade?
- **Yes, conditionally.**

### Production-grade?
- **No.**

---

## 2.6 Validation and evaluation

### What currently exists
- Walk-forward evaluator and splitter in [src/evaluation/walkforward.py](src/evaluation/walkforward.py).
- Forecast metrics including MAE, RMSE, MAPE, sMAPE, directional accuracy, hit rate in [src/evaluation/walkforward.py:111](src/evaluation/walkforward.py:111)–[125](src/evaluation/walkforward.py:125).
- Statistical helpers in [src/ml/statistics/bootstrap_eval.py](src/ml/statistics/bootstrap_eval.py) and [src/ml/statistics/dm_test.py](src/ml/statistics/dm_test.py).

### Train/test split
- Walk-forward windows are explicit and gap-aware: [src/evaluation/walkforward.py:149](src/evaluation/walkforward.py:149)–[194](src/evaluation/walkforward.py:194).
- **Verification status:** Verified by source code.

### Time-series split
- Explicitly time-based, not random.
- **Verification status:** Verified by source code.

### Walk-forward validation
- This is a central and defensible strength.
- **Verification status:** Verified by source code.

### Metrics
- Forecast metrics are implemented directly.
- Backtest metrics are separately implemented in [src/evaluation/backtest.py](src/evaluation/backtest.py).
- **Verification status:** Verified by source code.

### Bootstrap CI
- Available and directly tested.
- **Verification status:** Verified by tests.

### Diebold-Mariano test
- Available and directly tested.
- **Verification status:** Verified by tests.

### Research-grade?
- **Yes, partially.** The infrastructure is research-capable.

### Weakness
- The presence of DM/bootstrap utilities does **not** prove that all performance claims in the repo are already statistically disciplined.

### Must be fixed
- Thesis experiments must explicitly use DM/bootstrap in final result tables, not just leave them as optional utilities.

### Production-grade?
- **Not the relevant standard here; still no.**

---

## 2.7 Backtesting

### What currently exists
- Cost-aware helper in [src/evaluation/backtest.py](src/evaluation/backtest.py).
- Transaction fee and slippage assumptions in [src/evaluation/backtest.py:24](src/evaluation/backtest.py:24)–[33](src/evaluation/backtest.py:33).
- Non-overlapping control in the backtester config and run loop: [src/evaluation/backtest.py:32](src/evaluation/backtest.py:32), [187](src/evaluation/backtest.py:187)–[188](src/evaluation/backtest.py:188).

### Signal generation / return calculation
- Net trade return calculation is explicit: [src/evaluation/backtest.py:43](src/evaluation/backtest.py:43)–[68](src/evaluation/backtest.py:68).
- Daily marked-to-market trade return path is explicit: [src/evaluation/backtest.py:71](src/evaluation/backtest.py:71)–[99](src/evaluation/backtest.py:99).
- **Verification status:** Verified by source code.

### Transaction costs / slippage
- Real numeric assumptions are encoded, not merely documented.
- **Verification status:** Verified by source code.

### Position sizing / overlap
- Position size is explicit and overlap can be disabled.
- **Verification status:** Verified by source code.

### Benchmark comparison
- Benchmark runners compare system modes, not just single-model raw metrics.
- **Verification status:** Verified by source code and tests.

### Look-ahead bias risk
- Core walk-forward + horizon-aware structure reduces obvious look-ahead risk.
- **Verification status:** Inferred from architecture plus verified code.

### Financial realism
- Better than many student repositories because costs and overlap controls exist.
- Still not enough to claim a production-trading-ready backtest stack.

### Strong
- This is a meaningful research asset.

### Weak
- End-to-end service/runtime still contains mock branches.

### Must be fixed
- Thesis should clearly separate research backtest engine from API/demo behavior.

### Research-grade?
- **Yes, conditionally.**

### Production-grade?
- **No.**

---

## 2.8 Risk and portfolio layer

### What currently exists
- Deterministic risk scoring in [src/risk_governance/scoring.py](src/risk_governance/scoring.py).
- Deterministic allocation gating in [src/portfolio_allocator/gating.py](src/portfolio_allocator/gating.py).
- Deterministic diagnostic routing in [src/phase3_router/routing.py](src/phase3_router/routing.py).

### VaR / CVaR / downside risk
- Downside-risk normalization is explicit in [src/risk_governance/scoring.py:82](src/risk_governance/scoring.py:82)–[100](src/risk_governance/scoring.py:100).
- **Verification status:** Verified by source code.

### Drawdown / volatility / disagreement / calibration
- Multiple normalized components are encoded: drawdown, volatility, downside risk, model health, disagreement, scenario dispersion, calibration.
- **Verification status:** Verified by source code.

### Risk-aware allocation
- Allocation gates cover risk level, confidence, disagreement, dominance, and scenario alignment in [src/portfolio_allocator/gating.py](src/portfolio_allocator/gating.py).
- **Verification status:** Verified by tests.

### Meaningful research or only engineering?
- Mostly **engineering plus empirical decision design**.
- Useful in a thesis as a downstream applied layer.
- Weak as the sole research contribution unless the thesis specifically studies risk-aware decision rules.

### Strong
- The governance/routing discipline is better than average student code.

### Weak
- It is hard to defend this layer alone as a novel academic contribution.

### Research-grade?
- **Medium, as a secondary contribution.**

### Production-grade?
- **No.**

---

## 2.9 Reproducibility

### What currently exists
- Manifest/artifact-oriented trainer design and numerous committed reports/artifacts.
- Deterministic seeds in benchmark/statistics/tuning modules.
- Existing baseline report at [reports/cleanup/CODE_AUDIT_REPORT.md](reports/cleanup/CODE_AUDIT_REPORT.md).

### What is strong
- The repo is at least aware of manifests, reports, and benchmark outputs.
- Determinism is not ignored.

### What is weak
- Earlier verified evidence showed hardcoded local environment assumptions, machine-specific script guidance, committed caches/artifacts, and repo-noise.

### What is risky
- Another researcher may not be able to reproduce the same outputs from a clean machine without undocumented local setup.

### Must be fixed
- A thesis must declare exact data source, time span, environment, seeds, and output lineage.

### Research-grade?
- **Partially.**

### Production-grade?
- **No.**

---

## 2.10 Test coverage

### What currently exists
- Statistical helper tests: [tests/ml/statistics/test_bootstrap_eval.py](tests/ml/statistics/test_bootstrap_eval.py), [tests/ml/statistics/test_dm_test.py](tests/ml/statistics/test_dm_test.py)
- Benchmark tests: [tests/ml/test_system_benchmark.py](tests/ml/test_system_benchmark.py), [tests/ml/test_stress_test.py](tests/ml/test_stress_test.py)
- Allocator/router tests: [tests/portfolio_allocator/test_allocator_gating.py](tests/portfolio_allocator/test_allocator_gating.py), [tests/phase3_router/test_router_decisions.py](tests/phase3_router/test_router_decisions.py), [tests/phase3_router/test_router_outputs.py](tests/phase3_router/test_router_outputs.py)
- API tests: [tests/test_api.py](tests/test_api.py)

### What is strong
- Direct tests exist for several important research modules.

### What is weak
- API test acceptance does not resolve governance contradictions; it currently tolerates BUY/SELL style payload semantics.
- Mock/demo surfaces remain under-disciplined as academic evidence.

### Missing tests
- The academically important missing piece is not only unit tests; it is **thesis-level protocol enforcement**: strict baselines, statistical comparison, and scope locking.

### Research-grade?
- **Moderately supportive.**

### Production-grade?
- **No.**

---

# 3. Research Identity Diagnosis

This project is **not primarily a generic stock forecasting thesis**.

It is also **not primarily a portfolio optimization thesis**, **not primarily an explainable-AI thesis**, and **not primarily a production trading platform thesis**.

Its strongest academic identity is:

> **A walk-forward, regime-aware, multi-model empirical forecasting study for Vietnamese equities, with supporting backtest and risk diagnostics.**

Why this is the best fit:
1. The repo has real forecasting, feature, and trainer infrastructure.
2. The walk-forward code is explicit and credible.
3. Benchmark and stress-testing modules exist and are tested.
4. Regime-aware analysis is directly represented in committed code and reports.
5. The downstream allocation/router layers are interesting, but they are stronger as **secondary engineering/application layers** than as the primary thesis claim.

What the project is **really about academically**:
- emerging-market forecasting,
- model comparison under realistic temporal splits,
- regime/risk context in predictive systems,
- empirical validation discipline.

What the project is **not yet ready to claim**:
- a production trading system,
- a novel execution engine,
- state-of-the-art alpha generation,
- a publishable journal contribution on methodological novelty alone.

---

# 4. Research Direction Scoring

## 4.1 Direction scores

| Direction | Fit with current repo | Academic novelty | Technical depth | Feasibility | Thesis suitability | Conference suitability | Journal suitability | Industry value | Reviewer criticism risk | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A. Stock return forecasting | 8 | 4 | 7 | 8 | 8 | 5 | 3 | 7 | 6 | Good but generic if framed poorly |
| B. Directional movement prediction | 7 | 3 | 6 | 8 | 7 | 4 | 2 | 7 | 7 | Too common unless tightly scoped |
| C. Robust walk-forward model evaluation | 10 | 7 | 8 | 9 | 9 | 8 | 6 | 7 | 4 | One of the strongest directions |
| D. Regime-aware forecasting | 9 | 7 | 8 | 8 | 9 | 8 | 6 | 8 | 5 | Strong and defensible |
| E. Ensemble learning for emerging markets | 8 | 6 | 8 | 8 | 8 | 7 | 5 | 8 | 5 | Strong if benchmarked carefully |
| F. Risk-aware portfolio construction | 7 | 5 | 7 | 6 | 7 | 5 | 4 | 8 | 6 | Better as secondary layer |
| G. Transaction-cost-aware backtesting | 8 | 5 | 7 | 8 | 8 | 6 | 4 | 8 | 5 | Useful but narrower |
| H. Reproducible financial ML infrastructure | 7 | 6 | 7 | 6 | 7 | 6 | 5 | 7 | 6 | Honest but less impressive |
| I. Quantitative trading system design | 9 | 3 | 9 | 7 | 7 | 4 | 2 | 9 | 8 | Too engineering-heavy |
| J. Explainable AI for investment decision support | 4 | 4 | 5 | 4 | 4 | 3 | 2 | 6 | 8 | Not well supported by verified evidence |
| K. Vietnamese stock market inefficiency | 5 | 6 | 6 | 4 | 5 | 4 | 4 | 7 | 8 | Requires stronger economic identification than repo proves |
| L. Financial engineering for emerging equity markets | 8 | 5 | 8 | 7 | 8 | 6 | 4 | 8 | 6 | Acceptable umbrella, but still broad |

## 4.2 Why these scores are justified

- **Highest fit:** walk-forward evaluation and regime-aware forecasting because these are directly implemented and tested.
- **Lower novelty:** generic directional prediction or generic “AI stock prediction.”
- **High industry value but weak academic defensibility:** full trading-system framing.
- **Weak fit:** explainable AI; the repository has an LLM/explainer surface, but this is not the strongest verified research core.

---

# 5. Research Title Candidates

## 5.1 At least 20 title candidates

| # | English title | Vietnamese title |
| --- | --- | --- |
| 1 | Walk-Forward Evaluation of Regime-Aware Multi-Model Forecasting for Vietnamese Equities | Đánh giá Walk-Forward cho Hệ Dự báo Đa Mô Hình Có Nhận diện Chế độ Thị trường trên Cổ phiếu Việt Nam |
| 2 | Regime-Aware Forecasting and Walk-Forward Validation for Vietnamese Equity Time Series | Dự báo Có Nhận diện Chế độ Thị trường và Kiểm định Walk-Forward cho Chuỗi Thời gian Cổ phiếu Việt Nam |
| 3 | A Walk-Forward Comparison of Machine Learning and Statistical Forecasting Models for Vietnamese Stocks | So sánh Walk-Forward giữa các Mô hình Học máy và Mô hình Thống kê cho Cổ phiếu Việt Nam |
| 4 | Multi-Horizon Forecasting for Vietnamese Equities under Walk-Forward Evaluation | Dự báo Đa Chân trời cho Cổ phiếu Việt Nam dưới Khung Đánh giá Walk-Forward |
| 5 | Regime-Aware Multi-Horizon Forecasting for Vietnamese Equities | Dự báo Đa Chân trời Có Nhận diện Chế độ Thị trường cho Cổ phiếu Việt Nam |
| 6 | Benchmarking Forecasting Models for Vietnamese Equities with Walk-Forward Validation and Statistical Testing | Đối sánh các Mô hình Dự báo cho Cổ phiếu Việt Nam bằng Kiểm định Walk-Forward và Kiểm định Thống kê |
| 7 | Empirical Evaluation of Ensemble Forecasting for Vietnamese Equities under Temporal Validation | Đánh giá Thực nghiệm Dự báo Tổ hợp cho Cổ phiếu Việt Nam dưới Kiểm định Theo Thời gian |
| 8 | Walk-Forward Benchmarking of Ensemble and Baseline Forecasting Models in the Vietnamese Stock Market | Đối sánh Walk-Forward giữa Mô hình Tổ hợp và Mô hình Cơ sở trên Thị trường Chứng khoán Việt Nam |
| 9 | Transaction-Cost-Aware Evaluation of Forecast-Driven Equity Strategies in Vietnam | Đánh giá Có Tính Chi phí Giao dịch cho các Chiến lược Cổ phiếu Dựa trên Dự báo tại Việt Nam |
| 10 | Risk-Aware Diagnostic Allocation from Forecast Signals in Vietnamese Equities | Phân bổ Danh mục Chẩn đoán Có Nhận thức Rủi ro từ Tín hiệu Dự báo trên Cổ phiếu Việt Nam |
| 11 | Statistical Comparison of Forecasting Models for Vietnamese Equities Using Bootstrap and Diebold-Mariano Tests | So sánh Thống kê các Mô hình Dự báo cho Cổ phiếu Việt Nam bằng Bootstrap và Kiểm định Diebold-Mariano |
| 12 | Robust Evaluation of Vietnamese Equity Forecasting Models across Market Regimes | Đánh giá Vững chắc các Mô hình Dự báo Cổ phiếu Việt Nam qua các Chế độ Thị trường |
| 13 | Walk-Forward Robustness Analysis of Forecasting Systems for Vietnamese Equities | Phân tích Độ Vững Walk-Forward của các Hệ Dự báo cho Cổ phiếu Việt Nam |
| 14 | Emerging-Market Equity Forecasting with Regime-Aware Validation: Evidence from Vietnam | Dự báo Cổ phiếu ở Thị trường Mới nổi với Kiểm định Có Nhận diện Chế độ: Bằng chứng từ Việt Nam |
| 15 | Empirical Study of Multi-Model Forecasting under Regime and Risk Diagnostics for Vietnamese Stocks | Nghiên cứu Thực nghiệm về Dự báo Đa Mô Hình dưới Chẩn đoán Chế độ và Rủi ro cho Cổ phiếu Việt Nam |
| 16 | Reproducible Walk-Forward Forecasting Experiments for Vietnamese Equities | Thực nghiệm Dự báo Walk-Forward Có Thể Tái lập cho Cổ phiếu Việt Nam |
| 17 | Temporal Validation and Stress Testing of Forecasting Models in the Vietnamese Stock Market | Kiểm định Theo Thời gian và Kiểm thử Căng thẳng cho các Mô hình Dự báo trên Thị trường Chứng khoán Việt Nam |
| 18 | Regime-Aware Benchmarking of Forecasting Models for Vietnamese Equities | Đối sánh Có Nhận diện Chế độ của các Mô hình Dự báo cho Cổ phiếu Việt Nam |
| 19 | A Diagnostic Forecasting Framework for Vietnamese Equities with Walk-Forward and Risk Evaluation | Khung Dự báo Chẩn đoán cho Cổ phiếu Việt Nam với Đánh giá Walk-Forward và Rủi ro |
| 20 | Multi-Model Forecast Evaluation for Vietnamese Equities with Walk-Forward Validation | Đánh giá Dự báo Đa Mô Hình cho Cổ phiếu Việt Nam bằng Kiểm định Walk-Forward |
| 21 | Regime-Aware Evaluation of Forecast-Driven Equity Strategies in Vietnam | Đánh giá Có Nhận diện Chế độ đối với các Chiến lược Cổ phiếu Dẫn dắt bởi Dự báo tại Việt Nam |
| 22 | Forecasting Vietnamese Equities under Temporal and Regime Constraints | Dự báo Cổ phiếu Việt Nam dưới các Ràng buộc về Thời gian và Chế độ Thị trường |

## 5.2 Research title decision matrix

| Title short form | Fit /10 | Academic defensibility /10 | Novelty /10 | Feasibility for thesis /10 | Reviewer criticism risk /10 | Required additional work | Final status |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Walk-Forward Regime-Aware Multi-Model Forecasting | 10 | 9 | 7 | 9 | 4 | Medium | Recommended |
| Regime-Aware Forecasting and Walk-Forward Validation | 9 | 9 | 7 | 9 | 4 | Medium | Recommended |
| Walk-Forward ML vs Statistical Models | 9 | 8 | 6 | 9 | 5 | Medium | Recommended |
| Multi-Horizon Forecasting under Walk-Forward | 8 | 7 | 5 | 8 | 6 | Medium | Conditional |
| Regime-Aware Multi-Horizon Forecasting | 8 | 8 | 6 | 8 | 5 | Medium | Recommended |
| Benchmarking with Statistical Testing | 9 | 9 | 7 | 8 | 4 | Medium | Recommended |
| Ensemble Forecasting for Vietnamese Equities | 8 | 7 | 6 | 8 | 5 | Medium | Conditional |
| Walk-Forward Ensemble Benchmarking | 8 | 8 | 6 | 8 | 5 | Medium | Conditional |
| Transaction-Cost-Aware Strategy Evaluation | 8 | 7 | 5 | 8 | 5 | Medium | Conditional |
| Risk-Aware Diagnostic Allocation | 7 | 6 | 5 | 6 | 6 | High | Conditional |
| Bootstrap and DM Comparison of Forecast Models | 8 | 8 | 6 | 7 | 5 | Medium | Conditional |
| Robust Evaluation across Market Regimes | 9 | 8 | 6 | 8 | 5 | Medium | Recommended |
| Walk-Forward Robustness Analysis | 9 | 8 | 6 | 8 | 5 | Medium | Recommended |
| Emerging-Market Forecasting Evidence from Vietnam | 8 | 7 | 6 | 7 | 6 | Medium | Conditional |
| Multi-Model Forecasting under Regime and Risk | 8 | 7 | 6 | 7 | 6 | Medium | Conditional |
| Reproducible Walk-Forward Forecasting Experiments | 7 | 6 | 6 | 6 | 7 | High | Conditional |
| Temporal Validation and Stress Testing | 8 | 7 | 6 | 8 | 5 | Medium | Conditional |
| Regime-Aware Benchmarking | 9 | 8 | 6 | 8 | 5 | Medium | Recommended |
| Diagnostic Forecasting Framework | 7 | 6 | 5 | 7 | 6 | High | Conditional |
| Multi-Model Forecast Evaluation | 8 | 7 | 5 | 8 | 6 | Medium | Conditional |
| Regime-Aware Forecast-Driven Strategies | 7 | 6 | 5 | 7 | 7 | High | Conditional |
| Forecasting under Temporal and Regime Constraints | 8 | 8 | 6 | 8 | 5 | Medium | Recommended |

## 5.3 Rejected title section

### Rejected title 1
**Stock Price Prediction Using Machine Learning**
- **Status:** Reject
- Too generic.
- Says nothing about walk-forward validation, regime context, or Vietnamese market specifics.
- Reviewer criticism would be immediate.

### Rejected title 2
**AI for the Vietnamese Stock Market**
- **Status:** Reject
- Sounds like a slogan, not a thesis title.
- “AI” is vague and hides the actual contribution.

### Rejected title 3
**An Intelligent Trading System for Vietnam**
- **Status:** Reject
- Sounds like a product or startup pitch.
- The repository does not justify “intelligent trading system” as an academically safe primary claim.

### Rejected title 4
**Optimal Portfolio Construction with Deep Learning and Reinforcement Learning**
- **Status:** Reject
- Too ambitious and not supported by the strongest verified repository evidence.
- “Optimal” is indefensible here.

### Rejected title 5
**Explainable AI for Investment Decisions in Emerging Markets**
- **Status:** Reject
- Explainability is not the strongest verified core of the repo.
- LLM/explainer layers exist, but not enough to support this as the main thesis identity.

### Rejected title 6
**A Production-Ready Quantitative Trading Platform for Vietnamese Equities**
- **Status:** Reject
- Contradicted by code and repository hygiene.
- Active API/runtime semantics and reproducibility weaknesses make this unsafe.

---

# 6. Top 5 Recommended Research Topics

## Topic 1
### English title
**Walk-Forward Evaluation of Regime-Aware Multi-Model Forecasting for Vietnamese Equities**

### Vietnamese title
**Đánh giá Walk-Forward cho Hệ Dự báo Đa Mô Hình Có Nhận diện Chế độ Thị trường trên Cổ phiếu Việt Nam**

### Short title
**Walk-Forward Regime-Aware Forecasting**

### Research question
Does a regime-aware, multi-model forecasting framework improve the robustness of Vietnamese equity forecasting under walk-forward evaluation?

### Main contribution
A disciplined empirical comparison of forecasting models under temporal validation and regime-aware analysis, rather than a vague claim of AI-based prediction.

### Why this fits the codebase
Because the repository clearly implements multi-model forecasting, walk-forward evaluation, benchmark runners, stress testing, and regime-aware layers.

### Required experiments
- Fixed ticker universe
- Fixed date range
- Multiple horizons
- Baseline vs regime-aware variants
- Walk-forward windows
- Statistical comparison with DM and bootstrap

### Tables and figures
- Model comparison table by horizon
- Walk-forward split diagram
- Regime-wise performance table
- DM-test significance table
- Bootstrap CI chart
- Equity curve / drawdown plot

### Dependent variable / target
Forward return and/or directional movement over predefined horizons.

### Models to compare
LightGBM, XGBoost, CART/Random Forest, SARIMAX, ETS, optionally LSTM/BiLSTM if data and compute permit.

### Required benchmarks
- naive return baseline
- persistence / sign baseline
- non-regime-aware model variant

### Required statistical tests
Bootstrap CIs, Diebold-Mariano tests, possibly paired tests across windows if justified.

### Limitations
No claim of market inefficiency proof, no claim of trading optimality, no claim of production readiness.

### Reviewer criticism
“Regime awareness may be engineering overlay rather than methodological novelty.”

### Suitability
- Undergraduate thesis: High
- Master’s thesis: High
- Conference paper: Medium
- Journal paper: Low/Medium
- Open-source quant platform: High
- Startup product: Medium

---

## Topic 2
### English title
**Benchmarking Forecasting Models for Vietnamese Equities with Walk-Forward Validation and Statistical Testing**

### Vietnamese title
**Đối sánh các Mô hình Dự báo cho Cổ phiếu Việt Nam bằng Kiểm định Walk-Forward và Kiểm định Thống kê**

### Research question
Which forecasting model families are most robust for Vietnamese equities when compared under leakage-aware walk-forward evaluation and statistical testing?

### Main contribution
A benchmark study with stronger evaluation discipline than typical static train/test comparisons.

### Why this fits the codebase
The repository already contains multi-model infrastructure, benchmark runners, and DM/bootstrap utilities.

### Required experiments
Same split for all models, common features, common target definitions, multiple horizons, statistical ranking.

### Suitability
- Undergraduate thesis: High
- Master’s thesis: High
- Conference paper: Medium
- Journal paper: Low/Medium
- Open-source quant platform: High
- Startup product: Low

---

## Topic 3
### English title
**Robust Evaluation of Vietnamese Equity Forecasting Models across Market Regimes**

### Vietnamese title
**Đánh giá Vững chắc các Mô hình Dự báo Cổ phiếu Việt Nam qua các Chế độ Thị trường**

### Research question
How stable are forecasting model rankings across bull, bear, and sideways regimes in the Vietnamese stock market?

### Main contribution
A regime-conditioned robustness study rather than a one-number leaderboard.

### Why this fits the codebase
Regime-aware benchmarking and robustness files are real and directly relevant.

### Required experiments
Regime labeling, regime-wise metrics, fold-wise stability analysis, stress scenarios.

### Suitability
- Undergraduate thesis: Medium/High
- Master’s thesis: High
- Conference paper: Medium
- Journal paper: Low/Medium
- Open-source quant platform: Medium
- Startup product: Low

---

## Topic 4
### English title
**Transaction-Cost-Aware Evaluation of Forecast-Driven Equity Strategies in Vietnam**

### Vietnamese title
**Đánh giá Có Tính Chi phí Giao dịch cho các Chiến lược Cổ phiếu Dựa trên Dự báo tại Việt Nam**

### Research question
Do forecast-driven strategies for Vietnamese equities remain economically meaningful after transaction costs, slippage, and non-overlapping execution constraints?

### Main contribution
Bridges predictive accuracy and economic usefulness.

### Why this fits the codebase
The backtest layer contains explicit fee/slippage/non-overlap assumptions.

### Suitability
- Undergraduate thesis: Medium
- Master’s thesis: Medium/High
- Conference paper: Medium
- Journal paper: Low
- Open-source quant platform: High
- Startup product: Medium

---

## Topic 5
### English title
**Regime-Aware Benchmarking of Forecasting Models for Vietnamese Equities**

### Vietnamese title
**Đối sánh Có Nhận diện Chế độ của các Mô hình Dự báo cho Cổ phiếu Việt Nam**

### Research question
Does regime-aware benchmarking materially change the relative ranking of forecasting models in the Vietnamese equity market?

### Main contribution
A narrower, cleaner research question than “build the best stock predictor.”

### Why this fits the codebase
It uses the benchmark and regime modules without overclaiming a full trading contribution.

### Suitability
- Undergraduate thesis: High
- Master’s thesis: High
- Conference paper: Medium
- Journal paper: Low/Medium
- Open-source quant platform: Medium
- Startup product: Low

---

# 7. Best Final Title Recommendation

## 7.1 Chosen title

**Final English title:**
**Walk-Forward Evaluation of Regime-Aware Multi-Model Forecasting for Vietnamese Equities**

**Final Vietnamese title:**
**Đánh giá Walk-Forward cho Hệ Dự báo Đa Mô Hình Có Nhận diện Chế độ Thị trường trên Cổ phiếu Việt Nam**

## 7.2 Why this is the most defensible title
- It matches verified code evidence.
- It does not pretend the repository is a production trading platform.
- It does not make unjustified claims about optimal portfolios, explainability, or market inefficiency.
- It directly reflects the strongest real assets: forecasting stack, walk-forward validation, benchmark/stress/statistical utilities, and regime-aware analysis.

## 7.3 Why it is better than generic stock prediction titles
Generic titles hide the actual contribution and invite reviewer criticism for being shallow. This title instead names:
- the evaluation protocol (**walk-forward**),
- the modeling context (**regime-aware**),
- the system structure (**multi-model forecasting**),
- and the domain (**Vietnamese equities**).

## 7.4 What must be completed before using this title in a thesis or paper
- Lock the ticker universe and date range.
- Freeze the final feature set.
- Choose a controlled model subset.
- Produce final walk-forward result tables.
- Use bootstrap CIs and DM tests in the final comparison.
- Report cost-aware backtest results as secondary evidence, not the main claim.
- Explicitly isolate or exclude mock/demo API surfaces from the thesis evidence base.

## 7.5 Final thesis framing

### Research problem
Financial forecasting for Vietnamese equities is difficult because model performance can be unstable across time and market regimes, and naive random or static validation can overstate model quality.

### Research gap
Many student-level financial ML projects compare models using weak evaluation discipline. This repository is stronger because it already contains walk-forward, benchmark, stress-test, and statistical-comparison utilities, but these need to be turned into a focused empirical study.

### Research objective
To evaluate whether regime-aware, multi-model forecasting improves robustness for Vietnamese equity forecasting under a leakage-aware walk-forward framework.

### Research questions
1. Which forecasting models perform most robustly under walk-forward evaluation for Vietnamese equities?
2. Does regime-aware analysis improve robustness or stability of forecasting performance?
3. Are observed model differences statistically meaningful?
4. Do predictive gains translate into economically meaningful strategy performance after basic transaction-cost assumptions?

### Main contribution
A disciplined empirical evaluation framework for Vietnamese equity forecasting that combines walk-forward validation, regime-aware analysis, multi-model comparison, and statistical significance testing.

### Methodology
- Build forecasting datasets from canonical VN data path.
- Compute fixed, time-series-safe features.
- Train a controlled set of models.
- Evaluate on walk-forward windows.
- Compare by forecast and strategy metrics.
- Apply bootstrap and DM tests.
- Report regime-wise robustness.

### Dataset scope
Vietnamese equity OHLCV data, chosen ticker universe, fixed historical period, fixed forecast horizons.

### Evaluation framework
- Walk-forward windows
- MAE, RMSE, directional accuracy, hit rate
- Cost-aware backtest metrics: Sharpe, Sortino, max drawdown, CAGR, turnover
- Regime-wise breakdown
- DM tests and bootstrap confidence intervals

### Expected tables and figures
- Data summary table
- Feature set table
- Model configuration table
- Walk-forward window diagram
- Main forecast metric table
- Regime-wise metric table
- DM-test matrix
- Bootstrap CI table/plot
- Equity curve and drawdown charts
- Stress-test summary table

### Required experiments
- Baseline vs regime-aware variants
- Baseline vs ensemble/stacking variants if retained
- Horizon sensitivity
- Cost sensitivity as secondary robustness test
- Regime robustness summary

### Expected limitations
- No proof of market inefficiency
- No production-readiness claim
- Limited novelty at the algorithm level
- Reproducibility depends on cleaning up environment assumptions

### What must be completed before thesis defense
- Final reproducible experiment protocol
- Final locked data split and feature specification
- Final significance tests and confidence intervals
- Final ablation of regime-aware vs non-regime-aware setup
- Honest limitation section about API/runtime contradictions and repo hygiene

---

# 8. Reviewer Criticism Simulation

## 8.1 Novelty criticism
**Reviewer:** “This is not a novel algorithm. It is an evaluation study.”

**My judgment:** Correct. The thesis should not claim algorithmic novelty. It should claim **evaluation rigor and empirical evidence**.

## 8.2 Data quality criticism
**Reviewer:** “How do I know your Vietnamese equity data pipeline is stable and free from hidden fallback behavior?”

**My judgment:** Valid criticism. The thesis must explicitly document data provenance and exclude mock/fallback runs from final evidence.

## 8.3 Methodology criticism
**Reviewer:** “You implemented too many modules. What exactly is the experiment?”

**My judgment:** Also valid. The thesis must narrow to one target, one evaluation protocol, one universe design, and a limited model comparison set.

## 8.4 Leakage criticism
**Reviewer:** “Time-series projects often leak future information through features or split design.”

**My judgment:** The core path looks relatively careful, especially in walk-forward and feature module contracts, but the thesis must still present the split and target construction transparently.

## 8.5 Backtest realism criticism
**Reviewer:** “You report strategy results, but are they realistic?”

**My judgment:** Partially defensible. Fee/slippage and non-overlap are present, which is good. But the thesis must avoid pretending this is a full production execution study.

## 8.6 Statistical significance criticism
**Reviewer:** “Do your performance differences survive statistical testing?”

**My judgment:** This is exactly why the thesis must actually use DM/bootstrap in final results, not merely cite that helper files exist.

## 8.7 Overengineering criticism
**Reviewer:** “This codebase contains allocation, routing, APIs, chat, and mock execution. Are you hiding weak forecasting under a big software shell?”

**My judgment:** Serious risk. The thesis should explicitly state that many modules are engineering scaffolding, not primary research contribution.

## 8.8 Weak claim criticism
**Reviewer:** “You say regime-aware and robust. Show the ablation.”

**My judgment:** Fair. Regime-aware vs non-regime-aware comparisons must be explicit.

## 8.9 Missing baselines criticism
**Reviewer:** “Where are the naive baselines?”

**My judgment:** Also fair. The thesis must include naive/persistence baselines and non-regime-aware baselines.

---

# 9. Roadmap to Make the Topic Defensible

## Phase 1 — Minimum defensible undergraduate thesis scope
### Objective
Narrow the repo into one clean empirical forecasting study.

### Tasks
- Fix ticker universe and date range.
- Choose one or two horizons.
- Freeze canonical feature set.
- Compare a small model family set.
- Use walk-forward evaluation only.

### Expected artifacts
- data scope table
- feature list
- split diagram
- forecast metric tables

### Acceptance criteria
- Every main result is reproducible and tied to a fixed protocol.

## Phase 2 — Strong master’s thesis / conference paper scope
### Objective
Add regime-aware comparison and statistical testing.

### Tasks
- Add regime-aware vs non-regime-aware ablation.
- Add bootstrap CIs and DM tests.
- Add stress-test and sensitivity summaries.
- Add cost-aware secondary strategy evaluation.

### Expected artifacts
- significance tables
- robustness tables
- regime-wise performance figures

### Acceptance criteria
- Main claims are statistically supported and robust across windows/regimes.

## Phase 3 — Overly ambitious scope that should be avoided for now
### Objective
Avoid turning the thesis into a pseudo-production platform claim.

### Tasks to avoid as primary thesis scope
- claiming full execution/trading readiness,
- claiming optimal portfolio construction,
- claiming explainable AI as the central novelty,
- claiming market inefficiency proof,
- claiming journal-grade contribution from software breadth alone.

### Expected artifacts
- none; this scope should be explicitly rejected.

### Acceptance criteria
- Thesis narrative stays narrow and defensible.

---

# 10. Final Verdict

## 10.1 Direct answers

1. **Can this project become an undergraduate thesis?**
   - **Yes.**

2. **Can this project become a master’s thesis?**
   - **Yes, if narrowed and statistically hardened.**

3. **Can this project become a conference paper?**
   - **Not yet, but possibly after scope narrowing and stronger final experiments.**

4. **Can this project become a journal paper?**
   - **No, not in its current state.**

5. **What is the best research title right now?**
   - **Walk-Forward Evaluation of Regime-Aware Multi-Model Forecasting for Vietnamese Equities**

6. **What is the safest research title?**
   - **A Walk-Forward Comparison of Machine Learning and Statistical Forecasting Models for Vietnamese Stocks**

7. **What is the highest-upside research title?**
   - **Walk-Forward Evaluation of Regime-Aware Multi-Model Forecasting for Vietnamese Equities**

8. **What title should be avoided?**
   - **An Intelligent Trading System for Vietnam**

9. **What parts of the system are academic contribution?**
   - walk-forward evaluation design,
   - multi-model benchmark comparison,
   - regime-aware robustness analysis,
   - cost-aware forecast-to-strategy evaluation,
   - bootstrap/DM-based statistical comparison.

10. **What parts are only engineering?**
   - large API/service shell,
   - chat/LLM/demo layers,
   - routing/allocation/runtime glue beyond what is needed for the research question,
   - repo-scale orchestration breadth.

11. **What must be fixed before making serious research claims?**
   - freeze scope,
   - lock data and features,
   - produce final ablations,
   - use significance testing in final tables,
   - separate mock/demo/runtime code from research evidence,
   - document reproducibility limitations honestly.

## 10.2 Final recommended research title

**English:** *Walk-Forward Evaluation of Regime-Aware Multi-Model Forecasting for Vietnamese Equities*

**Vietnamese:** *Đánh giá Walk-Forward cho Hệ Dự báo Đa Mô Hình Có Nhận diện Chế độ Thị trường trên Cổ phiếu Việt Nam*

## 10.3 Final judgment in one sentence

This repository is strong enough to become a **serious empirical forecasting thesis** centered on **walk-forward, regime-aware model evaluation for Vietnamese equities**, but it is **not yet defensible as a production trading system thesis or as a broad “AI for investing” research claim**.
