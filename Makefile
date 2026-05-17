.PHONY: help hygiene preflight preflight-venv provider-policy test-provider test-metrics test-fast pycompile-active validate-all

VENV_PY ?= $(USERPROFILE)\.venv\Scripts\python.exe
PYCACHE_PREFIX := $(TEMP)\pycache_vn_market_benchmark

help:
	@echo Safe validation targets:
	@echo   hygiene
	@echo   preflight
	@echo   preflight-venv
	@echo   provider-policy
	@echo   test-provider
	@echo   test-metrics
	@echo   test-fast
	@echo   pycompile-active
	@echo   validate-all

hygiene:
	python scripts/check_repo_hygiene.py

preflight:
	python scripts/check_runtime_preflight.py

preflight-venv:
	$(VENV_PY) scripts/check_runtime_preflight.py

provider-policy:
	$(VENV_PY) scripts/check_provider_usage_policy.py

test-provider:
	$(VENV_PY) -m pytest tests/data/test_provider_usage_policy.py -q
	$(VENV_PY) -m pytest tests/data/test_vn_price_gateway_contract.py -q

test-metrics:
	$(VENV_PY) -m pytest tests/ml/test_directional_accuracy_metrics.py -q

test-fast: provider-policy test-provider test-metrics

pycompile-active:
	PowerShell -NoProfile -Command "$$env:PYTHONPYCACHEPREFIX='$(PYCACHE_PREFIX)'; & '$(VENV_PY)' -m py_compile scripts/research/vn30_hourly_2015_canonical_eval.py scripts/research/run_vn30_daily_2015_benchmark.py scripts/research/run_supported_indices_directional_benchmark.py scripts/research/run_vn30_hourly_available_window_benchmark.py"

validate-all: hygiene preflight preflight-venv provider-policy test-provider test-metrics pycompile-active
