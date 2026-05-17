param(
    [ValidateSet(
        "hygiene",
        "preflight",
        "provider-policy",
        "test-provider",
        "test-metrics",
        "test-fast",
        "pycompile-active",
        "validate-all"
    )]
    [string]$Task = "validate-all"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$VenvPython = Join-Path $env:USERPROFILE ".venv\Scripts\python.exe"
$PycachePrefix = Join-Path $env:TEMP "pycache_vn_market_benchmark"

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )
    Write-Host "==> $Name"
    & $Command
}

function Invoke-Hygiene {
    Invoke-Step "hygiene" { python scripts/check_repo_hygiene.py }
}

function Invoke-Preflight {
    Invoke-Step "preflight" { python scripts/check_runtime_preflight.py }
}

function Invoke-PreflightVenv {
    Invoke-Step "preflight-venv" { & $VenvPython scripts/check_runtime_preflight.py }
}

function Invoke-ProviderPolicy {
    Invoke-Step "provider-policy" { & $VenvPython scripts/check_provider_usage_policy.py }
}

function Invoke-TestProvider {
    Invoke-Step "test-provider: provider usage policy" { & $VenvPython -m pytest tests/data/test_provider_usage_policy.py -q }
    Invoke-Step "test-provider: gateway contract" { & $VenvPython -m pytest tests/data/test_vn_price_gateway_contract.py -q }
}

function Invoke-TestMetrics {
    Invoke-Step "test-metrics" { & $VenvPython -m pytest tests/ml/test_directional_accuracy_metrics.py -q }
}

function Invoke-PycompileActive {
    Invoke-Step "pycompile-active" {
        $env:PYTHONPYCACHEPREFIX = $PycachePrefix
        & $VenvPython -m py_compile `
            scripts/research/vn30_hourly_2015_canonical_eval.py `
            scripts/research/run_vn30_daily_2015_benchmark.py `
            scripts/research/run_supported_indices_directional_benchmark.py `
            scripts/research/run_vn30_hourly_available_window_benchmark.py
    }
}

Push-Location $RepoRoot
try {
    switch ($Task) {
        "hygiene" { Invoke-Hygiene }
        "preflight" { Invoke-Preflight }
        "provider-policy" { Invoke-ProviderPolicy }
        "test-provider" { Invoke-TestProvider }
        "test-metrics" { Invoke-TestMetrics }
        "test-fast" {
            Invoke-ProviderPolicy
            Invoke-TestProvider
            Invoke-TestMetrics
        }
        "pycompile-active" { Invoke-PycompileActive }
        "validate-all" {
            Invoke-Hygiene
            Invoke-Preflight
            Invoke-PreflightVenv
            Invoke-ProviderPolicy
            Invoke-TestProvider
            Invoke-TestMetrics
            Invoke-PycompileActive
        }
    }
}
finally {
    Pop-Location
}
