import pytest
import numpy as np

from src.ml.risk import MonteCarloRiskSimulator

def test_monte_carlo_reproducibility():
    sim1 = MonteCarloRiskSimulator(simulations=5000, random_seed=42)
    res1 = sim1.simulate_risk(forecast_mean=0.05, volatility_proxy=0.02)
    
    sim2 = MonteCarloRiskSimulator(simulations=5000, random_seed=42)
    res2 = sim2.simulate_risk(forecast_mean=0.05, volatility_proxy=0.02)
    
    sim3 = MonteCarloRiskSimulator(simulations=5000, random_seed=99)
    res3 = sim3.simulate_risk(forecast_mean=0.05, volatility_proxy=0.02)
    
    # Assert deterministic equality for same seed
    assert res1["var"]["95.0"] == res2["var"]["95.0"]
    assert res1["cvar"]["99.0"] == res2["cvar"]["99.0"]
    assert res1["summary"]["mean"] == res2["summary"]["mean"]
    
    # Assert different seed yields different paths
    assert res1["var"]["95.0"] != res3["var"]["95.0"]
    
def test_monte_carlo_interface_bounds():
    sim = MonteCarloRiskSimulator()
    
    with pytest.raises(ValueError, match="Volatility proxy must be strictly positive"):
        sim.simulate_risk(forecast_mean=0.0, volatility_proxy=-0.1)
        
    with pytest.raises(ValueError, match="Confidence levels must be in range"):
        sim.simulate_risk(forecast_mean=0.0, volatility_proxy=0.01, confidence_levels=[1.05])
        
def test_monte_carlo_output_structure():
    sim = MonteCarloRiskSimulator(simulations=1000)
    res = sim.simulate_risk(forecast_mean=0.1, volatility_proxy=0.05, horizon="mid")
    
    # Check output dictionary keys representing interface completeness
    assert "var" in res
    assert "cvar" in res
    assert "summary" in res
    assert "metadata" in res
    
    assert "95.0" in res["var"]
    assert "99.0" in res["var"]
    
    assert res["summary"]["mean"] == pytest.approx(0.1, abs=0.01)
    
    assert res["metadata"]["horizon"] == "mid"
    assert res["metadata"]["random_seed"] == 42
    assert res["metadata"]["simulations"] == 1000
    assert res["metadata"]["risk_model_type"] == "residual_normal_scenario_simulation"
    assert res["metadata"]["calibration_status"] == "heuristic_not_calibrated"
    assert "not calibrated forecast confidence" in res["metadata"]["assumptions"]
