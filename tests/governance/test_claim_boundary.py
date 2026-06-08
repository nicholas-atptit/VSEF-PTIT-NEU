from src.governance.claim_boundary import claim_statement


def test_forecast_engine_claim_boundary_blocks_deployment_claims():
    statement = claim_statement()
    assert "no trading" in statement
    assert "no BUY/SELL" in statement
    assert "no production" in statement
