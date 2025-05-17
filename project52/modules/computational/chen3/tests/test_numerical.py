"""
Tests for numerical methods in the Chen3 model.
"""
import pytest
import numpy as np
from chen3 import Chen3Model
from chen3.pricing import EuropeanCall
from chen3.numerical import (
    MonteCarloEngine,
    GPUMonteCarloEngine,
    QuasiMonteCarloEngine
)

def test_monte_carlo_convergence(model_params, european_call_params, european_call_bs_price):
    """Test convergence of Monte Carlo method."""
    model = Chen3Model(model_params)
    option = EuropeanCall(
        strike=european_call_params['K'],
        maturity=european_call_params['T']
    )
    
    # Test convergence with increasing number of paths
    n_paths_list = [1000, 10000, 100000]
    prices = []
    
    for n_paths in n_paths_list:
        price = model.price(option, n_paths=n_paths)
        prices.append(price)
        
        # Basic validation
        assert price > 0
        assert not np.isnan(price)
        assert not np.isinf(price)
    
    # Check convergence to Black-Scholes price
    for price in prices:
        assert abs(price - european_call_bs_price) < 0.1

@pytest.mark.gpu
def test_gpu_monte_carlo(model_params, european_call_params):
    """Test GPU-accelerated Monte Carlo simulation."""
    model = Chen3Model(model_params)
    option = EuropeanCall(
        strike=european_call_params['K'],
        maturity=european_call_params['T']
    )
    
    # Test GPU pricing
    price = model.price(option, n_paths=10000, engine=GPUMonteCarloEngine())
    
    # Basic validation
    assert price > 0
    assert not np.isnan(price)
    assert not np.isinf(price)

def test_quasi_monte_carlo(model_params, european_call_params):
    """Test Quasi-Monte Carlo method using Sobol sequences."""
    model = Chen3Model(model_params)
    option = EuropeanCall(
        strike=european_call_params['K'],
        maturity=european_call_params['T']
    )
    
    # Test standard Monte Carlo
    price_mc = model.price(option, n_paths=10000)
    
    # Test Quasi-Monte Carlo
    price_qmc = model.price(option, n_paths=10000, engine=QuasiMonteCarloEngine())
    
    # Basic validation
    assert price_qmc > 0
    assert not np.isnan(price_qmc)
    assert not np.isinf(price_qmc)
    
    # QMC should have lower variance than standard MC
    assert abs(price_qmc - price_mc) < 0.1 