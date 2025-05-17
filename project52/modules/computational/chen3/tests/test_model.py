"""
Tests for the core Chen3 model functionality.
"""
import pytest
import numpy as np
from chen3 import Chen3Model
from chen3.pricing import EuropeanCall

def test_model_initialization(model_params):
    """Test model initialization with valid parameters."""
    model = Chen3Model(model_params)
    assert model is not None
    assert model.params == model_params

def test_european_call_pricing(model_params, european_call_params, european_call_bs_price):
    """Test European call option pricing."""
    model = Chen3Model(model_params)
    option = EuropeanCall(
        strike=european_call_params['K'],
        maturity=european_call_params['T']
    )
    
    # Test Monte Carlo pricing
    price_mc = model.price(option, n_paths=10000)
    assert price_mc > 0
    assert not np.isnan(price_mc)
    assert not np.isinf(price_mc)
    
    # Compare with Black-Scholes price (should be close for simple case)
    assert abs(price_mc - european_call_bs_price) < 0.5

def test_model_simulation(model_params):
    """Test model simulation functionality."""
    model = Chen3Model(model_params)
    T = 1.0
    n_steps = 100
    n_paths = 1000
    
    # Simulate paths
    paths = model.simulate(T, n_steps, n_paths)
    
    # Check output shape
    assert paths.shape == (n_paths, n_steps + 1, 3)  # 3 factors: rate, equity, variance
    
    # Check initial values
    assert np.allclose(paths[:, 0, 0], model_params.rate.r0)  # Initial rate
    assert np.allclose(paths[:, 0, 1], model_params.equity.S0)  # Initial stock price
    assert np.allclose(paths[:, 0, 2], model_params.equity.v0)  # Initial variance
    
    # Check for NaN and infinite values
    assert not np.any(np.isnan(paths))
    assert not np.any(np.isinf(paths))
    
    # Check that variance is always positive
    assert np.all(paths[:, :, 2] > 0) 