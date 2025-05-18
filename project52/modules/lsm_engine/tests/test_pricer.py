"""
Test suite for the LSM pricing engine.

This module contains tests for the Longstaff-Schwartz Method pricing engine,
verifying its functionality and numerical properties.
"""

# File: tests/test_pricer.py
import numpy as np
from math import exp
from lsm_engine.engine.pricer import LSMPricer
from lsm_engine.engine.pathgen import generate_gbm_paths
from lsm_engine.payoffs.vanilla import american_put

def test_price_shape():
    """
    Test the shape and bounds of the LSM price output.
    
    This test verifies that:
    1. The price is a float
    2. The price is non-negative
    3. The price is bounded above by the strike price
    
    Test parameters:
    - Initial stock price (S0) = 100
    - Strike price (K) = 100
    - Risk-free rate (r) = 5%
    - Volatility (σ) = 20%
    - Time to maturity (T) = 0.5 years
    - Number of steps = 3
    - Number of paths = 1000
    """
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 0.5
    n_steps, n_paths = 3, 1000
    paths = generate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=0)
    exercise_dates = list(range(1, n_steps+1))
    discount = exp(-r * T / n_steps)
    pricer = LSMPricer(american_put(K), discount)
    pv = pricer.price(paths, exercise_dates)
    assert isinstance(pv, float)
    assert 0 <= pv <= K