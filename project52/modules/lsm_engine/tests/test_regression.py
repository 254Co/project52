"""
Test suite for the polynomial regression module.

This module contains tests for the polynomial regression functionality
used in the Longstaff-Schwartz Method for estimating continuation values.
"""

# File: tests/test_regression.py
import numpy as np
from lsm_engine.engine.regression import PolyRegressor

def test_polynomial_regression():
    """
    Test the polynomial regression on a simple linear function.
    
    This test verifies that the PolyRegressor can accurately fit and predict
    a simple linear function (y = 2x + 1) using polynomial basis expansion.
    
    Test setup:
    - Input: 10 evenly spaced points in [0, 1]
    - Target: Linear function y = 2x + 1
    - Polynomial degree: 1 (linear)
    - Tolerance: 1e-6 for numerical comparison
    """
    x = np.linspace(0, 1, 10)[:, None]
    y = 2 * x.flatten() + 1
    reg = PolyRegressor(degree=1)
    reg.fit(x, y)
    yc = reg.predict(x)
    assert np.allclose(yc, y, atol=1e-6)