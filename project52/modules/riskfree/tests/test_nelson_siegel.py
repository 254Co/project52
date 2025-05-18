"""Tests for the Nelson-Siegel model implementation."""
import pytest
import numpy as np
from riskfree.model.nelson_siegel import fit_nelson_siegel, ns_zero

def test_ns_zero_basic():
    """Test basic Nelson-Siegel zero rate calculation."""
    # Test with standard parameters
    beta = [0.05, -0.02, 0.01]  # [level, slope, curvature]
    tau = 2.0  # decay parameter
    
    # Test at t=0
    rate = ns_zero(0, *beta, tau)
    assert isinstance(rate, float)
    assert rate == beta[0] + beta[1]  # At t=0, only level and slope components matter
    
    # Test at t=infinity
    rate_inf = ns_zero(1000, *beta, tau)
    assert isinstance(rate_inf, float)
    assert abs(rate_inf - beta[0]) < 1e-10  # At t=infinity, only level component matters
    
    # Test at t=tau
    rate_tau = ns_zero(tau, *beta, tau)
    assert isinstance(rate_tau, float)
    assert rate_tau > 0

def test_ns_zero_components():
    """Test individual components of the Nelson-Siegel model."""
    beta = [0.05, -0.02, 0.01]
    tau = 2.0
    
    # Test level component (beta0)
    rate_level = ns_zero(1.0, beta[0], 0, 0, tau)
    assert abs(rate_level - beta[0]) < 1e-10
    
    # Test slope component (beta1)
    rate_slope = ns_zero(1.0, 0, beta[1], 0, tau)
    assert isinstance(rate_slope, float)
    assert rate_slope < 0  # For negative beta1
    
    # Test curvature component (beta2)
    rate_curv = ns_zero(1.0, 0, 0, beta[2], tau)
    assert isinstance(rate_curv, float)
    assert rate_curv > 0  # For positive beta2

def test_ns_zero_monotonicity():
    """Test that the Nelson-Siegel model produces reasonable rate behavior."""
    beta = [0.05, -0.02, 0.01]
    tau = 2.0
    
    # Test rates at different tenors
    tenors = np.linspace(0, 30, 100)
    rates = [ns_zero(t, *beta, tau) for t in tenors]
    
    # Check that rates are non-negative
    assert all(r >= 0 for r in rates)
    
    # Check that rates approach the level parameter
    assert abs(rates[-1] - beta[0]) < 1e-3

def test_fit_nelson_siegel():
    """Test fitting the Nelson-Siegel model to data."""
    # Create synthetic data
    tenors = [1, 2, 3, 5, 7, 10, 20, 30]
    true_beta = [0.05, -0.02, 0.01]
    tau = 2.0
    
    # Generate rates using the true parameters
    rates = {t: ns_zero(t, *true_beta, tau) for t in tenors}
    
    # Fit the model
    fitted_beta = fit_nelson_siegel(list(tenors), list(rates.values()))
    
    # Check that we got the expected number of parameters
    assert len(fitted_beta) == 3
    
    # Check that fitted parameters are reasonable
    assert all(isinstance(b, float) for b in fitted_beta)
    assert fitted_beta[0] > 0  # Level should be positive
    
    # Check that fitted curve is close to original data
    fitted_rates = {t: ns_zero(t, *fitted_beta, tau) for t in tenors}
    max_diff = max(abs(rates[t] - fitted_rates[t]) for t in tenors)
    assert max_diff < 0.01  # Maximum 1% difference

def test_fit_nelson_siegel_edge_cases():
    """Test fitting the Nelson-Siegel model with edge cases."""
    # Test with single point
    with pytest.raises(ValueError):
        fit_nelson_siegel([1], [0.05])
    
    # Test with invalid data
    with pytest.raises(ValueError):
        fit_nelson_siegel([1, 2], [0.05])  # Mismatched lengths
    
    # Test with negative rates
    with pytest.raises(ValueError):
        fit_nelson_siegel([1, 2, 3], [-0.05, -0.04, -0.03])

def test_ns_zero_parameter_validation():
    """Test parameter validation in the Nelson-Siegel zero rate calculation."""
    # Test with invalid tau
    with pytest.raises(ValueError):
        ns_zero(1.0, 0.05, -0.02, 0.01, 0)  # tau = 0
    
    # Test with negative tenor
    with pytest.raises(ValueError):
        ns_zero(-1.0, 0.05, -0.02, 0.01, 2.0)
    
    # Test with invalid beta parameters
    with pytest.raises(ValueError):
        ns_zero(1.0, -0.05, -0.02, 0.01, 2.0)  # Negative level 