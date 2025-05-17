"""
Tests for parameter validation in the Chen3 model.
"""
import pytest
import numpy as np
from chen3 import RateParams, EquityParams
from chen3.correlation import TimeDependentCorrelation

def test_rate_params_validation():
    """Test validation of interest rate parameters."""
    # Test valid parameters
    valid_params = RateParams(
        kappa=0.1,
        theta=0.05,
        sigma=0.1,
        r0=0.03
    )
    assert valid_params.kappa > 0
    assert valid_params.theta > 0
    assert valid_params.sigma > 0
    assert valid_params.r0 > 0
    
    # Test invalid parameters
    with pytest.raises(ValueError):
        RateParams(
            kappa=-0.1,  # Negative mean reversion
            theta=0.05,
            sigma=0.1,
            r0=0.03
        )
    
    with pytest.raises(ValueError):
        RateParams(
            kappa=0.1,
            theta=-0.05,  # Negative long-term mean
            sigma=0.1,
            r0=0.03
        )

def test_equity_params_validation():
    """Test validation of equity parameters."""
    # Test valid parameters
    valid_params = EquityParams(
        mu=0.05,
        q=0.02,
        S0=100.0,
        v0=0.04,
        kappa_v=2.0,
        theta_v=0.04,
        sigma_v=0.3
    )
    assert valid_params.S0 > 0
    assert valid_params.v0 > 0
    assert valid_params.kappa_v > 0
    assert valid_params.theta_v > 0
    assert valid_params.sigma_v > 0
    
    # Test invalid parameters
    with pytest.raises(ValueError):
        EquityParams(
            mu=0.05,
            q=0.02,
            S0=-100.0,  # Negative initial price
            v0=0.04,
            kappa_v=2.0,
            theta_v=0.04,
            sigma_v=0.3
        )
    
    with pytest.raises(ValueError):
        EquityParams(
            mu=0.05,
            q=0.02,
            S0=100.0,
            v0=-0.04,  # Negative initial variance
            kappa_v=2.0,
            theta_v=0.04,
            sigma_v=0.3
        )

def test_feller_condition():
    """Test Feller condition for CIR process."""
    # Test valid parameters (satisfying Feller condition)
    valid_params = EquityParams(
        mu=0.05,
        q=0.02,
        S0=100.0,
        v0=0.04,
        kappa_v=2.0,
        theta_v=0.04,
        sigma_v=0.3
    )
    assert 2 * valid_params.kappa_v * valid_params.theta_v > valid_params.sigma_v**2
    
    # Test invalid parameters (violating Feller condition)
    with pytest.raises(ValueError):
        EquityParams(
            mu=0.05,
            q=0.02,
            S0=100.0,
            v0=0.04,
            kappa_v=0.1,  # Too small mean reversion
            theta_v=0.04,
            sigma_v=0.3    # Too large volatility
        )

def test_correlation_validation():
    """Test validation of correlation parameters."""
    # Test valid correlation matrix
    valid_matrix = np.array([[1.0, 0.5, 0.3],
                           [0.5, 1.0, 0.2],
                           [0.3, 0.2, 1.0]])
    
    time_corr = TimeDependentCorrelation(
        time_points=np.array([0.0, 1.0]),
        correlation_matrices=[valid_matrix, valid_matrix]
    )
    
    # Test invalid correlation matrix (not positive definite)
    invalid_matrix = np.array([[1.0, 0.9, 0.9],
                             [0.9, 1.0, 0.9],
                             [0.9, 0.9, 1.0]])
    
    with pytest.raises(ValueError):
        TimeDependentCorrelation(
            time_points=np.array([0.0, 1.0]),
            correlation_matrices=[invalid_matrix, invalid_matrix]
        ) 