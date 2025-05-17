"""
Tests for parameter validation in the Chen3 model.
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st

from chen3 import EquityParams, RateParams
from chen3.correlation import TimeDependentCorrelation
from chen3.datatypes import ModelParams
from chen3.utils.exceptions import ValidationError


def test_rate_params_validation():
    """Test validation of rate parameters."""
    # Test valid parameters
    valid_params = RateParams(
        kappa=1.0,  # Mean reversion speed
        theta=0.05,  # Long-term mean
        sigma=0.1,  # Volatility
        r0=0.03,  # Initial rate
    )
    assert valid_params.kappa == 1.0
    assert valid_params.theta == 0.05
    assert valid_params.sigma == 0.1
    assert valid_params.r0 == 0.03

    # Test invalid parameters
    with pytest.raises(ValidationError):
        RateParams(kappa=-1.0, theta=0.05, sigma=0.1, r0=0.03)  # Negative mean reversion

    with pytest.raises(ValidationError):
        RateParams(kappa=1.0, theta=-0.05, sigma=0.1, r0=0.03)  # Negative long-term mean

    with pytest.raises(ValidationError):
        RateParams(kappa=1.0, theta=0.05, sigma=-0.1, r0=0.03)  # Negative volatility

    with pytest.raises(ValidationError):
        RateParams(kappa=1.0, theta=0.05, sigma=0.1, r0=-0.03)  # Negative initial rate

    # Test Feller condition
    with pytest.raises(ValidationError):
        RateParams(kappa=0.1, theta=0.05, sigma=0.3, r0=0.03)  # Violates Feller condition


def test_equity_params_validation():
    """Test validation of equity parameters."""
    # Test valid parameters
    valid_params = EquityParams(
        mu=0.05,  # Drift
        q=0.02,  # Dividend yield
        S0=100.0,  # Initial stock price
        v0=0.04,  # Initial variance
        kappa_v=2.0,  # Variance mean reversion speed
        theta_v=0.04,  # Variance long-term mean
        sigma_v=0.3,  # Variance volatility
    )
    assert valid_params.mu == 0.05
    assert valid_params.q == 0.02
    assert valid_params.S0 == 100.0
    assert valid_params.v0 == 0.04
    assert valid_params.kappa_v == 2.0
    assert valid_params.theta_v == 0.04
    assert valid_params.sigma_v == 0.3

    # Test invalid parameters
    with pytest.raises(ValidationError):
        EquityParams(
            mu=-0.05, q=0.02, S0=100.0, v0=0.04, kappa_v=2.0, theta_v=0.04, sigma_v=0.3
        )  # Negative drift

    with pytest.raises(ValidationError):
        EquityParams(
            mu=0.05, q=-0.02, S0=100.0, v0=0.04, kappa_v=2.0, theta_v=0.04, sigma_v=0.3
        )  # Negative dividend yield

    with pytest.raises(ValidationError):
        EquityParams(
            mu=0.05, q=0.02, S0=-100.0, v0=0.04, kappa_v=2.0, theta_v=0.04, sigma_v=0.3
        )  # Negative initial stock price

    with pytest.raises(ValidationError):
        EquityParams(
            mu=0.05, q=0.02, S0=100.0, v0=-0.04, kappa_v=2.0, theta_v=0.04, sigma_v=0.3
        )  # Negative initial variance

    with pytest.raises(ValidationError):
        EquityParams(
            mu=0.05, q=0.02, S0=100.0, v0=0.04, kappa_v=-2.0, theta_v=0.04, sigma_v=0.3
        )  # Negative variance mean reversion

    with pytest.raises(ValidationError):
        EquityParams(
            mu=0.05, q=0.02, S0=100.0, v0=0.04, kappa_v=2.0, theta_v=-0.04, sigma_v=0.3
        )  # Negative variance long-term mean

    with pytest.raises(ValidationError):
        EquityParams(
            mu=0.05, q=0.02, S0=100.0, v0=0.04, kappa_v=2.0, theta_v=0.04, sigma_v=-0.3
        )  # Negative variance volatility

    # Test Feller condition for variance process
    with pytest.raises(ValidationError):
        EquityParams(
            mu=0.05, q=0.02, S0=100.0, v0=0.04, kappa_v=0.1, theta_v=0.04, sigma_v=0.3
        )  # Violates Feller condition


def test_feller_condition():
    """Test Feller condition for CIR process."""
    # Test valid parameters (satisfying Feller condition)
    valid_params = EquityParams(
        mu=0.05, q=0.02, S0=100.0, v0=0.04, kappa_v=2.0, theta_v=0.04, sigma_v=0.3
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
            sigma_v=0.3,  # Too large volatility
        )


def test_correlation_validation():
    """Test validation of correlation parameters."""
    # Test valid correlation matrix
    valid_matrix = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])

    time_corr = TimeDependentCorrelation(
        time_points=np.array([0.0, 1.0]),
        correlation_matrices=[valid_matrix, valid_matrix],
    )

    # Test invalid correlation matrix (not positive definite)
    invalid_matrix = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, 0.9], [0.9, 0.9, 1.0]])

    with pytest.raises(ValueError):
        TimeDependentCorrelation(
            time_points=np.array([0.0, 1.0]),
            correlation_matrices=[invalid_matrix, invalid_matrix],
        )


def test_model_params_validation():
    """Test validation of model parameters."""
    # Test valid parameters
    rate_params = RateParams(kappa=1.0, theta=0.05, sigma=0.1, r0=0.03)
    equity_params = EquityParams(
        mu=0.05, q=0.02, S0=100.0, v0=0.04, kappa_v=2.0, theta_v=0.04, sigma_v=0.3
    )
    valid_params = ModelParams(rate=rate_params, equity=equity_params, correlation=None)
    assert valid_params.rate == rate_params
    assert valid_params.equity == equity_params
    assert valid_params.correlation is None

    # Test invalid parameters
    with pytest.raises(ValidationError):
        ModelParams(rate=None, equity=equity_params, correlation=None)  # Missing rate params

    with pytest.raises(ValidationError):
        ModelParams(rate=rate_params, equity=None, correlation=None)  # Missing equity params

    # Test with invalid correlation matrix
    invalid_corr = np.array([[1.0, 0.5], [0.5, 1.0]])  # Wrong shape
    with pytest.raises(ValidationError):
        ModelParams(rate=rate_params, equity=equity_params, correlation=invalid_corr)


def test_parameter_properties():
    """Test properties of model parameters."""
    # Test rate parameter properties
    rate_params = RateParams(kappa=1.0, theta=0.05, sigma=0.1, r0=0.03)
    assert rate_params.feller_condition_satisfied
    assert rate_params.mean_reversion_time == 1.0 / rate_params.kappa
    assert rate_params.volatility_scale == rate_params.sigma / np.sqrt(2 * rate_params.kappa)

    # Test equity parameter properties
    equity_params = EquityParams(
        mu=0.05, q=0.02, S0=100.0, v0=0.04, kappa_v=2.0, theta_v=0.04, sigma_v=0.3
    )
    assert equity_params.feller_condition_satisfied
    assert equity_params.mean_reversion_time == 1.0 / equity_params.kappa_v
    assert equity_params.volatility_scale == equity_params.sigma_v / np.sqrt(2 * equity_params.kappa_v)
    assert equity_params.total_return == equity_params.mu - equity_params.q

    # Test model parameter properties
    model_params = ModelParams(rate=rate_params, equity=equity_params, correlation=None)
    assert model_params.n_factors == 3  # Rate, stock, variance
    assert model_params.has_correlation is False
    assert model_params.is_time_dependent is False
    assert model_params.is_state_dependent is False
    assert model_params.is_regime_switching is False
    assert model_params.is_stochastic is False
    assert model_params.is_copula is False


@given(
    kappa=st.floats(min_value=0.1, max_value=10.0),
    theta=st.floats(min_value=0.01, max_value=0.1),
    sigma=st.floats(min_value=0.01, max_value=0.5),
    r0=st.floats(min_value=0.01, max_value=0.1),
)
def test_rate_params_properties(kappa, theta, sigma, r0):
    """Test properties of rate parameters with hypothesis."""
    # Ensure Feller condition is satisfied
    if 2 * kappa * theta > sigma**2:
        params = RateParams(kappa=kappa, theta=theta, sigma=sigma, r0=r0)
        assert params.feller_condition_satisfied
        assert params.mean_reversion_time == 1.0 / kappa
        assert params.volatility_scale == sigma / np.sqrt(2 * kappa)


@given(
    mu=st.floats(min_value=-0.1, max_value=0.2),
    q=st.floats(min_value=0.0, max_value=0.1),
    S0=st.floats(min_value=1.0, max_value=1000.0),
    v0=st.floats(min_value=0.01, max_value=0.5),
    kappa_v=st.floats(min_value=0.1, max_value=10.0),
    theta_v=st.floats(min_value=0.01, max_value=0.5),
    sigma_v=st.floats(min_value=0.01, max_value=0.5),
)
def test_equity_params_properties(mu, q, S0, v0, kappa_v, theta_v, sigma_v):
    """Test properties of equity parameters with hypothesis."""
    # Ensure Feller condition is satisfied
    if 2 * kappa_v * theta_v > sigma_v**2:
        params = EquityParams(
            mu=mu, q=q, S0=S0, v0=v0, kappa_v=kappa_v, theta_v=theta_v, sigma_v=sigma_v
        )
        assert params.feller_condition_satisfied
        assert params.mean_reversion_time == 1.0 / kappa_v
        assert params.volatility_scale == sigma_v / np.sqrt(2 * kappa_v)
        assert params.total_return == mu - q
