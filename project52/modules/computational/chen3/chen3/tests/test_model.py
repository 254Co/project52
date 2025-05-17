"""
Test suite for the Chen3 model implementation.

This module contains tests for the core model functionality, including
parameter validation, simulation, and option pricing.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

from .. import datatypes, model
from ..utils.exceptions import ValidationError


@pytest.fixture
def default_rate_params():
    """Create default rate parameters for testing."""
    return datatypes.RateParams(
        kappa=0.1,
        theta=0.05,
        sigma=0.1,
        r0=0.03,
    )


@pytest.fixture
def default_equity_params():
    """Create default equity parameters for testing."""
    return datatypes.EquityParams(
        mu=0.05,
        q=0.02,
        S0=100.0,
        v0=0.04,
        kappa_v=2.0,
        theta_v=0.04,
        sigma_v=0.3,
    )


@pytest.fixture
def default_model_params(default_rate_params, default_equity_params):
    """Create default model parameters for testing."""
    return datatypes.ModelParams(
        rate_params=default_rate_params,
        equity_params=default_equity_params,
        correlation_type="constant",
        correlation_params={"rho": 0.5},
    )


@pytest.fixture
def chen_model(default_model_params):
    """Create a Chen model instance for testing."""
    return model.ChenModel(default_model_params)


def test_rate_params_validation():
    """Test rate parameter validation."""
    # Test valid parameters
    params = datatypes.RateParams(
        kappa=0.1,
        theta=0.05,
        sigma=0.1,
        r0=0.03,
    )
    params.validate_parameters()

    # Test invalid kappa
    with pytest.raises(ValidationError):
        datatypes.RateParams(
            kappa=-0.1,
            theta=0.05,
            sigma=0.1,
            r0=0.03,
        )

    # Test invalid theta
    with pytest.raises(ValidationError):
        datatypes.RateParams(
            kappa=0.1,
            theta=-0.05,
            sigma=0.1,
            r0=0.03,
        )

    # Test invalid sigma
    with pytest.raises(ValidationError):
        datatypes.RateParams(
            kappa=0.1,
            theta=0.05,
            sigma=-0.1,
            r0=0.03,
        )

    # Test invalid r0
    with pytest.raises(ValidationError):
        datatypes.RateParams(
            kappa=0.1,
            theta=0.05,
            sigma=0.1,
            r0=-0.03,
        )

    # Test Feller condition
    with pytest.raises(ValidationError):
        datatypes.RateParams(
            kappa=0.1,
            theta=0.05,
            sigma=0.3,  # Too large for Feller condition
            r0=0.03,
        )


def test_equity_params_validation():
    """Test equity parameter validation."""
    # Test valid parameters
    params = datatypes.EquityParams(
        mu=0.05,
        q=0.02,
        S0=100.0,
        v0=0.04,
        kappa_v=2.0,
        theta_v=0.04,
        sigma_v=0.3,
    )
    params.validate_parameters()

    # Test invalid mu
    with pytest.raises(ValidationError):
        datatypes.EquityParams(
            mu=2.0,  # Too large
            q=0.02,
            S0=100.0,
            v0=0.04,
            kappa_v=2.0,
            theta_v=0.04,
            sigma_v=0.3,
        )

    # Test invalid q
    with pytest.raises(ValidationError):
        datatypes.EquityParams(
            mu=0.05,
            q=-0.02,
            S0=100.0,
            v0=0.04,
            kappa_v=2.0,
            theta_v=0.04,
            sigma_v=0.3,
        )

    # Test invalid S0
    with pytest.raises(ValidationError):
        datatypes.EquityParams(
            mu=0.05,
            q=0.02,
            S0=-100.0,
            v0=0.04,
            kappa_v=2.0,
            theta_v=0.04,
            sigma_v=0.3,
        )

    # Test invalid v0
    with pytest.raises(ValidationError):
        datatypes.EquityParams(
            mu=0.05,
            q=0.02,
            S0=100.0,
            v0=-0.04,
            kappa_v=2.0,
            theta_v=0.04,
            sigma_v=0.3,
        )

    # Test Feller condition for variance
    with pytest.raises(ValidationError):
        datatypes.EquityParams(
            mu=0.05,
            q=0.02,
            S0=100.0,
            v0=0.04,
            kappa_v=2.0,
            theta_v=0.04,
            sigma_v=1.0,  # Too large for Feller condition
        )


def test_model_initialization(default_model_params):
    """Test model initialization."""
    model = model.ChenModel(default_model_params)
    assert model.params == default_model_params
    assert model.num_factors == 3


def test_simulation(chen_model):
    """Test model simulation."""
    # Run simulation
    num_paths = 1000
    num_steps = 252
    paths = chen_model.simulate(num_paths=num_paths, num_steps=num_steps)

    # Check output shape
    assert paths.shape == (num_paths, num_steps + 1, 3)

    # Check initial values
    assert_array_almost_equal(
        paths[:, 0, 0],
        chen_model.params.equity_params.S0,
        decimal=5,
    )
    assert_array_almost_equal(
        paths[:, 0, 1],
        chen_model.params.equity_params.v0,
        decimal=5,
    )
    assert_array_almost_equal(
        paths[:, 0, 2],
        chen_model.params.rate_params.r0,
        decimal=5,
    )

    # Check positivity of stock price and variance
    assert np.all(paths[:, :, 0] > 0)
    assert np.all(paths[:, :, 1] > 0)


def test_option_pricing(chen_model):
    """Test option pricing."""
    # Price European call option
    strike = 100.0
    maturity = 1.0
    price = chen_model.price_option(
        option_type="call",
        strike=strike,
        maturity=maturity,
    )

    # Check price is positive
    assert price > 0

    # Check put-call parity
    call_price = chen_model.price_option(
        option_type="call",
        strike=strike,
        maturity=maturity,
    )
    put_price = chen_model.price_option(
        option_type="put",
        strike=strike,
        maturity=maturity,
    )
    forward_price = chen_model.params.equity_params.S0 * np.exp(
        -chen_model.params.equity_params.q * maturity
    )
    discount_factor = np.exp(-chen_model.params.rate_params.r0 * maturity)
    assert_allclose(
        call_price - put_price,
        forward_price - strike * discount_factor,
        rtol=1e-2,
    )


def test_correlation_structures(chen_model):
    """Test different correlation structures."""
    # Test constant correlation
    model = chen_model
    assert model.correlation_matrix.shape == (3, 3)
    assert np.all(np.diag(model.correlation_matrix) == 1.0)
    assert np.all(np.abs(model.correlation_matrix) <= 1.0)

    # Test time-dependent correlation
    params = model.params.copy()
    params.correlation_type = "time-dependent"
    params.correlation_params = {
        "rho0": 0.5,
        "rho1": 0.3,
        "alpha": 0.1,
    }
    model = model.ChenModel(params)
    assert model.correlation_matrix.shape == (3, 3)

    # Test stochastic correlation
    params = model.params.copy()
    params.correlation_type = "stochastic"
    params.correlation_params = {
        "rho0": 0.5,
        "kappa_rho": 1.0,
        "theta_rho": 0.5,
        "sigma_rho": 0.1,
    }
    model = model.ChenModel(params)
    assert model.correlation_matrix.shape == (3, 3)


def test_numerical_methods(chen_model):
    """Test different numerical methods."""
    # Test Euler method
    paths_euler = chen_model.simulate(
        num_paths=100,
        num_steps=252,
        method="euler",
    )

    # Test Milstein method
    paths_milstein = chen_model.simulate(
        num_paths=100,
        num_steps=252,
        method="milstein",
    )

    # Check both methods produce similar results
    assert_allclose(
        paths_euler[:, -1, 0],
        paths_milstein[:, -1, 0],
        rtol=0.1,
    )


def test_risk_metrics(chen_model):
    """Test risk metrics calculation."""
    # Calculate risk metrics
    metrics = chen_model.calculate_risk_metrics(
        option_type="call",
        strike=100.0,
        maturity=1.0,
    )

    # Check metrics
    assert "delta" in metrics
    assert "gamma" in metrics
    assert "vega" in metrics
    assert "theta" in metrics
    assert "rho" in metrics

    # Check delta is between 0 and 1 for call option
    assert 0 <= metrics["delta"] <= 1

    # Check gamma is positive
    assert metrics["gamma"] > 0


def test_calibration(chen_model):
    """Test model calibration."""
    # Create market data
    market_data = {
        "strikes": [90.0, 100.0, 110.0],
        "maturities": [0.25, 0.5, 1.0],
        "prices": np.array([
            [10.0, 5.0, 2.0],
            [12.0, 7.0, 3.0],
            [15.0, 10.0, 5.0],
        ]),
    }

    # Calibrate model
    calibrated_params = chen_model.calibrate(market_data)

    # Check calibrated parameters
    assert isinstance(calibrated_params, datatypes.ModelParams)
    assert calibrated_params.equity_params.v0 > 0
    assert calibrated_params.equity_params.kappa_v > 0
    assert calibrated_params.equity_params.theta_v > 0
    assert calibrated_params.equity_params.sigma_v > 0


def test_error_handling(chen_model):
    """Test error handling."""
    # Test invalid option type
    with pytest.raises(ValueError):
        chen_model.price_option(
            option_type="invalid",
            strike=100.0,
            maturity=1.0,
        )

    # Test invalid strike
    with pytest.raises(ValueError):
        chen_model.price_option(
            option_type="call",
            strike=-100.0,
            maturity=1.0,
        )

    # Test invalid maturity
    with pytest.raises(ValueError):
        chen_model.price_option(
            option_type="call",
            strike=100.0,
            maturity=-1.0,
        )

    # Test invalid number of paths
    with pytest.raises(ValueError):
        chen_model.simulate(
            num_paths=-100,
            num_steps=252,
        )

    # Test invalid number of steps
    with pytest.raises(ValueError):
        chen_model.simulate(
            num_paths=100,
            num_steps=-252,
        ) 