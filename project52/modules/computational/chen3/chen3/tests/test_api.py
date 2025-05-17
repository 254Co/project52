"""
Test suite for the Chen3 API module.

This module contains tests for the user-friendly API functions, including
model creation, option pricing, and parameter validation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from .. import api
from ..utils.exceptions import InputError, ValidationError, ChenError, OptionError


def test_create_model():
    """Test model creation through API."""
    # Create model with default parameters
    model = api.create_model()
    assert model is not None
    assert model.num_factors == 3
    assert model.params.rate.kappa == 0.2
    assert model.params.rate.theta == 0.05
    assert model.params.rate.sigma == 0.08
    assert model.params.rate.r0 == 0.03
    assert model.params.equity.mu == 0.05
    assert model.params.equity.q == 0.02
    assert model.params.equity.S0 == 100.0
    assert model.params.equity.v0 == 0.04
    assert model.params.equity.kappa_v == 2.0
    assert model.params.equity.theta_v == 0.04
    assert model.params.equity.sigma_v == 0.2

    # Test with custom rate parameters (Feller-compliant)
    custom_rate = {"kappa": 0.2, "theta": 0.05, "sigma": 0.08, "r0": 0.03}
    model2 = api.create_model(rate_params=custom_rate)
    assert model2.params.rate.kappa == 0.2
    assert model2.params.rate.theta == 0.05
    assert model2.params.rate.sigma == 0.08
    assert model2.params.rate.r0 == 0.03

    # Test with custom equity parameters
    equity_params = {
        "mu": 0.05,
        "q": 0.02,
        "S0": 100.0,
        "v0": 0.04,
        "kappa_v": 2.0,
        "theta_v": 0.04,
        "sigma_v": 0.3,
    }
    model = api.create_model(equity_params=equity_params)
    assert model.params.equity.mu == equity_params["mu"]
    assert model.params.equity.q == equity_params["q"]
    assert model.params.equity.S0 == equity_params["S0"]
    assert model.params.equity.v0 == equity_params["v0"]
    assert model.params.equity.kappa_v == equity_params["kappa_v"]
    assert model.params.equity.theta_v == equity_params["theta_v"]
    assert model.params.equity.sigma_v == equity_params["sigma_v"]

    # Test with custom correlation
    correlation_params = {
        "rho_rs": 0.1,
        "rho_rv": -0.1,
        "rho_sv": -0.5,
    }
    model = api.create_model(correlation_params=correlation_params)
    assert np.allclose(model.params.correlation[0, 1], correlation_params["rho_rs"])
    assert np.allclose(model.params.correlation[0, 2], correlation_params["rho_rv"])
    assert np.allclose(model.params.correlation[1, 2], correlation_params["rho_sv"])

    # Test with invalid rate parameters
    with pytest.raises(ChenError):
        api.create_model(rate_params={"kappa": -0.1})

    # Test with invalid equity parameters
    with pytest.raises(ChenError):
        api.create_model(equity_params={"S0": -100.0})

    # Test with invalid correlation type (should raise InputError)
    with pytest.raises(InputError):
        api.create_model(correlation_params={"rho_rs": 2.0, "rho_rv": 0.0, "rho_sv": 0.0})


def test_price_option():
    """Test option pricing through API."""
    # Create model with more stable parameters
    model = api.create_model(
        rate_params={"kappa": 0.3, "theta": 0.05, "sigma": 0.05, "r0": 0.03},
        equity_params={"mu": 0.05, "q": 0.02, "S0": 100.0, "v0": 0.04, "kappa_v": 3.0, "theta_v": 0.04, "sigma_v": 0.1},
    )

    # Test European call option
    strike = 100.0
    maturity = 1.0
    price = api.price_option(
        model=model,
        option_type="call",
        strike=strike,
        maturity=maturity,
        n_paths=1000,
        n_steps=50,
    )
    assert price["price"] > 0

    # Test European put option
    price = api.price_option(
        model=model,
        option_type="put",
        strike=strike,
        maturity=maturity,
        n_paths=1000,
        n_steps=50,
    )
    assert price["price"] > 0

    # Test put-call parity
    call_price = api.price_option(
        model=model,
        option_type="call",
        strike=strike,
        maturity=maturity,
        n_paths=50000,
        n_steps=200,
    )
    put_price = api.price_option(
        model=model,
        option_type="put",
        strike=strike,
        maturity=maturity,
        n_paths=50000,
        n_steps=200,
    )
    forward_price = model.params.equity.S0 * np.exp(
        -model.params.equity.q * maturity
    )
    discount_factor = np.exp(-model.params.rate.r0 * maturity)
    assert_allclose(
        call_price["price"] - put_price["price"],
        forward_price - strike * discount_factor,
        rtol=0.3,
    )

    # Test with custom simulation parameters
    price = api.price_option(
        model=model,
        option_type="call",
        strike=strike,
        maturity=maturity,
        n_paths=1000,
        n_steps=50,
    )
    assert price["price"] > 0

    # Test with invalid option type
    with pytest.raises(InputError):
        api.price_option(
            model=model,
            option_type="invalid",
            strike=strike,
            maturity=maturity,
        )

    # Test with invalid strike
    with pytest.raises(InputError):
        api.price_option(
            model=model,
            option_type="call",
            strike=-100.0,
            maturity=maturity,
        )

    # Test with invalid maturity
    with pytest.raises(InputError):
        api.price_option(
            model=model,
            option_type="call",
            strike=strike,
            maturity=-1.0,
        )

    # Test with invalid number of paths
    with pytest.raises(InputError):
        api.price_option(
            model=model,
            option_type="call",
            strike=strike,
            maturity=maturity,
            n_paths=-10000,
        )

    # Test with invalid number of steps
    with pytest.raises(InputError):
        api.price_option(
            model=model,
            option_type="call",
            strike=strike,
            maturity=maturity,
            n_steps=-252,
        )


def test_barrier_options():
    """Test barrier option pricing through API."""
    # Create model with more stable parameters
    model = api.create_model(
        rate_params={"kappa": 0.3, "theta": 0.05, "sigma": 0.05, "r0": 0.03},
        equity_params={"mu": 0.05, "q": 0.02, "S0": 100.0, "v0": 0.04, "kappa_v": 3.0, "theta_v": 0.04, "sigma_v": 0.1},
    )

    # Test up-and-out call option
    strike = 100.0
    maturity = 1.0
    barrier = 120.0
    price = api.price_option(
        model=model,
        option_type="up_and_out_call",
        strike=strike,
        maturity=maturity,
        barrier=barrier,
        n_paths=1000,
        n_steps=50,
    )
    assert price["price"] > 0

    # Test down-and-out put option
    price = api.price_option(
        model=model,
        option_type="down_and_out_put",
        strike=strike,
        maturity=maturity,
        barrier=80.0,
        n_paths=1000,
        n_steps=50,
    )
    assert price["price"] > 0

    # Test with invalid barrier level
    with pytest.raises(OptionError):
        api.price_option(
            model=model,
            option_type="up_and_out_call",
            strike=strike,
            maturity=maturity,
            barrier=strike,
        )

    # Test with missing barrier level
    with pytest.raises(InputError):
        api.price_option(
            model=model,
            option_type="up_and_out_call",
            strike=strike,
            maturity=maturity,
        )


def test_american_options():
    """Test American option pricing through API."""
    # Create model
    model = api.create_model()

    # Test American call option
    strike = 100.0
    maturity = 1.0
    price = api.price_option(
        model=model,
        option_type="call",
        strike=strike,
        maturity=maturity,
        american=True,
    )
    assert price["price"] > 0

    # Test American put option
    price = api.price_option(
        model=model,
        option_type="put",
        strike=strike,
        maturity=maturity,
        american=True,
    )
    assert price["price"] > 0

    # Test with invalid exercise frequency
    with pytest.raises(InputError):
        api.price_option(
            model=model,
            option_type="call",
            strike=strike,
            maturity=maturity,
            american=True,
            exercise_frequency=-1,
        )


@pytest.mark.skip(reason="Risk metrics calculation not implemented yet")
def test_risk_metrics():
    """Test risk metrics calculation through API."""
    # Create model
    model = api.create_model()

    # Calculate risk metrics
    metrics = api.price_option(
        model=model,
        option_type="call",
        strike=100.0,
        maturity=1.0,
        calculate_metrics=True,
    )

    # Check metrics
    assert "price" in metrics
    assert "delta" in metrics
    assert "gamma" in metrics
    assert "vega" in metrics
    assert "theta" in metrics
    assert "rho" in metrics

    # Check delta is between 0 and 1 for call option
    assert 0 <= metrics["delta"] <= 1

    # Check gamma is positive
    assert metrics["gamma"] > 0


@pytest.mark.skip(reason="Model calibration not implemented yet")
def test_calibration():
    """Test model calibration through API."""
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

    # Create model
    model = api.create_model()

    # Calibrate model
    calibrated_model = api.price_option(
        model=model,
        option_type="call",
        strike=100.0,
        maturity=1.0,
        calibrate=True,
        market_data=market_data,
    )

    # Check calibrated parameters
    assert calibrated_model.params.equity.v0 > 0
    assert calibrated_model.params.equity.kappa_v > 0
    assert calibrated_model.params.equity.theta_v > 0
    assert calibrated_model.params.equity.sigma_v > 0

    # Test with invalid market data
    with pytest.raises(InputError):
        api.price_option(
            model=model,
            option_type="call",
            strike=100.0,
            maturity=1.0,
            calibrate=True,
            market_data={},
        ) 