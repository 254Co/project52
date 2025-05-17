"""
Shared test fixtures for the Chen3 test suite.
"""

import numpy as np
import pytest

from chen3 import EquityParams, ModelParams, RateParams
from chen3.correlation import TimeDependentCorrelation


@pytest.fixture
def rate_params():
    """Fixture for valid interest rate parameters."""
    return RateParams(kappa=0.1, theta=0.05, sigma=0.1, r0=0.03)


@pytest.fixture
def equity_params():
    """Fixture for valid equity parameters."""
    return EquityParams(
        mu=0.05, q=0.02, S0=100.0, v0=0.04, kappa_v=2.0, theta_v=0.04, sigma_v=0.3
    )


@pytest.fixture
def time_correlation():
    """Fixture for time-dependent correlation structure."""
    time_points = np.array([0.0, 1.0, 2.0])
    corr_matrices = [
        np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]]),
        np.array([[1.0, 0.6, 0.4], [0.6, 1.0, 0.3], [0.4, 0.3, 1.0]]),
        np.array([[1.0, 0.7, 0.5], [0.7, 1.0, 0.4], [0.5, 0.4, 1.0]]),
    ]
    return TimeDependentCorrelation(
        time_points=time_points, correlation_matrices=corr_matrices
    )


@pytest.fixture
def model_params(rate_params, equity_params, time_correlation):
    """Fixture for complete model parameters."""
    return ModelParams(
        rate=rate_params, equity=equity_params, correlation=time_correlation
    )


@pytest.fixture
def european_call_params():
    """Fixture for European call option parameters."""
    return {"S0": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2}


@pytest.fixture
def european_call_bs_price(european_call_params):
    """Fixture for European call Black-Scholes price."""
    S0 = european_call_params["S0"]
    K = european_call_params["K"]
    T = european_call_params["T"]
    r = european_call_params["r"]
    sigma = european_call_params["sigma"]

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * np.exp(-r * T) * 0.5 * (1 + np.erf(d1 / np.sqrt(2))) - K * np.exp(
        -r * T
    ) * 0.5 * (1 + np.erf(d2 / np.sqrt(2)))
