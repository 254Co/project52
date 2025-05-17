"""
Integration tests for the Chen3 model.
"""

import numpy as np
import pytest

from chen3 import Chen3Model, EquityParams, ModelParams, RateParams
from chen3.correlation import TimeDependentCorrelation
from chen3.payoffs.asian import Asian
from chen3.payoffs.barrier import Barrier
from chen3.payoffs.exotic.callable_puttable import CallablePuttable
from chen3.payoffs.vanilla import Vanilla
from chen3.schemes import euler, milstein, runge_kutta
from chen3.utils.exceptions import NumericalError, ValidationError


def test_full_pricing_pipeline():
    """Test the complete pricing pipeline from model creation to option pricing."""
    # Create model parameters
    rate_params = RateParams(kappa=0.1, theta=0.05, sigma=0.05, r0=0.03)

    equity_params = EquityParams(
        mu=0.05, q=0.02, S0=100.0, v0=0.04, kappa_v=2.0, theta_v=0.04, sigma_v=0.3
    )

    # Create correlation structure
    time_points = np.array([0.0, 1.0, 2.0])
    corr_matrices = [
        np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]]),
        np.array([[1.0, 0.6, 0.4], [0.6, 1.0, 0.3], [0.4, 0.3, 1.0]]),
        np.array([[1.0, 0.7, 0.5], [0.7, 1.0, 0.4], [0.5, 0.4, 1.0]]),
    ]

    correlation = TimeDependentCorrelation(time_points, corr_matrices)

    # Create model
    model_params = ModelParams(
        rate=rate_params, equity=equity_params, correlation=correlation
    )
    model = Chen3Model(model_params)

    # Test different option types
    options = [
        Vanilla(strike=100.0, call=True),  # EuropeanCall
        Vanilla(strike=100.0, call=False),  # EuropeanPut
        CallablePuttable(
            obs_times=[100], call_strike=0.0, put_strike=100.0
        ),  # AmericanPut (approx)
        Asian(strike=100.0, call=True),  # AsianCall
        Barrier(
            strike=100.0, barrier=110.0, knock_in=False, call=True
        ),  # BarrierOption (up-and-out call)
    ]

    # Price all options
    prices = []
    for option in options:
        price = model.price(option, n_paths=10000)
        prices.append(price)

        # Basic validation
        assert price > 0
        assert not np.isnan(price)
        assert not np.isinf(price)

    # Test put-call parity
    call_price = prices[0]
    put_price = prices[1]
    expected_diff = equity_params.S0 * np.exp(-equity_params.q * 1.0) - 100.0 * np.exp(
        -rate_params.r0 * 1.0
    )
    assert abs(call_price - put_price - expected_diff) < 0.1


def test_numerical_methods_integration():
    """Test integration of numerical methods with the model."""
    # Create model parameters
    params = ModelParams(
        rate=RateParams(
            kappa=0.1,
            theta=0.05,
            sigma=0.02,
            r0=0.03
        ),
        equity=EquityParams(
            mu=0.05,      # Drift rate
            q=0.02,       # Dividend yield
            S0=100.0,     # Initial stock price
            v0=0.04,      # Initial variance
            kappa_v=2.0,  # Variance mean reversion
            theta_v=0.04, # Long-term variance
            sigma_v=0.3   # Volatility of variance
        ),
        correlation=TimeDependentCorrelation(
            initial_corr=np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]])
        )
    )

    # Create model
    model = Chen3Model(params)

    # Test different numerical schemes
    schemes = [
        euler.EulerMaruyama(
            mu=lambda t, x: 0.05 * x,
            sigma=lambda t, x: 0.2 * x
        ),
        milstein.Milstein(
            mu=lambda t, x: 0.05 * x,
            sigma=lambda t, x: 0.2 * x,
            sigma_derivative=lambda t, x: 0.2
        )
    ]

    for scheme in schemes:
        # Simulate paths
        paths = scheme.simulate(S0=100.0, dt=0.01, n_steps=100, n_paths=1000)

        # Check paths shape
        assert paths.shape == (1000, 101)

        # Check paths are positive
        assert np.all(paths > 0)

        # Check paths are finite
        assert np.all(np.isfinite(paths))

        # Check paths are not constant
        assert not np.allclose(paths[:, 1:], paths[:, :-1])

        # Check paths have expected properties
        assert np.mean(paths[:, -1]) > 0
        assert np.std(paths[:, -1]) > 0


def test_correlation_integration():
    """Test integration of different correlation structures."""
    # Create base model parameters
    rate_params = RateParams(kappa=0.1, theta=0.05, sigma=0.05, r0=0.03)
    equity_params = EquityParams(
        mu=0.05, q=0.02, S0=100.0, v0=0.04, kappa_v=2.0, theta_v=0.04, sigma_v=0.3
    )

    # Test different correlation structures
    from chen3.correlation import (
        RegimeSwitchingCorrelation,
        StateDependentCorrelation,
        StochasticCorrelation,
        TimeDependentCorrelation,
    )

    # Time-dependent correlation
    time_corr = TimeDependentCorrelation(
        time_points=np.array([0.0, 1.0]), correlation_matrices=[np.eye(3), np.eye(3)]
    )

    # State-dependent correlation
    def corr_func(state):
        r, S, v = state["r"], state["S"], state["v"]
        return np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])

    state_corr = StateDependentCorrelation(
        correlation_function=corr_func,
        default_state={"r": 0.03, "S": 100.0, "v": 0.04}
    )

    # Regime-switching correlation
    correlation_matrices = [
        np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.1], [0.2, 0.1, 1.0]]),
        np.array([[1.0, 0.7, 0.5], [0.7, 1.0, 0.4], [0.5, 0.4, 1.0]])
    ]
    transition_rates = np.array([[-0.2, 0.2], [0.3, -0.3]])
    regime_corr = RegimeSwitchingCorrelation(
        correlation_matrices=correlation_matrices,
        transition_rates=transition_rates,
        initial_regime=0
    )

    # Stochastic correlation
    mean_reversion = np.array([[1.0, 2.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 1.0]])
    long_term_mean = np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]])
    volatility = np.array([[0.0, 0.2, 0.2], [0.2, 0.0, 0.2], [0.2, 0.2, 0.0]])
    initial_corr = np.array([[1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 1.0]])
    stoch_corr = StochasticCorrelation(
        mean_reversion=mean_reversion,
        long_term_mean=long_term_mean,
        volatility=volatility,
        initial_corr=initial_corr,
    )

    # Test each correlation structure
    correlations = [time_corr, state_corr, regime_corr, stoch_corr]
    for correlation in correlations:
        # Create model
        model_params = ModelParams(
            rate=rate_params, equity=equity_params, correlation=correlation
        )
        model = Chen3Model(model_params)

        # Simulate paths
        paths = model.simulate(T=1.0, n_steps=100, n_paths=1000)

        # Basic validation
        assert paths.shape == (1000, 101, 3)
        assert np.all(np.isfinite(paths))
        assert not np.any(np.isnan(paths))
        assert not np.any(np.isinf(paths))


def test_error_handling_integration():
    """Test integration of error handling across components."""
    # Create model with default parameters
    model_params = ModelParams(
        rate=RateParams(kappa=0.1, theta=0.05, sigma=0.05, r0=0.03),
        equity=EquityParams(
            mu=0.05, q=0.02, S0=100.0, v0=0.04, kappa_v=2.0, theta_v=0.04, sigma_v=0.3
        ),
        correlation=TimeDependentCorrelation(
            time_points=np.array([0.0, 1.0]),
            correlation_matrices=[np.eye(3), np.eye(3)],
        ),
    )
    model = Chen3Model(model_params)

    # Test invalid option parameters
    with pytest.raises(ValidationError):
        model.price(Vanilla(strike=-100.0, call=True))

    # Test invalid simulation parameters
    with pytest.raises(ValidationError):
        model.simulate(T=-1.0, n_steps=100, n_paths=1000)

    # Test numerical stability
    with pytest.raises(NumericalError):
        model.simulate(T=100.0, n_steps=100, n_paths=1000)  # Too long maturity

    with pytest.raises(NumericalError):
        model.simulate(T=1.0, n_steps=1, n_paths=1000)  # Too few steps
