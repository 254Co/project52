"""
Tests for numerical methods and discretization schemes.
"""

import time

import numpy as np
import pytest
from hypothesis import given, strategies as st
from numba import njit, float64

from chen3 import Chen3Model
from chen3.numerical_engines import monte_carlo
from chen3.payoffs.vanilla import Vanilla
from chen3.schemes import euler, milstein, runge_kutta
from chen3.utils.exceptions import NumericalError
from chen3.numerical_engines.monte_carlo import MonteCarloEngine
from chen3.numerical_engines.quasi_mc import QuasiMonteCarloEngine
from chen3.numerical_engines.integration import IntegrationScheme
from chen3.schemes.euler import euler_scheme
from chen3.schemes.milstein import milstein_scheme
from chen3.schemes.adaptive import AdaptiveTimeStepper
from chen3.datatypes import RateParams, EquityParams, ModelParams
from chen3.numerical_engines.variance_reduction import AntitheticVariates, ControlVariates
from chen3.correlation.models.time_dependent import TimeDependentCorrelation

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def create_test_model_params():
    """Create test model parameters with a valid correlation matrix."""
    rate_params = RateParams(kappa=1.0, theta=0.05, sigma=0.1, r0=0.03)
    equity_params = EquityParams(
        mu=0.05, q=0.02, S0=100.0, v0=0.04, kappa_v=2.0, theta_v=0.04, sigma_v=0.3
    )
    # Create a simple time-dependent correlation
    correlation = TimeDependentCorrelation(
        time_points=np.array([0.0, 1.0]),
        correlation_matrices=[np.eye(3), np.eye(3)]  # Identity matrix for no correlation
    )
    return ModelParams(
        rate=rate_params,
        equity=equity_params,
        correlation=correlation
    )


def test_monte_carlo_convergence():
    """Test Monte Carlo convergence."""
    model_params = create_test_model_params()
    n_steps = 100
    T = 1.0
    n_paths_list = [1000, 10000, 100000]
    errors = []

    for n_paths in n_paths_list:
        engine = MonteCarloEngine(
            n_paths=n_paths,
            n_steps=n_steps,
            dt=T/n_steps,
            use_antithetic=True,
            use_control_variate=True
        )
        paths = engine.simulate_paths(
            initial_state={
                "r": model_params.rate.r0,
                "S": model_params.equity.S0,
                "v": model_params.equity.v0
            },
            drift_function=lambda state, t: (
                model_params.rate.kappa * (model_params.rate.theta - state["r"]),
                (model_params.equity.mu - model_params.equity.q) * state["S"],
                model_params.equity.kappa_v * (model_params.equity.theta_v - state["v"])
            ),
            diffusion_function=lambda state, t: (
                model_params.rate.sigma * np.sqrt(state["r"]),
                np.sqrt(state["v"]) * state["S"],
                model_params.equity.sigma_v * np.sqrt(state["v"])
            ),
            correlation_matrix=model_params.correlation.get_correlation_matrix(0.0)
        )
        final_prices = paths[1][:, -1]  # Stock prices at final time
        expected_value = model_params.equity.S0 * np.exp(
            (model_params.equity.mu - model_params.equity.q) * 1.0
        )
        error = np.abs(np.mean(final_prices) - expected_value)
        errors.append(error)
    for i in range(len(errors) - 1):
        assert errors[i] > errors[i + 1]


@pytest.mark.skipif(not HAS_CUPY, reason="cupy is not available")
def test_gpu_monte_carlo(model_params, european_call_params):
    """Test GPU-accelerated Monte Carlo simulation."""
    from chen3.numerical_engines import gpu_mc_engine

    model = Chen3Model(model_params)
    option = Vanilla(strike=european_call_params["K"], call=True)

    # Test GPU pricing
    price = model.price(
        option, n_paths=10000, engine=gpu_mc_engine.GPUMonteCarloEngine()
    )

    # Basic validation
    assert price > 0
    assert not np.isnan(price)
    assert not np.isinf(price)


def test_quasi_monte_carlo():
    """Test Quasi-Monte Carlo methods."""
    model_params = create_test_model_params()
    n_steps = 100
    T = 1.0
    n_paths_list = [1000, 10000]
    errors = []
    for n_paths in n_paths_list:
        engine = QuasiMonteCarloEngine(
            n_paths=n_paths,
            n_steps=n_steps,
            dt=T/n_steps,
            use_antithetic=True,
            use_control_variate=True
        )
        paths = engine.simulate_paths(
            initial_state={
                "r": model_params.rate.r0,
                "S": model_params.equity.S0,
                "v": model_params.equity.v0
            },
            drift_function=lambda state, t: (
                model_params.rate.kappa * (model_params.rate.theta - state["r"]),
                (model_params.equity.mu - model_params.equity.q) * state["S"],
                model_params.equity.kappa_v * (model_params.equity.theta_v - state["v"])
            ),
            diffusion_function=lambda state, t: (
                model_params.rate.sigma * np.sqrt(state["r"]),
                np.sqrt(state["v"]) * state["S"],
                model_params.equity.sigma_v * np.sqrt(state["v"])
            ),
            correlation_matrix=model_params.correlation.get_correlation_matrix(0.0)
        )
        final_prices = paths[1][:, -1]  # Stock prices at final time
        expected_value = model_params.equity.S0 * np.exp(
            (model_params.equity.mu - model_params.equity.q) * 1.0
        )
        error = np.abs(np.mean(final_prices) - expected_value)
        errors.append(error)
    for i in range(len(errors) - 1):
        assert errors[i] > errors[i + 1]


def test_euler_maruyama_convergence():
    """Test Euler-Maruyama scheme convergence."""
    model_params = create_test_model_params()

    # Extract parameters for Numba functions
    mu = model_params.equity.mu
    q = model_params.equity.q
    v0 = model_params.equity.v0
    S0 = model_params.equity.S0

    # Test convergence with decreasing time step
    dt_list = [0.1, 0.05, 0.025]
    errors = []

    for dt in dt_list:
        # Simulate paths
        n_steps = int(1.0 / dt)
        n_paths = 50000  # Further increased number of paths for stability
        
        # Create drift and diffusion functions
        @njit
        def drift_fn(x: float64) -> float64:
            return (mu - q) * x

        @njit
        def diffusion_fn(x: float64) -> float64:
            return np.sqrt(v0) * x

        # Create step function
        step_fn = euler_scheme(drift_fn, diffusion_fn, dt)
        
        # Simulate multiple paths
        final_values = np.zeros(n_paths)
        rng = np.random.default_rng()
        
        for path in range(n_paths):
            state = S0
            for i in range(n_steps):
                z = rng.normal(0, 1)
                state = step_fn(state, z)
                state = max(state, 0)  # Enforce non-negativity
            final_values[path] = state

        # Compute error
        expected_value = S0 * np.exp((mu - q) * 1.0)
        error = np.abs(np.mean(final_values) - expected_value)
        errors.append(error)

    print(f"Euler-Maruyama errors for dt_list {dt_list}: {errors}")

    # Check that error decreases with decreasing dt, allowing for small fluctuations
    for i in range(len(errors) - 1):
        assert errors[i] > errors[i + 1] or np.isclose(errors[i], errors[i + 1], rtol=0.2)


def test_milstein_convergence():
    """Test Milstein scheme convergence."""
    model_params = create_test_model_params()

    # Test convergence with decreasing time step
    dt_list = [0.1, 0.05, 0.025]
    errors = []

    for dt in dt_list:
        # Simulate paths
        n_steps = int(1.0 / dt)
        result = milstein_scheme(
            x0=model_params.equity.S0,
            t0=0.0,
            T=1.0,
            n_steps=n_steps,
            drift=lambda t, x: (model_params.equity.mu - model_params.equity.q) * x,
            diffusion=lambda t, x: np.sqrt(model_params.equity.v0) * x,
            diffusion_derivative=lambda t, x: np.sqrt(model_params.equity.v0),
            rng=np.random.default_rng()
        )
        final_prices = result.paths[-1]  # Stock prices at final time

        # Compute error
        expected_value = model_params.equity.S0 * np.exp(
            (model_params.equity.mu - model_params.equity.q) * 1.0
        )
        error = np.abs(np.mean(final_prices) - expected_value)
        errors.append(error)

    # Check that error decreases with decreasing dt
    for i in range(len(errors) - 1):
        assert errors[i] > errors[i + 1]


def test_runge_kutta_stability():
    """Test stability of Runge-Kutta scheme."""
    # Test with stiff system
    kappa = 10.0  # High mean reversion
    theta = 0.05
    sigma = 0.5
    r0 = 0.03
    T = 1.0

    # Test with different step sizes
    steps = [10, 100, 1000]
    results = []

    for n_steps in steps:
        dt = T / n_steps
        result = runge_kutta.rk4_scheme(
            x0=r0,
            t0=0.0,
            T=T,
            n_steps=n_steps,
            drift=lambda t, x: kappa * (theta - x),
            diffusion=lambda t, x: sigma * np.sqrt(x),
            rng=np.random.default_rng()
        )
        mean = np.mean(result.paths[-1])
        results.append(abs(mean - theta))

    # Check convergence - allow for some numerical noise
    for i in range(len(results) - 1):
        assert results[i + 1] <= results[i] * 1.1  # Allow 10% tolerance


def test_adaptive_time_stepping():
    """Test adaptive time stepping."""
    model_params = create_test_model_params()

    # Create adaptive time stepper
    stepper = AdaptiveTimeStepper(
        min_steps=10,
        max_steps=1000,
        min_dt=1e-4,
        max_dt=0.1
    )

    # Test with different tolerances
    tol_list = [1e-2, 1e-3, 1e-4]
    errors = []

    for tol in tol_list:
        # Simulate paths
        times, states = stepper.integrate(
            t0=0.0,
            t1=1.0,
            state0=np.array([model_params.equity.S0]),
            step_function=lambda t, x, dt: (
                x + (model_params.equity.mu - model_params.equity.q) * x * dt +
                np.sqrt(model_params.equity.v0) * x * np.random.normal(0, np.sqrt(dt))
            )
        )
        final_prices = states[-1]  # Stock prices at final time

        # Compute error
        expected_value = model_params.equity.S0 * np.exp(
            (model_params.equity.mu - model_params.equity.q) * 1.0
        )
        error = np.abs(np.mean(final_prices) - expected_value)
        errors.append(error)

    # Check that error decreases with decreasing tolerance
    for i in range(len(errors) - 1):
        assert errors[i] > errors[i + 1]


def test_numerical_error_handling():
    """Test numerical error handling."""
    # Test with invalid parameters
    with pytest.raises(NumericalError):
        euler_scheme(
            drift_fn=lambda x: 0.05 * x,
            diffusion_fn=lambda x: -0.2 * x,  # Negative volatility
            dt=0.01
        )

    # Test with zero volatility
    with pytest.raises(NumericalError):
        euler_scheme(
            drift_fn=lambda x: 0.05 * x,
            diffusion_fn=lambda x: 0.0 * x,  # Zero volatility
            dt=0.01
        )

    # Test with invalid time step
    with pytest.raises(NumericalError):
        euler_scheme(
            drift_fn=lambda x: 0.05 * x,
            diffusion_fn=lambda x: 0.2 * x,
            dt=0.0  # Invalid time step
        )


def test_parallel_computation():
    """Test parallel computation capabilities."""
    # Test with large number of paths
    mu = 0.05
    sigma = 0.2
    S0 = 100.0
    T = 1.0
    n_steps = 100
    n_paths = 100000

    # Test serial computation
    result_serial = euler_scheme(
        drift_fn=lambda x: mu * x,
        diffusion_fn=lambda x: sigma * x,
        dt=T/n_steps
    )
    paths_serial = np.zeros((n_paths, n_steps + 1))
    paths_serial[:, 0] = S0
    rng = np.random.default_rng()
    for i in range(n_steps):
        z = rng.normal(0, 1, size=n_paths)
        paths_serial[:, i + 1] = result_serial(paths_serial[:, i], z)

    # Test parallel computation
    result_parallel = euler_scheme(
        drift_fn=lambda x: mu * x,
        diffusion_fn=lambda x: sigma * x,
        dt=T/n_steps
    )
    paths_parallel = np.zeros((n_paths, n_steps + 1))
    paths_parallel[:, 0] = S0
    rng = np.random.default_rng()
    for i in range(n_steps):
        z = rng.normal(0, 1, size=n_paths)
        paths_parallel[:, i + 1] = result_parallel(paths_parallel[:, i], z)

    # Compare results
    assert np.allclose(paths_serial, paths_parallel)


def test_variance_reduction_properties():
    """Test properties of variance reduction techniques."""
    model_params = create_test_model_params()

    n_paths = 10000
    n_steps = 100
    T = 1.0

    # Test antithetic variates
    engine_no_av = MonteCarloEngine(
        n_paths=n_paths,
        n_steps=n_steps,
        dt=T/n_steps,
        variance_reduction=None,
        use_antithetic=False
    )
    engine_av = MonteCarloEngine(
        n_paths=n_paths,
        n_steps=n_steps,
        dt=T/n_steps,
        variance_reduction=AntitheticVariates(),
        use_antithetic=True
    )

    # Simulate paths and compute payoffs
    paths_no_av = engine_no_av.simulate_paths(
        initial_state={
            "r": model_params.rate.r0,
            "S": model_params.equity.S0,
            "v": model_params.equity.v0
        },
        drift_function=lambda state, t: (
            model_params.rate.kappa * (model_params.rate.theta - state["r"]),
            (model_params.equity.mu - model_params.equity.q) * state["S"],
            model_params.equity.kappa_v * (model_params.equity.theta_v - state["v"])
        ),
        diffusion_function=lambda state, t: (
            model_params.rate.sigma * np.sqrt(state["r"]),
            np.sqrt(state["v"]) * state["S"],
            model_params.equity.sigma_v * np.sqrt(state["v"])
        ),
        correlation_matrix=model_params.correlation.get_correlation_matrix(0.0)
    )
    paths_av = engine_av.simulate_paths(
        initial_state={
            "r": model_params.rate.r0,
            "S": model_params.equity.S0,
            "v": model_params.equity.v0
        },
        drift_function=lambda state, t: (
            model_params.rate.kappa * (model_params.rate.theta - state["r"]),
            (model_params.equity.mu - model_params.equity.q) * state["S"],
            model_params.equity.kappa_v * (model_params.equity.theta_v - state["v"])
        ),
        diffusion_function=lambda state, t: (
            model_params.rate.sigma * np.sqrt(state["r"]),
            np.sqrt(state["v"]) * state["S"],
            model_params.equity.sigma_v * np.sqrt(state["v"])
        ),
        correlation_matrix=model_params.correlation.get_correlation_matrix(0.0)
    )
    var_no_av = np.var(paths_no_av[1][:, -1])
    var_av = np.var(paths_av[1][:, -1])
    assert var_av < var_no_av

    # Test control variates
    engine_no_cv = MonteCarloEngine(
        n_paths=n_paths,
        n_steps=n_steps,
        dt=T/n_steps,
        variance_reduction=None
    )
    engine_cv = MonteCarloEngine(
        n_paths=n_paths,
        n_steps=n_steps,
        dt=T/n_steps,
        variance_reduction=ControlVariates()
    )

    # Simulate paths and compute payoffs
    paths_no_cv = engine_no_cv.simulate_paths(
        initial_state={
            "r": model_params.rate.r0,
            "S": model_params.equity.S0,
            "v": model_params.equity.v0
        },
        drift_function=lambda state, t: (
            model_params.rate.kappa * (model_params.rate.theta - state["r"]),
            (model_params.equity.mu - model_params.equity.q) * state["S"],
            model_params.equity.kappa_v * (model_params.equity.theta_v - state["v"])
        ),
        diffusion_function=lambda state, t: (
            model_params.rate.sigma * np.sqrt(state["r"]),
            np.sqrt(state["v"]) * state["S"],
            model_params.equity.sigma_v * np.sqrt(state["v"])
        ),
        correlation_matrix=model_params.correlation.get_correlation_matrix(0.0)
    )
    paths_cv = engine_cv.simulate_paths(
        initial_state={
            "r": model_params.rate.r0,
            "S": model_params.equity.S0,
            "v": model_params.equity.v0
        },
        drift_function=lambda state, t: (
            model_params.rate.kappa * (model_params.rate.theta - state["r"]),
            (model_params.equity.mu - model_params.equity.q) * state["S"],
            model_params.equity.kappa_v * (model_params.equity.theta_v - state["v"])
        ),
        diffusion_function=lambda state, t: (
            model_params.rate.sigma * np.sqrt(state["r"]),
            np.sqrt(state["v"]) * state["S"],
            model_params.equity.sigma_v * np.sqrt(state["v"])
        ),
        correlation_matrix=model_params.correlation.get_correlation_matrix(0.0)
    )
    var_no_cv = np.var(paths_no_cv[1][:, -1])
    var_cv = np.var(paths_cv[1][:, -1])
    assert var_cv < var_no_cv


def test_integration_scheme():
    """Test the IntegrationScheme class."""
    # Test initialization with different schemes
    schemes = ["euler", "milstein", "predictor_corrector"]
    for scheme_type in schemes:
        scheme = IntegrationScheme(scheme_type=scheme_type)
        assert scheme.scheme_type == scheme_type
        assert scheme.order > 0
        assert isinstance(scheme.use_implicit, bool)
        assert 0 <= scheme.theta <= 1

    # Test invalid scheme type
    with pytest.raises(NumericalError):
        IntegrationScheme(scheme_type="invalid_scheme")

    # Test invalid order
    with pytest.raises(NumericalError):
        IntegrationScheme(order=0)

    # Test invalid theta
    with pytest.raises(NumericalError):
        IntegrationScheme(use_implicit=True, theta=1.5)

    # Test integration step
    scheme = IntegrationScheme(scheme_type="euler")
    state = {
        "r": np.array([0.03]),
        "S": np.array([100.0]),
        "v": np.array([0.04])
    }
    drift = (
        np.array([0.1 * (0.05 - state["r"])]),
        np.array([state["r"] * state["S"]]),
        np.array([0.2 * (0.04 - state["v"])])
    )
    diffusion = (
        np.array([0.05 * np.sqrt(state["r"])]),
        np.array([np.sqrt(state["v"]) * state["S"]]),
        np.array([0.3 * np.sqrt(state["v"])])
    )
    dW = np.random.normal(0, 1, size=(1, 3))
    dt = 0.01

    # Test Euler step
    new_r, new_S, new_v = scheme.step(state, drift, diffusion, dW, dt)
    assert isinstance(new_r, np.ndarray)
    assert isinstance(new_S, np.ndarray)
    assert isinstance(new_v, np.ndarray)
    assert new_r.shape == (1,)
    assert new_S.shape == (1,)
    assert new_v.shape == (1,)
    assert np.all(np.isfinite(new_r))
    assert np.all(np.isfinite(new_S))
    assert np.all(np.isfinite(new_v))
    assert np.all(new_r >= 0)
    assert np.all(new_S >= 0)
    assert np.all(new_v >= 0)

    # Test Milstein step
    scheme = IntegrationScheme(scheme_type="milstein")
    new_r, new_S, new_v = scheme.step(state, drift, diffusion, dW, dt)
    assert isinstance(new_r, np.ndarray)
    assert isinstance(new_S, np.ndarray)
    assert isinstance(new_v, np.ndarray)
    assert new_r.shape == (1,)
    assert new_S.shape == (1,)
    assert new_v.shape == (1,)
    assert np.all(np.isfinite(new_r))
    assert np.all(np.isfinite(new_S))
    assert np.all(np.isfinite(new_v))
    assert np.all(new_r >= 0)
    assert np.all(new_S >= 0)
    assert np.all(new_v >= 0)

    # Test predictor-corrector step
    scheme = IntegrationScheme(scheme_type="predictor_corrector")
    new_r, new_S, new_v = scheme.step(state, drift, diffusion, dW, dt)
    assert isinstance(new_r, np.ndarray)
    assert isinstance(new_S, np.ndarray)
    assert isinstance(new_v, np.ndarray)
    assert new_r.shape == (1,)
    assert new_S.shape == (1,)
    assert new_v.shape == (1,)
    assert np.all(np.isfinite(new_r))
    assert np.all(np.isfinite(new_S))
    assert np.all(np.isfinite(new_v))
    assert np.all(new_r >= 0)
    assert np.all(new_S >= 0)
    assert np.all(new_v >= 0)
