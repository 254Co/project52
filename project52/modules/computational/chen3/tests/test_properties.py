"""
Property-based tests for the Chen3 model using hypothesis.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import floats, lists, one_of, sampled_from

from chen3 import Chen3Model, EquityParams, ModelParams, RateParams
from chen3.correlation import (
    TimeDependentCorrelation,
    StateDependentCorrelation,
    RegimeSwitchingCorrelation,
    StochasticCorrelation,
    CopulaCorrelation,
)
from chen3.utils.exceptions import ValidationError


# Define strategies for generating test data
@st.composite
def valid_rate_params(draw):
    """Generate valid interest rate parameters."""
    kappa = draw(st.floats(min_value=0.1, max_value=10.0))
    theta = draw(st.floats(min_value=0.01, max_value=0.2))
    # Ensure Feller condition: 2κθ > σ²
    max_sigma = np.sqrt(2 * kappa * theta * 0.9)  # Use 0.9 to ensure strict inequality
    sigma = draw(st.floats(min_value=0.01, max_value=max_sigma))
    r0 = draw(st.floats(min_value=0.01, max_value=0.2))
    return RateParams(kappa=kappa, theta=theta, sigma=sigma, r0=r0)


@st.composite
def valid_equity_params(draw):
    """Generate valid equity parameters."""
    mu = draw(st.floats(min_value=-0.2, max_value=0.2))
    q = draw(st.floats(min_value=0.0, max_value=0.1))
    S0 = draw(st.floats(min_value=1.0, max_value=1000.0))
    v0 = draw(st.floats(min_value=0.01, max_value=0.5))
    kappa_v = draw(st.floats(min_value=0.1, max_value=10.0))
    theta_v = draw(st.floats(min_value=0.01, max_value=0.5))
    # Ensure Feller condition for variance: 2κ_vθ_v > σ_v²
    max_sigma_v = np.sqrt(
        2 * kappa_v * theta_v * 0.9
    )  # Use 0.9 to ensure strict inequality
    sigma_v = draw(st.floats(min_value=0.1, max_value=max_sigma_v))
    return EquityParams(
        mu=mu, q=q, S0=S0, v0=v0, kappa_v=kappa_v, theta_v=theta_v, sigma_v=sigma_v
    )


@st.composite
def valid_correlation_matrix(draw, size=3):
    """Generate valid correlation matrix."""
    # Generate random correlations
    corr = draw(
        st.builds(
            np.array,
            st.lists(
                st.lists(st.floats(min_value=-0.9, max_value=0.9), min_size=size, max_size=size),
                min_size=size, max_size=size
            )
        )
    )
    # Make symmetric and positive definite
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    while not np.all(np.linalg.eigvals(corr) > 0):
        corr = draw(
            st.builds(
                np.array,
                st.lists(
                    st.lists(st.floats(min_value=-0.9, max_value=0.9), min_size=size, max_size=size),
                    min_size=size, max_size=size
                )
            )
        )
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)
    return corr


# Property-based tests
@given(valid_rate_params())
def test_rate_params_properties(rate_params):
    """Test properties of interest rate parameters."""
    # Test Feller condition
    assert 2 * rate_params.kappa * rate_params.theta > rate_params.sigma**2

    # Test parameter bounds
    assert rate_params.kappa > 0
    assert rate_params.theta > 0
    assert rate_params.sigma > 0
    assert rate_params.r0 > 0


@given(valid_equity_params())
def test_equity_params_properties(equity_params):
    """Test properties of equity parameters."""
    # Test parameter bounds
    assert equity_params.S0 > 0
    assert equity_params.v0 > 0
    assert equity_params.kappa_v > 0
    assert equity_params.theta_v > 0
    assert equity_params.sigma_v > 0
    assert equity_params.q >= 0

    # Test Feller condition for variance process
    assert 2 * equity_params.kappa_v * equity_params.theta_v > equity_params.sigma_v**2

    # Test basic properties
    assert equity_params.total_return == equity_params.mu - equity_params.q
    assert equity_params.feller_condition_satisfied
    assert equity_params.mean_reversion_time == 1.0 / equity_params.kappa_v
    assert equity_params.volatility_scale == equity_params.sigma_v / np.sqrt(2 * equity_params.kappa_v)

    # Test invalid parameters
    with pytest.raises(ValueError):
        equity_params.mu = -0.1  # Negative drift

    with pytest.raises(ValueError):
        equity_params.q = -0.1  # Negative dividend yield

    with pytest.raises(ValueError):
        equity_params.S0 = -100.0  # Negative initial price

    with pytest.raises(ValueError):
        equity_params.v0 = -0.04  # Negative initial variance

    with pytest.raises(ValueError):
        equity_params.kappa_v = -2.0  # Negative mean reversion

    with pytest.raises(ValueError):
        equity_params.theta_v = -0.04  # Negative long-term mean

    with pytest.raises(ValueError):
        equity_params.sigma_v = -0.3  # Negative volatility


@given(
    st.lists(st.floats(min_value=0.0, max_value=10.0), min_size=2, max_size=5),
    st.lists(valid_correlation_matrix(), min_size=2, max_size=5),
)
def test_time_dependent_correlation_properties(time_points, corr_matrices):
    """Test properties of time-dependent correlation."""
    # Ensure time points are sorted
    time_points = sorted(time_points)

    # Create correlation structure
    corr = TimeDependentCorrelation(
        time_points=np.array(time_points), correlation_matrices=corr_matrices
    )

    # Test interpolation at random points
    for t in np.linspace(time_points[0], time_points[-1], 10):
        matrix = corr.get_correlation_matrix(t)

        # Test matrix properties
        assert matrix.shape == (3, 3)
        assert np.all(np.diag(matrix) == 1.0)
        assert np.all(np.abs(matrix) <= 1.0)
        assert np.allclose(matrix, matrix.T)
        assert np.all(np.linalg.eigvals(matrix) > 0)

    # Test basic properties
    assert corr.n_factors == 3
    assert corr.name == "TimeDependentCorrelation"
    assert np.array_equal(corr.time_points, time_points)
    assert len(corr.correlation_matrices) == 2
    assert np.array_equal(corr.correlation_matrices[0], corr_matrices[0])
    assert np.array_equal(corr.correlation_matrices[1], corr_matrices[1])

    # Test correlation matrix at different times
    assert np.array_equal(corr.get_correlation_matrix(0.0), corr_matrices[0])
    assert np.array_equal(corr.get_correlation_matrix(1.0), corr_matrices[1])
    assert np.array_equal(corr.get_correlation_matrix(0.5), (corr_matrices[0] + corr_matrices[1]) / 2)

    # Test invalid time points
    with pytest.raises(ValueError):
        TimeDependentCorrelation(
            time_points=np.array([1.0, 0.0]),  # Non-increasing
            correlation_matrices=corr_matrices,
        )

    # Test invalid correlation matrices
    invalid_corr = np.array([[1.0, 0.5], [0.5, 1.0]])  # Wrong shape
    with pytest.raises(ValueError):
        TimeDependentCorrelation(
            time_points=time_points,
            correlation_matrices=[invalid_corr, corr_matrices[1]],
        )


@given(valid_rate_params(), valid_equity_params(), valid_correlation_matrix())
def test_model_simulation_properties(rate_params, equity_params, corr_matrix):
    """Test properties of model simulation."""
    # Create model parameters
    model_params = ModelParams(
        rate=rate_params,
        equity=equity_params,
        correlation=TimeDependentCorrelation(
            time_points=np.array([0.0, 1.0]),
            correlation_matrices=[corr_matrix, corr_matrix],
        ),
    )

    # Create model
    model = Chen3Model(model_params)

    # Simulate paths
    T = 1.0
    n_steps = 100
    n_paths = 1000
    paths = model.simulate(T, n_steps, n_paths)

    # Test path properties
    assert paths.shape == (n_paths, n_steps + 1, 3)
    assert np.all(np.isfinite(paths))
    assert np.all(paths[:, :, 2] > 0)  # Variance is positive

    # Test initial values
    assert np.allclose(paths[:, 0, 0], rate_params.r0)
    assert np.allclose(paths[:, 0, 1], equity_params.S0)
    assert np.allclose(paths[:, 0, 2], equity_params.v0)

    # Test mean reversion of rates
    rate_mean = np.mean(paths[:, -1, 0])
    assert abs(rate_mean - rate_params.theta) < 0.1

    # Test mean reversion of variance
    var_mean = np.mean(paths[:, -1, 2])
    assert abs(var_mean - equity_params.theta_v) < 0.1


@given(
    st.floats(min_value=0.1, max_value=10.0),  # strike
    st.floats(min_value=0.1, max_value=5.0),  # maturity
    st.integers(min_value=1000, max_value=10000),  # n_paths
    valid_rate_params(),  # rate parameters
)
def test_option_pricing_properties(strike, maturity, n_paths, rate_params):
    """Test properties of option pricing."""
    # Create model with default parameters
    model_params = ModelParams(
        rate=rate_params,
        equity=EquityParams(
            mu=0.05, q=0.02, S0=100.0, v0=0.04, kappa_v=2.0, theta_v=0.04, sigma_v=0.3
        ),
        correlation=TimeDependentCorrelation(
            time_points=np.array([0.0, 1.0]),
            correlation_matrices=[np.eye(3), np.eye(3)],
        ),
    )
    model = Chen3Model(model_params)

    # Test European call
    from chen3.pricing import EuropeanCall

    option = EuropeanCall(strike=strike, maturity=maturity)
    price = model.price(option, n_paths=n_paths)

    # Test price properties
    assert price >= 0
    assert not np.isnan(price)
    assert not np.isinf(price)

    # Test put-call parity
    from chen3.pricing import EuropeanPut

    put = EuropeanPut(strike=strike, maturity=maturity)
    put_price = model.price(put, n_paths=n_paths)

    # Put-call parity with stochastic rates:
    # C - P = S0*exp(-q*T) - K*E[exp(-∫r(t)dt)]
    # For small maturities and mean-reverting rates, we can approximate:
    # E[exp(-∫r(t)dt)] ≈ exp(-r0*T)
    expected_diff = model_params.equity.S0 * np.exp(
        -model_params.equity.q * maturity
    ) - strike * np.exp(-model_params.rate.r0 * maturity)

    # Allow for larger error due to stochastic rates and Monte Carlo noise
    assert abs(price - put_price - expected_diff) < 2.0  # Increased tolerance


@given(
    S0=st.floats(min_value=1.0, max_value=1000.0),
    K=st.floats(min_value=1.0, max_value=1000.0),
    T=st.floats(min_value=0.1, max_value=5.0),
    r=st.floats(min_value=0.01, max_value=0.1),
    q=st.floats(min_value=0.0, max_value=0.1),
    v0=st.floats(min_value=0.01, max_value=0.5),
    kappa_v=st.floats(min_value=0.1, max_value=10.0),
    theta_v=st.floats(min_value=0.01, max_value=0.5),
    sigma_v=st.floats(min_value=0.01, max_value=0.5),
)
def test_option_pricing_properties_hypothesis(
    S0, K, T, r, q, v0, kappa_v, theta_v, sigma_v
):
    """Test properties of option pricing with hypothesis."""
    # Ensure Feller condition is satisfied
    if 2 * kappa_v * theta_v > sigma_v**2:
        # Create model parameters
        rate_params = RateParams(kappa=1.0, theta=r, sigma=0.1, r0=r)
        equity_params = EquityParams(
            mu=r, q=q, S0=S0, v0=v0, kappa_v=kappa_v, theta_v=theta_v, sigma_v=sigma_v
        )
        # Create default correlation matrix
        corr_matrix = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])
        model_params = ModelParams(rate=rate_params, equity=equity_params, correlation=corr_matrix)

        # Test European option pricing
        from chen3.pricing.vanilla import EuropeanCall, EuropeanPut

        call_option = EuropeanCall(strike=K, maturity=T)
        put_option = EuropeanPut(strike=K, maturity=T)
        model = Chen3Model(model_params)

        # Test call option
        call_price = model.price(call_option, n_paths=1000)
        assert call_price > 0
        assert call_price < S0

        # Test put option
        put_price = model.price(put_option, n_paths=1000)
        assert put_price > 0
        assert put_price < K

        # Test put-call parity
        forward_price = S0 * np.exp((r - q) * T)
        assert abs(call_price - put_price - (forward_price - K)) < 1e-10

        # Test American option pricing
        from chen3.pricing.american import AmericanOption

        option = AmericanOption(model_params)

        # Test call option
        call_price = option.price_call(K=K, T=T)
        assert call_price > 0
        assert call_price < S0

        # Test put option
        put_price = option.price_put(K=K, T=T)
        assert put_price > 0
        assert put_price < K

        # Test early exercise premium
        european_put = EuropeanPut(model_params).price_put(K=K, T=T)
        assert put_price >= european_put  # American put should be worth more


@given(st.floats(min_value=2.1, max_value=10.0))
def test_copula_correlation(df):
    """Test copula correlation initialization and properties."""
    # Create base correlation matrix
    base_corr = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])
    
    # Test Gaussian copula
    gaussian_corr = CopulaCorrelation(
        copula_type="gaussian",
        correlation_matrix=base_corr,
        copula_params={}
    )
    assert gaussian_corr.copula_type == "gaussian"
    assert gaussian_corr.n_factors == 3
    
    # Test Student's t copula
    student_corr = CopulaCorrelation(
        copula_type="student",
        correlation_matrix=base_corr,
        copula_params={"df": df}
    )
    assert student_corr.copula_type == "student"
    assert student_corr.copula_params["df"] == df


@given(st.floats(min_value=0.1, max_value=5.0))
def test_stochastic_correlation(kappa):
    """Test stochastic correlation initialization and properties."""
    # Create parameters
    mean_reversion = np.full((3, 3), kappa)
    long_term_mean = np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]])
    volatility = np.array([[0.1, 0.2, 0.2], [0.2, 0.1, 0.2], [0.2, 0.2, 0.1]])
    initial_corr = np.array([[1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 1.0]])
    
    stoch_corr = StochasticCorrelation(
        mean_reversion=mean_reversion,
        long_term_mean=long_term_mean,
        volatility=volatility,
        initial_corr=initial_corr
    )
    
    assert stoch_corr.n_factors == 3
    assert np.all(stoch_corr.mean_reversion > 0)
    assert np.all(stoch_corr.long_term_mean >= 0)
    assert np.all(stoch_corr.volatility >= 0)
    assert np.all(stoch_corr.initial_corr >= 0)
