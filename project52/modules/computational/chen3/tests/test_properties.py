"""
Property-based tests for the Chen3 model using hypothesis.
"""
import pytest
import numpy as np
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
from chen3 import Chen3Model, ModelParams, RateParams, EquityParams
from chen3.correlation import TimeDependentCorrelation
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
    max_sigma_v = np.sqrt(2 * kappa_v * theta_v * 0.9)  # Use 0.9 to ensure strict inequality
    sigma_v = draw(st.floats(min_value=0.1, max_value=max_sigma_v))
    return EquityParams(
        mu=mu, q=q, S0=S0, v0=v0,
        kappa_v=kappa_v, theta_v=theta_v, sigma_v=sigma_v
    )

@st.composite
def valid_correlation_matrix(draw, size=3):
    """Generate valid correlation matrix."""
    # Generate random correlations
    corr = draw(arrays(
        dtype=np.float64,
        shape=(size, size),
        elements=st.floats(min_value=-0.9, max_value=0.9)
    ))
    
    # Make symmetric
    corr = (corr + corr.T) / 2
    
    # Set diagonal to 1
    np.fill_diagonal(corr, 1.0)
    
    # Ensure positive definiteness
    while not np.all(np.linalg.eigvals(corr) > 0):
        corr = draw(arrays(
            dtype=np.float64,
            shape=(size, size),
            elements=st.floats(min_value=-0.9, max_value=0.9)
        ))
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

@given(
    st.lists(st.floats(min_value=0.0, max_value=10.0), min_size=2, max_size=5),
    st.lists(valid_correlation_matrix(), min_size=2, max_size=5)
)
def test_time_dependent_correlation_properties(time_points, corr_matrices):
    """Test properties of time-dependent correlation."""
    # Ensure time points are sorted
    time_points = sorted(time_points)
    
    # Create correlation structure
    corr = TimeDependentCorrelation(
        time_points=np.array(time_points),
        correlation_matrices=corr_matrices
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

@given(
    valid_rate_params(),
    valid_equity_params(),
    valid_correlation_matrix()
)
def test_model_simulation_properties(rate_params, equity_params, corr_matrix):
    """Test properties of model simulation."""
    # Create model parameters
    model_params = ModelParams(
        rate=rate_params,
        equity=equity_params,
        correlation=TimeDependentCorrelation(
            time_points=np.array([0.0, 1.0]),
            correlation_matrices=[corr_matrix, corr_matrix]
        )
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
    st.floats(min_value=0.1, max_value=5.0),   # maturity
    st.integers(min_value=1000, max_value=10000),  # n_paths
    valid_rate_params()  # rate parameters
)
def test_option_pricing_properties(strike, maturity, n_paths, rate_params):
    """Test properties of option pricing."""
    # Create model with default parameters
    model_params = ModelParams(
        rate=rate_params,
        equity=EquityParams(
            mu=0.05, q=0.02, S0=100.0,
            v0=0.04, kappa_v=2.0, theta_v=0.04, sigma_v=0.3
        ),
        correlation=TimeDependentCorrelation(
            time_points=np.array([0.0, 1.0]),
            correlation_matrices=[np.eye(3), np.eye(3)]
        )
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
    expected_diff = (
        model_params.equity.S0 * np.exp(-model_params.equity.q * maturity) -
        strike * np.exp(-model_params.rate.r0 * maturity)
    )
    
    # Allow for larger error due to stochastic rates
    assert abs(price - put_price - expected_diff) < 1.0 