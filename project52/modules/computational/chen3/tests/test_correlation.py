"""
Tests for correlation structures and their properties.
"""
import pytest
import numpy as np
from chen3.correlation import (
    TimeDependentCorrelation,
    StateDependentCorrelation,
    RegimeSwitchingCorrelation,
    StochasticCorrelation,
    CopulaCorrelation
)
from chen3.utils.exceptions import ValidationError

def test_time_dependent_correlation():
    """Test time-dependent correlation."""
    # Create valid correlation matrices
    corr1 = np.array([[1.0, 0.5, 0.3],
                      [0.5, 1.0, 0.2],
                      [0.3, 0.2, 1.0]])
    corr2 = np.array([[1.0, 0.4, 0.2],
                      [0.4, 1.0, 0.3],
                      [0.2, 0.3, 1.0]])
    
    # Test with strictly increasing time points
    time_points = np.array([0.0, 1.0])
    corr = TimeDependentCorrelation(
        time_points=time_points,
        correlation_matrices=[corr1, corr2]
    )
    assert corr is not None
    
    # Test with non-increasing time points
    with pytest.raises(ValidationError):
        TimeDependentCorrelation(
            time_points=np.array([1.0, 0.0]),
            correlation_matrices=[corr1, corr2]
        )
    
    # Test with invalid correlation matrix
    with pytest.raises(ValidationError):
        TimeDependentCorrelation(
            time_points=np.array([0.0, 1.0]),
            correlation_matrices=[
                np.array([[1.0, 1.5],  # Invalid correlation > 1
                         [1.5, 1.0]])
            ]
        )
    
    # Test interpolation
    t = 0.5
    corr_matrix = corr.get_correlation_matrix(t)
    assert corr_matrix.shape == (3, 3)
    assert np.all(np.diag(corr_matrix) == 1.0)
    assert np.all(np.abs(corr_matrix) <= 1.0)
    
    # Test symmetry
    assert np.allclose(corr_matrix, corr_matrix.T)
    
    # Test positive definiteness
    eigenvalues = np.linalg.eigvals(corr_matrix)
    assert np.all(eigenvalues > 0)
    
    # Test boundary conditions
    assert np.allclose(corr.get_correlation_matrix(0.0), corr1)
    assert np.allclose(corr.get_correlation_matrix(1.0), corr2)
    
    # Test extrapolation
    with pytest.raises(ValidationError):
        corr.get_correlation_matrix(2.0)

def test_state_dependent_correlation():
    """Test state-dependent correlation structure."""
    # Create state-dependent correlation function
    def corr_func(state):
        r, S, v = state
        # Correlation increases with interest rate
        base_corr = 0.5
        adjustment = 0.2 * (r - 0.03) / 0.03
        return np.array([[1.0, base_corr + adjustment, 0.3],
                        [base_corr + adjustment, 1.0, 0.2],
                        [0.3, 0.2, 1.0]])
    
    # Create correlation structure
    corr = StateDependentCorrelation(corr_func)
    
    # Test different states
    states = [
        {'r': 0.03, 'S': 100.0, 'v': 0.04},  # Base state
        {'r': 0.06, 'S': 100.0, 'v': 0.04},  # High rate
        {'r': 0.01, 'S': 100.0, 'v': 0.04}   # Low rate
    ]
    
    for state in states:
        corr_matrix = corr.get_correlation_matrix(state=state)
        assert corr_matrix.shape == (3, 3)
        assert np.all(np.diag(corr_matrix) == 1.0)
        assert np.all(np.abs(corr_matrix) <= 1.0)
        assert np.allclose(corr_matrix, corr_matrix.T)
        
        # Check positive definiteness
        eigenvalues = np.linalg.eigvals(corr_matrix)
        assert np.all(eigenvalues > 0)

def test_regime_switching_correlation():
    """Test regime-switching correlation structure."""
    # Define regimes and their correlation matrices
    regimes = {
        'low_vol': np.array([[1.0, 0.3, 0.2],
                            [0.3, 1.0, 0.1],
                            [0.2, 0.1, 1.0]]),
        'high_vol': np.array([[1.0, 0.7, 0.5],
                            [0.7, 1.0, 0.4],
                            [0.5, 0.4, 1.0]])
    }
    
    # Define regime transition probabilities
    transition_probs = {
        'low_vol': {'low_vol': 0.8, 'high_vol': 0.2},
        'high_vol': {'low_vol': 0.3, 'high_vol': 0.7}
    }
    
    # Create correlation structure
    corr = RegimeSwitchingCorrelation(regimes, transition_probs)
    
    # Test initial regime
    corr_matrix = corr.get_correlation_matrix(regime='low_vol')
    assert np.allclose(corr_matrix, regimes['low_vol'])
    
    # Test regime switching
    corr.switch_regime('high_vol')
    corr_matrix = corr.get_correlation_matrix()
    assert np.allclose(corr_matrix, regimes['high_vol'])
    
    # Test invalid regime
    with pytest.raises(ValidationError):
        corr.get_correlation_matrix(regime='invalid_regime')

def test_stochastic_correlation():
    """Test stochastic correlation structure."""
    # Define parameters for stochastic correlation
    kappa = 2.0  # Mean reversion speed
    theta = 0.5  # Long-term mean
    sigma = 0.2  # Volatility
    rho0 = 0.3  # Initial correlation
    
    # Create correlation structure
    corr = StochasticCorrelation(kappa, theta, sigma, rho0)
    
    # Test correlation evolution
    T = 1.0
    n_steps = 100
    dt = T / n_steps
    
    correlations = []
    for i in range(n_steps):
        t = i * dt
        corr_matrix = corr.get_correlation_matrix(t)
        correlations.append(corr_matrix[0, 1])  # Store rate-equity correlation
    
    # Check correlation properties
    correlations = np.array(correlations)
    assert np.all(np.abs(correlations) <= 1.0)
    assert np.all(np.diff(correlations) != 0)  # Should be stochastic
    
    # Test mean reversion
    mean_corr = np.mean(correlations)
    assert abs(mean_corr - theta) < 0.1

def test_copula_correlation():
    """Test copula-based correlation structure."""
    # Define copula parameters
    params = {
        'type': 'gaussian',
        'rho': 0.5
    }
    
    # Create correlation structure
    corr = CopulaCorrelation(params)
    
    # Generate correlated uniforms
    n_samples = 10000
    uniforms = corr.generate_correlated_uniforms(n_samples)
    
    # Check properties
    assert uniforms.shape == (n_samples, 3)
    assert np.all(uniforms >= 0)
    assert np.all(uniforms <= 1)
    
    # Check correlation
    corr_matrix = np.corrcoef(uniforms.T)
    assert np.all(np.diag(corr_matrix) == 1.0)
    assert np.all(np.abs(corr_matrix) <= 1.0)
    
    # Test different copula types
    copula_types = ['gaussian', 'student', 'clayton', 'gumbel']
    for copula_type in copula_types:
        params['type'] = copula_type
        corr = CopulaCorrelation(params)
        uniforms = corr.generate_correlated_uniforms(n_samples)
        assert np.all(np.isfinite(uniforms))
        assert np.all(uniforms >= 0)
        assert np.all(uniforms <= 1)

def test_correlation_validation():
    """Test correlation validation."""
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
    
    with pytest.raises(ValidationError):
        TimeDependentCorrelation(
            time_points=np.array([0.0, 1.0]),
            correlation_matrices=[invalid_matrix, invalid_matrix]
        )
    
    # Test non-symmetric correlation matrix
    with pytest.raises(ValidationError):
        TimeDependentCorrelation(
            time_points=np.array([0.0, 1.0]),
            correlation_matrices=[
                np.array([[1.0, 0.5],
                         [0.3, 1.0]])  # Not symmetric
            ]
        )
    
    # Test non-positive definite correlation matrix
    with pytest.raises(ValidationError):
        TimeDependentCorrelation(
            time_points=np.array([0.0, 1.0]),
            correlation_matrices=[
                np.array([[1.0, 0.9, 0.9],
                         [0.9, 1.0, 0.9],
                         [0.9, 0.9, 1.0]])  # Not positive definite
            ]
        ) 