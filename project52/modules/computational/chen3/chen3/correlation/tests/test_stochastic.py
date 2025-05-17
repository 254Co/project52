"""
Unit tests for the stochastic correlation module.
"""

import numpy as np
import pytest
from ..models.stochastic import (
    StochasticCorrelation,
    create_stochastic_correlation
)
from ..utils.exceptions import CorrelationError, CorrelationValidationError

def test_initialization():
    """Test stochastic correlation initialization."""
    # Test default initialization
    corr = create_stochastic_correlation()
    assert corr.n_factors == 3
    assert corr.name == "StochasticCorrelation"
    assert corr.mean_reversion.shape == (3, 3)
    assert corr.long_term_mean.shape == (3, 3)
    assert corr.volatility.shape == (3, 3)
    assert corr.initial_corr.shape == (3, 3)
    
    # Test custom initialization
    n_factors = 4
    mean_reversion = np.ones((n_factors, n_factors))
    long_term_mean = np.ones((n_factors, n_factors)) * 0.5
    volatility = np.ones((n_factors, n_factors)) * 0.2
    initial_corr = np.ones((n_factors, n_factors)) * 0.3
    
    np.fill_diagonal(mean_reversion, 0)
    np.fill_diagonal(long_term_mean, 1)
    np.fill_diagonal(volatility, 0)
    np.fill_diagonal(initial_corr, 1)
    
    corr = StochasticCorrelation(
        mean_reversion=mean_reversion,
        long_term_mean=long_term_mean,
        volatility=volatility,
        initial_corr=initial_corr,
        n_factors=n_factors,
        name="CustomCorr"
    )
    assert corr.n_factors == n_factors
    assert corr.name == "CustomCorr"
    
    # Test invalid initialization
    with pytest.raises(CorrelationValidationError):
        StochasticCorrelation(
            mean_reversion=np.ones((3, 3)),
            long_term_mean=np.ones((3, 3)) * 1.5,  # Invalid long-term mean
            volatility=np.ones((3, 3)) * 0.2,
            initial_corr=np.ones((3, 3)) * 0.3
        )
    
    with pytest.raises(CorrelationValidationError):
        StochasticCorrelation(
            mean_reversion=np.ones((3, 3)) * -1,  # Invalid mean reversion
            long_term_mean=np.ones((3, 3)) * 0.5,
            volatility=np.ones((3, 3)) * 0.2,
            initial_corr=np.ones((3, 3)) * 0.3
        )
    
    with pytest.raises(CorrelationValidationError):
        StochasticCorrelation(
            mean_reversion=np.ones((3, 3)),
            long_term_mean=np.ones((3, 3)) * 0.5,
            volatility=np.ones((3, 3)) * -0.1,  # Invalid volatility
            initial_corr=np.ones((3, 3)) * 0.3
        )

def test_correlation_matrix():
    """Test correlation matrix computation."""
    corr = create_stochastic_correlation()
    
    # Test at t=0
    result = corr.get_correlation_matrix(t=0.0)
    assert result.shape == (3, 3)
    assert np.allclose(np.diag(result), 1.0)
    assert np.allclose(result, result.T)
    assert np.allclose(result, corr.initial_corr)
    
    # Test at t>0
    result = corr.get_correlation_matrix(t=1.0)
    assert result.shape == (3, 3)
    assert np.allclose(np.diag(result), 1.0)
    assert np.allclose(result, result.T)
    assert np.all(np.abs(result) <= 1)
    
    # Test with invalid time
    with pytest.raises(CorrelationError):
        corr.get_correlation_matrix(t=-1.0)

def test_simulation():
    """Test correlation simulation."""
    corr = create_stochastic_correlation()
    
    # Test simulation parameters
    result = corr._simulate_correlation(t=0.1, dt=0.01, n_steps=10)
    assert result.shape == (3, 3)
    assert np.allclose(np.diag(result), 1.0)
    assert np.allclose(result, result.T)
    assert np.all(np.abs(result) <= 1)
    
    # Test simulation with zero steps
    result = corr._simulate_correlation(t=0.0)
    assert np.allclose(result, corr.initial_corr)
    
    # Test simulation with large time
    result = corr._simulate_correlation(t=1000.0)
    assert result.shape == (3, 3)
    assert np.allclose(np.diag(result), 1.0)
    assert np.allclose(result, result.T)
    assert np.all(np.abs(result) <= 1)

def test_string_representations():
    """Test string representations."""
    corr = create_stochastic_correlation()
    
    # Test string representation
    expected_str = (
        f"StochasticCorrelation("
        f"mean_reversion={corr.mean_reversion.mean():.2f}, "
        f"volatility={corr.volatility.mean():.2f})"
    )
    assert str(corr) == expected_str
    
    # Test detailed representation
    expected_repr = (
        f"StochasticCorrelation("
        f"mean_reversion={corr.mean_reversion}, "
        f"long_term_mean={corr.long_term_mean}, "
        f"volatility={corr.volatility}, "
        f"initial_corr={corr.initial_corr}, "
        f"n_factors=3, "
        f"name='StochasticCorrelation')"
    )
    assert repr(corr) == expected_repr

def test_create_stochastic_correlation():
    """Test the stochastic correlation factory function."""
    # Test default parameters
    corr = create_stochastic_correlation()
    assert corr.n_factors == 3
    assert np.allclose(corr.mean_reversion, np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0]
    ]))
    assert np.allclose(corr.long_term_mean, np.array([
        [1.0, 0.5, 0.5],
        [0.5, 1.0, 0.5],
        [0.5, 0.5, 1.0]
    ]))
    assert np.allclose(corr.volatility, np.array([
        [0.0, 0.2, 0.2],
        [0.2, 0.0, 0.2],
        [0.2, 0.2, 0.0]
    ]))
    assert np.allclose(corr.initial_corr, np.array([
        [1.0, 0.3, 0.3],
        [0.3, 1.0, 0.3],
        [0.3, 0.3, 1.0]
    ]))
    
    # Test custom parameters
    corr = create_stochastic_correlation(
        mean_reversion=2.0,
        long_term_mean=0.7,
        volatility=0.3,
        initial_corr=0.4,
        n_factors=4
    )
    assert corr.n_factors == 4
    assert np.allclose(corr.mean_reversion, np.array([
        [0.0, 2.0, 2.0, 2.0],
        [2.0, 0.0, 2.0, 2.0],
        [2.0, 2.0, 0.0, 2.0],
        [2.0, 2.0, 2.0, 0.0]
    ]))
    assert np.allclose(corr.long_term_mean, np.array([
        [1.0, 0.7, 0.7, 0.7],
        [0.7, 1.0, 0.7, 0.7],
        [0.7, 0.7, 1.0, 0.7],
        [0.7, 0.7, 0.7, 1.0]
    ]))
    assert np.allclose(corr.volatility, np.array([
        [0.0, 0.3, 0.3, 0.3],
        [0.3, 0.0, 0.3, 0.3],
        [0.3, 0.3, 0.0, 0.3],
        [0.3, 0.3, 0.3, 0.0]
    ]))
    assert np.allclose(corr.initial_corr, np.array([
        [1.0, 0.4, 0.4, 0.4],
        [0.4, 1.0, 0.4, 0.4],
        [0.4, 0.4, 1.0, 0.4],
        [0.4, 0.4, 0.4, 1.0]
    ])) 