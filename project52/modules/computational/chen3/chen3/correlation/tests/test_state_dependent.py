"""
Unit tests for the state-dependent correlation module.
"""

import numpy as np
import pytest
from ..models.state_dependent import (
    StateDependentCorrelation,
    linear_state_correlation,
    exponential_state_correlation
)
from ..utils.exceptions import CorrelationError, CorrelationValidationError

def test_initialization():
    """Test state-dependent correlation initialization."""
    # Test default initialization
    corr = StateDependentCorrelation(linear_state_correlation)
    assert corr.n_factors == 3
    assert corr.name == "StateDependentCorrelation"
    assert corr.default_state == {'r': 0.0, 'S': 0.0, 'v': 0.0}
    
    # Test custom initialization
    default_state = {'r': 0.1, 'S': 0.2, 'v': 0.3}
    corr = StateDependentCorrelation(
        linear_state_correlation,
        default_state=default_state,
        n_factors=4,
        name="CustomCorr"
    )
    assert corr.n_factors == 4
    assert corr.name == "CustomCorr"
    assert corr.default_state == default_state
    
    # Test invalid initialization
    def invalid_corr_func(state):
        return np.array([[1.0, 1.5], [1.5, 1.0]])
    
    with pytest.raises(CorrelationValidationError):
        StateDependentCorrelation(invalid_corr_func)

def test_correlation_matrix():
    """Test correlation matrix computation."""
    corr = StateDependentCorrelation(linear_state_correlation)
    
    # Test with default state
    result = corr.get_correlation_matrix()
    assert result.shape == (3, 3)
    assert np.allclose(np.diag(result), 1.0)
    assert np.allclose(result, result.T)
    
    # Test with custom state
    state = {'r': 0.1, 'S': 0.2, 'v': 0.3}
    result = corr.get_correlation_matrix(state=state)
    assert result.shape == (3, 3)
    assert np.allclose(np.diag(result), 1.0)
    assert np.allclose(result, result.T)
    
    # Test linear correlation function
    result = linear_state_correlation(state)
    assert result[0, 1] == 0.5 + 0.1 * state['r']
    assert result[0, 2] == 0.3 + 0.05 * state['v']
    assert result[1, 2] == 0.2 + 0.15 * state['S']
    
    # Test exponential correlation function
    result = exponential_state_correlation(state)
    assert result[0, 1] == 0.5 * np.exp(0.1 * state['r'])
    assert result[0, 2] == 0.3 * np.exp(0.05 * state['v'])
    assert result[1, 2] == 0.2 * np.exp(0.15 * state['S'])

def test_validation():
    """Test correlation matrix validation."""
    def invalid_corr_func(state):
        return np.array([
            [1.0, 1.5, 0.3],
            [1.5, 1.0, 0.2],
            [0.3, 0.2, 1.0]
        ])
    
    with pytest.raises(CorrelationValidationError):
        StateDependentCorrelation(invalid_corr_func)
    
    def non_symmetric_corr_func(state):
        return np.array([
            [1.0, 0.5, 0.3],
            [0.6, 1.0, 0.2],
            [0.3, 0.2, 1.0]
        ])
    
    with pytest.raises(CorrelationValidationError):
        StateDependentCorrelation(non_symmetric_corr_func)
    
    def non_positive_definite_corr_func(state):
        return np.array([
            [1.0, 0.9, 0.9],
            [0.9, 1.0, 0.9],
            [0.9, 0.9, 1.0]
        ])
    
    with pytest.raises(CorrelationValidationError):
        StateDependentCorrelation(non_positive_definite_corr_func)

def test_string_representations():
    """Test string representations."""
    corr = StateDependentCorrelation(linear_state_correlation)
    
    # Test string representation
    expected_str = (
        f"StateDependentCorrelation("
        f"default_state={{'r': 0.0, 'S': 0.0, 'v': 0.0}})"
    )
    assert str(corr) == expected_str
    
    # Test detailed representation
    expected_repr = (
        f"StateDependentCorrelation("
        f"correlation_function=linear_state_correlation, "
        f"default_state={{'r': 0.0, 'S': 0.0, 'v': 0.0}}, "
        f"n_factors=3, "
        f"name='StateDependentCorrelation')"
    )
    assert repr(corr) == expected_repr 