"""
Tests for state-dependent correlation structures.

This module contains tests for the state-dependent correlation implementation,
including initialization, validation, and correlation matrix computation.
"""

import numpy as np
import pytest

from chen3.correlation.models.state_dependent import (
    StateDependentCorrelation,
    linear_state_correlation,
    exponential_state_correlation,
)
from chen3.correlation.utils.exceptions import CorrelationError, CorrelationValidationError


def test_state_dependent_correlation_initialization():
    """Test initialization of state-dependent correlation."""
    # Test with linear correlation function
    default_state = {"r": 0.05, "S": 100.0, "v": 0.2}
    corr = StateDependentCorrelation(
        correlation_function=linear_state_correlation,
        default_state=default_state,
        n_factors=3,
    )
    assert corr.n_factors == 3
    assert corr.name == "StateDependentCorrelation"
    assert corr.correlation_function == linear_state_correlation
    assert corr.default_state == default_state

    # Test with exponential correlation function
    corr = StateDependentCorrelation(
        correlation_function=exponential_state_correlation,
        default_state=default_state,
        n_factors=3,
    )
    assert corr.n_factors == 3
    assert corr.correlation_function == exponential_state_correlation


def test_state_dependent_correlation_validation():
    """Test validation of state-dependent correlation."""
    # Test invalid correlation function
    with pytest.raises(CorrelationValidationError):
        StateDependentCorrelation(
            correlation_function="not a function",
            default_state={"r": 0.05, "S": 100.0, "v": 0.2},
        )

    # Test invalid default state
    def invalid_corr_func(state):
        return np.array([[1.0, 2.0], [2.0, 1.0]])  # Invalid correlation matrix

    with pytest.raises(CorrelationValidationError):
        StateDependentCorrelation(
            correlation_function=invalid_corr_func,
            default_state={"r": 0.05, "S": 100.0, "v": 0.2},
        )


def test_state_dependent_correlation_matrix():
    """Test correlation matrix computation."""
    default_state = {"r": 0.05, "S": 100.0, "v": 0.2}
    corr = StateDependentCorrelation(
        correlation_function=linear_state_correlation,
        default_state=default_state,
    )

    # Test with default state
    matrix = corr.get_correlation_matrix()
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, matrix.T)  # Symmetric
    assert np.all(np.diag(matrix) == 1.0)  # Unit diagonal
    assert np.all(np.abs(matrix) <= 1.0)  # Valid correlations

    # Test with custom state
    custom_state = {"r": 0.06, "S": 110.0, "v": 0.25}
    matrix = corr.get_correlation_matrix(state=custom_state)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, matrix.T)
    assert np.all(np.diag(matrix) == 1.0)
    assert np.all(np.abs(matrix) <= 1.0)

    # Test with None state (should use default)
    matrix = corr.get_correlation_matrix(state=None)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, matrix.T)
    assert np.all(np.diag(matrix) == 1.0)
    assert np.all(np.abs(matrix) <= 1.0)


def test_state_dependent_correlation_errors():
    """Test error handling in state-dependent correlation."""
    default_state = {"r": 0.05, "S": 100.0, "v": 0.2}
    corr = StateDependentCorrelation(
        correlation_function=linear_state_correlation,
        default_state=default_state,
    )

    # Test with invalid state
    with pytest.raises(CorrelationError):
        corr.get_correlation_matrix(state={"invalid": 0.0})

    # Test with invalid correlation function
    def error_corr_func(state):
        raise ValueError("Test error")

    corr.correlation_function = error_corr_func
    with pytest.raises(CorrelationError):
        corr.get_correlation_matrix()


def test_linear_state_correlation():
    """Test linear state-dependent correlation function."""
    state = {"r": 0.05, "S": 100.0, "v": 0.2}
    corr = linear_state_correlation(state)

    assert isinstance(corr, np.ndarray)
    assert corr.shape == (3, 3)
    assert np.allclose(corr, corr.T)
    assert np.all(np.diag(corr) == 1.0)
    assert np.all(np.abs(corr) <= 1.0)


def test_exponential_state_correlation():
    """Test exponential state-dependent correlation function."""
    state = {"r": 0.05, "S": 100.0, "v": 0.2}
    corr = exponential_state_correlation(state)

    assert isinstance(corr, np.ndarray)
    assert corr.shape == (3, 3)
    assert np.allclose(corr, corr.T)
    assert np.all(np.diag(corr) == 1.0)
    assert np.all(np.abs(corr) <= 1.0)
