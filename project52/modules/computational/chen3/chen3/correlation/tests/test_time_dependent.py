"""
Unit tests for the time-dependent correlation module.
"""

import numpy as np
import pytest

from chen3.correlation import TimeDependentCorrelation

from ..utils.exceptions import CorrelationError, CorrelationValidationError


def create_test_data():
    """Create test data for time-dependent correlation."""
    time_points = np.array([0.0, 1.0, 2.0])
    corr_matrices = [
        np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]]),
        np.array([[1.0, 0.6, 0.4], [0.6, 1.0, 0.3], [0.4, 0.3, 1.0]]),
        np.array([[1.0, 0.7, 0.5], [0.7, 1.0, 0.4], [0.5, 0.4, 1.0]]),
    ]
    return time_points, corr_matrices


def test_initialization():
    """Test time-dependent correlation initialization."""
    time_points, corr_matrices = create_test_data()

    # Test default initialization
    corr = TimeDependentCorrelation(time_points, corr_matrices)
    assert corr.n_factors == 3
    assert corr.name == "TimeDependentCorrelation"
    assert np.array_equal(corr.time_points, time_points)
    assert len(corr.correlation_matrices) == len(corr_matrices)

    # Test custom initialization
    corr = TimeDependentCorrelation(
        time_points, corr_matrices, n_factors=4, name="CustomCorr"
    )
    assert corr.n_factors == 4
    assert corr.name == "CustomCorr"

    # Test invalid initialization
    with pytest.raises(CorrelationValidationError):
        TimeDependentCorrelation(np.array([0.0, 1.0]), corr_matrices)

    with pytest.raises(CorrelationValidationError):
        TimeDependentCorrelation(np.array([1.0, 0.0]), corr_matrices)

    with pytest.raises(CorrelationError):
        TimeDependentCorrelation(time_points, [np.array([[1.0, 0.5], [0.5, 1.0]])])


def test_correlation_matrix():
    """Test correlation matrix computation."""
    time_points, corr_matrices = create_test_data()
    corr = TimeDependentCorrelation(time_points, corr_matrices)

    # Test at time points
    for t, matrix in zip(time_points, corr_matrices):
        result = corr.get_correlation_matrix(t)
        assert np.allclose(result, matrix)

    # Test interpolation
    t = 0.5
    result = corr.get_correlation_matrix(t)
    assert result.shape == (3, 3)
    assert np.allclose(np.diag(result), 1.0)
    assert np.allclose(result, result.T)

    # Test extrapolation
    t = -0.5
    result = corr.get_correlation_matrix(t)
    assert np.allclose(result, corr_matrices[0])

    t = 2.5
    result = corr.get_correlation_matrix(t)
    assert np.allclose(result, corr_matrices[-1])


def test_validation():
    """Test correlation matrix validation."""
    time_points, corr_matrices = create_test_data()

    # Test invalid correlation values
    invalid_matrix = np.array([[1.0, 1.5, 0.3], [1.5, 1.0, 0.2], [0.3, 0.2, 1.0]])
    with pytest.raises(CorrelationError):
        TimeDependentCorrelation(
            time_points, [corr_matrices[0], invalid_matrix, corr_matrices[2]]
        )

    # Test non-symmetric matrix
    invalid_matrix = np.array([[1.0, 0.5, 0.3], [0.6, 1.0, 0.2], [0.3, 0.2, 1.0]])
    with pytest.raises(CorrelationError):
        TimeDependentCorrelation(
            time_points, [corr_matrices[0], invalid_matrix, corr_matrices[2]]
        )

    # Test non-positive definite matrix
    invalid_matrix = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, 0.9], [0.9, 0.9, 1.0]])
    with pytest.raises(CorrelationError):
        TimeDependentCorrelation(
            time_points, [corr_matrices[0], invalid_matrix, corr_matrices[2]]
        )


def test_string_representations():
    """Test string representations."""
    time_points, corr_matrices = create_test_data()
    corr = TimeDependentCorrelation(time_points, corr_matrices)

    # Test string representation
    expected_str = (
        f"TimeDependentCorrelation("
        f"time_points={time_points}, "
        f"n_matrices={len(corr_matrices)})"
    )
    assert str(corr) == expected_str

    # Test detailed representation
    expected_repr = (
        f"TimeDependentCorrelation("
        f"time_points={time_points}, "
        f"correlation_matrices={corr_matrices}, "
        f"n_factors=3, "
        f"name='TimeDependentCorrelation')"
    )
    assert repr(corr) == expected_repr
