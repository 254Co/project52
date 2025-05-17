"""
Unit tests for the regime-switching correlation module.
"""

import numpy as np
import pytest

from ..models.regime_switching import (
    RegimeSwitchingCorrelation,
    create_two_regime_correlation,
)
from ..utils.exceptions import CorrelationError, CorrelationValidationError


def test_initialization():
    """Test regime-switching correlation initialization."""
    # Test default initialization
    corr = create_two_regime_correlation()
    assert corr.n_factors == 3
    assert corr.name == "RegimeSwitchingCorrelation"
    assert len(corr.correlation_matrices) == 2
    assert corr.initial_regime == 0

    # Test custom initialization
    n_factors = 4
    corr = RegimeSwitchingCorrelation(
        correlation_matrices=[np.eye(n_factors), np.ones((n_factors, n_factors)) * 0.5],
        transition_rates=np.array([[-0.1, 0.1], [0.1, -0.1]]),
        initial_regime=1,
        n_factors=n_factors,
        name="CustomCorr",
    )
    assert corr.n_factors == n_factors
    assert corr.name == "CustomCorr"
    assert len(corr.correlation_matrices) == 2
    assert corr.initial_regime == 1

    # Test invalid initialization
    with pytest.raises(CorrelationValidationError):
        RegimeSwitchingCorrelation(
            correlation_matrices=[np.eye(3)],
            transition_rates=np.array([[-0.1, 0.1], [0.1, -0.1]]),
            initial_regime=0,
        )

    with pytest.raises(CorrelationValidationError):
        RegimeSwitchingCorrelation(
            correlation_matrices=[np.eye(3), np.eye(3)],
            transition_rates=np.array([[0.1, 0.1], [0.1, 0.1]]),
            initial_regime=0,
        )

    with pytest.raises(CorrelationValidationError):
        RegimeSwitchingCorrelation(
            correlation_matrices=[np.eye(3), np.eye(3)],
            transition_rates=np.array([[-0.1, 0.1], [0.1, -0.1]]),
            initial_regime=2,
        )


def test_correlation_matrix():
    """Test correlation matrix computation."""
    corr = create_two_regime_correlation()

    # Test at t=0 (initial regime)
    result = corr.get_correlation_matrix(t=0.0)
    assert result.shape == (3, 3)
    assert np.allclose(np.diag(result), 1.0)
    assert np.allclose(result, result.T)
    assert np.allclose(result, corr.correlation_matrices[0])

    # Test at t=infinity (steady state)
    result = corr.get_correlation_matrix(t=1000.0)
    assert result.shape == (3, 3)
    assert np.allclose(np.diag(result), 1.0)
    assert np.allclose(result, result.T)

    # Test intermediate time
    result = corr.get_correlation_matrix(t=1.0)
    assert result.shape == (3, 3)
    assert np.allclose(np.diag(result), 1.0)
    assert np.allclose(result, result.T)

    # Test with invalid time
    with pytest.raises(CorrelationError):
        corr.get_correlation_matrix(t=-1.0)


def test_regime_probabilities():
    """Test regime probability computation."""
    corr = create_two_regime_correlation()

    # Test at t=0
    probs = corr._compute_regime_probabilities(t=0.0)
    assert len(probs) == 2
    assert np.allclose(probs, [1.0, 0.0])

    # Test at t=infinity
    probs = corr._compute_regime_probabilities(t=1000.0)
    assert len(probs) == 2
    assert np.allclose(probs, [0.5, 0.5])

    # Test intermediate time
    probs = corr._compute_regime_probabilities(t=1.0)
    assert len(probs) == 2
    assert np.allclose(probs.sum(), 1.0)
    assert np.all(probs >= 0)


def test_string_representations():
    """Test string representations."""
    corr = create_two_regime_correlation()

    # Test string representation
    expected_str = f"RegimeSwitchingCorrelation(" f"n_regimes=2, " f"initial_regime=0)"
    assert str(corr) == expected_str

    # Test detailed representation
    expected_repr = (
        f"RegimeSwitchingCorrelation("
        f"correlation_matrices=[{corr.correlation_matrices[0]}, {corr.correlation_matrices[1]}], "
        f"transition_rates={corr.transition_rates}, "
        f"initial_regime=0, "
        f"n_factors=3, "
        f"name='RegimeSwitchingCorrelation')"
    )
    assert repr(corr) == expected_repr


def test_create_two_regime_correlation():
    """Test the two-regime correlation factory function."""
    # Test default parameters
    corr = create_two_regime_correlation()
    assert corr.n_factors == 3
    assert len(corr.correlation_matrices) == 2
    assert np.allclose(
        corr.correlation_matrices[0],
        np.array([[1.0, 0.2, 0.2], [0.2, 1.0, 0.2], [0.2, 0.2, 1.0]]),
    )
    assert np.allclose(
        corr.correlation_matrices[1],
        np.array([[1.0, 0.8, 0.8], [0.8, 1.0, 0.8], [0.8, 0.8, 1.0]]),
    )

    # Test custom parameters
    corr = create_two_regime_correlation(
        low_corr=0.1, high_corr=0.9, transition_rate=0.2, n_factors=4
    )
    assert corr.n_factors == 4
    assert len(corr.correlation_matrices) == 2
    assert np.allclose(
        corr.correlation_matrices[0],
        np.array(
            [
                [1.0, 0.1, 0.1, 0.1],
                [0.1, 1.0, 0.1, 0.1],
                [0.1, 0.1, 1.0, 0.1],
                [0.1, 0.1, 0.1, 1.0],
            ]
        ),
    )
    assert np.allclose(
        corr.correlation_matrices[1],
        np.array(
            [
                [1.0, 0.9, 0.9, 0.9],
                [0.9, 1.0, 0.9, 0.9],
                [0.9, 0.9, 1.0, 0.9],
                [0.9, 0.9, 0.9, 1.0],
            ]
        ),
    )
