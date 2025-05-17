"""
Tests for the spectral correlation model.
"""

import numpy as np
import pytest

from ..models.spectral import SpectralCorrelation, create_spectral_correlation
from ..utils.exceptions import CorrelationError, CorrelationValidationError


def test_initialization():
    """Test initialization of spectral correlation."""
    # Test default initialization
    corr = SpectralCorrelation()
    assert corr.n_factors == 3
    assert len(corr.eigenvalues) == 3
    assert corr.eigenvectors.shape == (3, 3)
    assert len(corr.eigenvalue_functions) == 3

    # Test custom initialization
    eigenvalues = [0.8, 0.5, 0.3]
    eigenvectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    eigenvalue_functions = [lambda t, s: 0.8, lambda t, s: 0.5, lambda t, s: 0.3]
    corr = SpectralCorrelation(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        eigenvalue_functions=eigenvalue_functions,
        n_factors=3,
        name="CustomSpectral",
    )
    assert corr.name == "CustomSpectral"
    assert np.allclose(corr.eigenvalues, eigenvalues)
    assert np.allclose(corr.eigenvectors, eigenvectors)

    # Test invalid initialization
    with pytest.raises(CorrelationValidationError):
        SpectralCorrelation(eigenvalues=[1.0, 0.5])  # Wrong number of eigenvalues

    with pytest.raises(CorrelationValidationError):
        SpectralCorrelation(eigenvectors=np.eye(2))  # Wrong eigenvector shape

    with pytest.raises(CorrelationValidationError):
        SpectralCorrelation(
            eigenvalue_functions=[lambda t, s: 1.0]
        )  # Wrong number of functions

    with pytest.raises(CorrelationValidationError):
        SpectralCorrelation(eigenvalues=[1.5, 0.5, 0.3])  # Invalid eigenvalue

    with pytest.raises(CorrelationValidationError):
        SpectralCorrelation(
            eigenvectors=np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        )  # Non-orthonormal eigenvectors


def test_correlation_matrix():
    """Test correlation matrix computation."""
    # Test constant eigenvalues
    eigenvalues = [0.8, 0.5, 0.3]
    eigenvectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    eigenvalue_functions = [lambda t, s: 0.8, lambda t, s: 0.5, lambda t, s: 0.3]
    corr = SpectralCorrelation(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        eigenvalue_functions=eigenvalue_functions,
    )
    matrix = corr.get_correlation_matrix()

    # Check shape and properties
    assert matrix.shape == (3, 3)
    assert np.all(np.diag(matrix) == 1.0)
    assert np.all(np.abs(matrix) <= 1.0)

    # Test time-dependent eigenvalues
    eigenvalue_functions = [
        lambda t, s: np.exp(-0.1 * t),
        lambda t, s: 0.5 * np.exp(-0.1 * t),
        lambda t, s: 0.3 * np.exp(-0.1 * t),
    ]
    corr = SpectralCorrelation(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        eigenvalue_functions=eigenvalue_functions,
    )
    matrix_t0 = corr.get_correlation_matrix(t=0.0)
    matrix_t1 = corr.get_correlation_matrix(t=1.0)
    assert not np.allclose(matrix_t0, matrix_t1)

    # Test state-dependent eigenvalues
    eigenvalue_functions = [
        lambda t, s: 0.8 if s is None else np.exp(-0.1 * abs(s.get("rate", 0.0))),
        lambda t, s: 0.5 if s is None else 0.5 * np.exp(-0.1 * abs(s.get("rate", 0.0))),
        lambda t, s: 0.3 if s is None else 0.3 * np.exp(-0.1 * abs(s.get("rate", 0.0))),
    ]
    corr = SpectralCorrelation(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        eigenvalue_functions=eigenvalue_functions,
    )
    matrix_no_state = corr.get_correlation_matrix(t=0.0)
    matrix_with_state = corr.get_correlation_matrix(t=0.0, state={"rate": 0.1})
    assert not np.allclose(matrix_no_state, matrix_with_state)


def test_eigenvalue_computation():
    """Test eigenvalue computation."""
    # Test constant eigenvalues
    eigenvalue_functions = [lambda t, s: 0.8, lambda t, s: 0.5, lambda t, s: 0.3]
    corr = SpectralCorrelation(eigenvalue_functions=eigenvalue_functions)
    eigenvalues = corr._compute_eigenvalues(t=0.0)
    assert np.allclose(eigenvalues, [0.8, 0.5, 0.3])

    # Test time-dependent eigenvalues
    eigenvalue_functions = [
        lambda t, s: np.exp(-0.1 * t),
        lambda t, s: 0.5 * np.exp(-0.1 * t),
        lambda t, s: 0.3 * np.exp(-0.1 * t),
    ]
    corr = SpectralCorrelation(eigenvalue_functions=eigenvalue_functions)
    eigenvalues_t0 = corr._compute_eigenvalues(t=0.0)
    eigenvalues_t1 = corr._compute_eigenvalues(t=1.0)
    assert not np.allclose(eigenvalues_t0, eigenvalues_t1)
    assert np.all(eigenvalues_t1 <= eigenvalues_t0)


def test_create_spectral_correlation():
    """Test factory function for creating spectral correlations."""
    # Test constant eigenvalues
    corr = create_spectral_correlation(eigenvalue_type="constant")
    assert isinstance(corr, SpectralCorrelation)
    assert corr.n_factors == 3
    matrix_t0 = corr.get_correlation_matrix(t=0.0)
    matrix_t1 = corr.get_correlation_matrix(t=1.0)
    assert np.allclose(matrix_t0, matrix_t1)

    # Test decaying eigenvalues
    corr = create_spectral_correlation(eigenvalue_type="decay", decay_rate=0.1)
    matrix_t0 = corr.get_correlation_matrix(t=0.0)
    matrix_t1 = corr.get_correlation_matrix(t=1.0)
    assert not np.allclose(matrix_t0, matrix_t1)

    # Test oscillating eigenvalues
    corr = create_spectral_correlation(eigenvalue_type="oscillate", decay_rate=0.1)
    matrix_t0 = corr.get_correlation_matrix(t=0.0)
    matrix_t1 = corr.get_correlation_matrix(t=1.0)
    assert not np.allclose(matrix_t0, matrix_t1)

    # Test invalid eigenvalue type
    with pytest.raises(ValueError):
        create_spectral_correlation(eigenvalue_type="invalid")


def test_string_representations():
    """Test string representations of spectral correlation."""
    corr = SpectralCorrelation()

    # Test string representation
    assert str(corr) == "SpectralCorrelation(n_factors=3)"

    # Test detailed representation
    repr_str = repr(corr)
    assert "SpectralCorrelation" in repr_str
    assert "eigenvalues=" in repr_str
    assert "eigenvectors=" in repr_str
    assert "eigenvalue_functions=" in repr_str
    assert "n_factors=3" in repr_str
