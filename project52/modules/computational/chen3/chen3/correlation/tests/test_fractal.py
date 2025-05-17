"""
Tests for the fractal correlation model.
"""

import numpy as np
import pytest

from ..models.fractal import FractalCorrelation, create_fractal_correlation
from ..utils.exceptions import CorrelationError, CorrelationValidationError


def test_initialization():
    """Test initialization of fractal correlation."""
    # Test default initialization
    corr = FractalCorrelation()
    assert corr.n_factors == 3
    assert corr.hurst == 0.5

    # Test custom initialization
    hurst = 0.7
    hurst_function = lambda t, s: 0.7
    corr = FractalCorrelation(
        hurst=hurst, hurst_function=hurst_function, n_factors=3, name="CustomFractal"
    )
    assert corr.name == "CustomFractal"
    assert corr.hurst == hurst

    # Test invalid initialization
    with pytest.raises(CorrelationValidationError):
        FractalCorrelation(hurst=0.0)  # Invalid Hurst exponent

    with pytest.raises(CorrelationValidationError):
        FractalCorrelation(hurst=1.0)  # Invalid Hurst exponent

    with pytest.raises(CorrelationValidationError):
        FractalCorrelation(hurst=-0.5)  # Invalid Hurst exponent


def test_correlation_matrix():
    """Test correlation matrix computation."""
    # Test constant Hurst exponent
    hurst = 0.5
    hurst_function = lambda t, s: 0.5
    corr = FractalCorrelation(hurst=hurst, hurst_function=hurst_function)
    matrix = corr.get_correlation_matrix()

    # Check shape and properties
    assert matrix.shape == (3, 3)
    assert np.all(np.diag(matrix) == 1.0)
    assert np.all(np.abs(matrix) <= 1.0)

    # Test time-dependent Hurst exponent
    hurst_function = lambda t, s: 0.5 * np.exp(-0.1 * t)
    corr = FractalCorrelation(hurst=0.5, hurst_function=hurst_function)
    matrix_t0 = corr.get_correlation_matrix(t=0.0)
    matrix_t1 = corr.get_correlation_matrix(t=1.0)
    assert not np.allclose(matrix_t0, matrix_t1)

    # Test state-dependent Hurst exponent
    hurst_function = lambda t, s: (
        0.5 if s is None else np.exp(-0.1 * abs(s.get("rate", 0.0)))
    )
    corr = FractalCorrelation(hurst=0.5, hurst_function=hurst_function)
    matrix_no_state = corr.get_correlation_matrix(t=0.0)
    matrix_with_state = corr.get_correlation_matrix(t=0.0, state={"rate": 0.1})
    assert not np.allclose(matrix_no_state, matrix_with_state)


def test_hurst_computation():
    """Test Hurst exponent computation."""
    # Test constant Hurst exponent
    hurst_function = lambda t, s: 0.5
    corr = FractalCorrelation(hurst_function=hurst_function)
    H = corr._compute_hurst(t=0.0)
    assert H == 0.5

    # Test time-dependent Hurst exponent
    hurst_function = lambda t, s: 0.5 * np.exp(-0.1 * t)
    corr = FractalCorrelation(hurst_function=hurst_function)
    H_t0 = corr._compute_hurst(t=0.0)
    H_t1 = corr._compute_hurst(t=1.0)
    assert not np.allclose(H_t0, H_t1)
    assert H_t1 < H_t0

    # Test clipping of Hurst exponent
    hurst_function = lambda t, s: 1.5  # Should be clipped to 0.99
    corr = FractalCorrelation(hurst_function=hurst_function)
    H = corr._compute_hurst(t=0.0)
    assert H == 0.99

    hurst_function = lambda t, s: -0.5  # Should be clipped to 0.01
    corr = FractalCorrelation(hurst_function=hurst_function)
    H = corr._compute_hurst(t=0.0)
    assert H == 0.01


def test_create_fractal_correlation():
    """Test factory function for creating fractal correlations."""
    # Test constant Hurst exponent
    corr = create_fractal_correlation(hurst_type="constant")
    assert isinstance(corr, FractalCorrelation)
    assert corr.n_factors == 3
    matrix_t0 = corr.get_correlation_matrix(t=0.0)
    matrix_t1 = corr.get_correlation_matrix(t=1.0)
    assert np.allclose(matrix_t0, matrix_t1)

    # Test decaying Hurst exponent
    corr = create_fractal_correlation(hurst_type="decay", decay_rate=0.1)
    matrix_t0 = corr.get_correlation_matrix(t=0.0)
    matrix_t1 = corr.get_correlation_matrix(t=1.0)
    assert not np.allclose(matrix_t0, matrix_t1)

    # Test oscillating Hurst exponent
    corr = create_fractal_correlation(hurst_type="oscillate", decay_rate=0.1)
    matrix_t0 = corr.get_correlation_matrix(t=0.0)
    matrix_t1 = corr.get_correlation_matrix(t=1.0)
    assert not np.allclose(matrix_t0, matrix_t1)

    # Test invalid Hurst type
    with pytest.raises(ValueError):
        create_fractal_correlation(hurst_type="invalid")


def test_string_representations():
    """Test string representations of fractal correlation."""
    corr = FractalCorrelation()

    # Test string representation
    assert str(corr) == "FractalCorrelation(n_factors=3, hurst=0.50)"

    # Test detailed representation
    repr_str = repr(corr)
    assert "FractalCorrelation" in repr_str
    assert "hurst=0.5" in repr_str
    assert "hurst_function=" in repr_str
    assert "n_factors=3" in repr_str
