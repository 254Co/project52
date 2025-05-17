"""
Tests for the wavelet correlation model.
"""

import numpy as np
import pytest
from ..models.wavelet import WaveletCorrelation, create_wavelet_correlation
from ..utils.exceptions import CorrelationValidationError, CorrelationError

def test_initialization():
    """Test initialization of wavelet correlation."""
    # Test default initialization
    corr = WaveletCorrelation()
    assert corr.n_factors == 3
    assert len(corr.scales) == 3
    assert len(corr.coefficients) == 3
    assert len(corr.coefficient_functions) == 3
    assert corr.wavelet_type == 'haar'
    
    # Test custom initialization
    scales = [1.0, 2.0, 4.0]
    coefficients = [0.8, 0.5, 0.3]
    coefficient_functions = [
        lambda t, s: 0.8,
        lambda t, s: 0.5,
        lambda t, s: 0.3
    ]
    corr = WaveletCorrelation(
        scales=scales,
        coefficients=coefficients,
        coefficient_functions=coefficient_functions,
        wavelet_type='db4',
        n_factors=3,
        name="CustomWavelet"
    )
    assert corr.name == "CustomWavelet"
    assert corr.scales == scales
    assert corr.coefficients == coefficients
    assert corr.wavelet_type == 'db4'
    
    # Test invalid initialization
    with pytest.raises(CorrelationValidationError):
        WaveletCorrelation(scales=[1.0, 2.0])  # Wrong number of scales
    
    with pytest.raises(CorrelationValidationError):
        WaveletCorrelation(coefficients=[1.0, 0.5])  # Wrong number of coefficients
    
    with pytest.raises(CorrelationValidationError):
        WaveletCorrelation(coefficient_functions=[lambda t, s: 1.0])  # Wrong number of functions
    
    with pytest.raises(CorrelationValidationError):
        WaveletCorrelation(scales=[-1.0, 2.0, 4.0])  # Invalid scale
    
    with pytest.raises(CorrelationValidationError):
        WaveletCorrelation(coefficients=[1.5, 0.5, 0.3])  # Invalid coefficient
    
    with pytest.raises(CorrelationValidationError):
        WaveletCorrelation(wavelet_type='invalid')  # Invalid wavelet type

def test_correlation_matrix():
    """Test correlation matrix computation."""
    # Test constant coefficients
    scales = [1.0, 2.0, 4.0]
    coefficients = [0.8, 0.5, 0.3]
    coefficient_functions = [
        lambda t, s: 0.8,
        lambda t, s: 0.5,
        lambda t, s: 0.3
    ]
    corr = WaveletCorrelation(
        scales=scales,
        coefficients=coefficients,
        coefficient_functions=coefficient_functions
    )
    matrix = corr.get_correlation_matrix()
    
    # Check shape and properties
    assert matrix.shape == (3, 3)
    assert np.all(np.diag(matrix) == 1.0)
    assert np.all(np.abs(matrix) <= 1.0)
    
    # Test time-dependent coefficients
    coefficient_functions = [
        lambda t, s: np.exp(-0.1 * t),
        lambda t, s: 0.5 * np.exp(-0.1 * t),
        lambda t, s: 0.3 * np.exp(-0.1 * t)
    ]
    corr = WaveletCorrelation(
        scales=scales,
        coefficients=coefficients,
        coefficient_functions=coefficient_functions
    )
    matrix_t0 = corr.get_correlation_matrix(t=0.0)
    matrix_t1 = corr.get_correlation_matrix(t=1.0)
    assert not np.allclose(matrix_t0, matrix_t1)
    
    # Test state-dependent coefficients
    coefficient_functions = [
        lambda t, s: 0.8 if s is None else np.exp(-0.1 * abs(s.get('rate', 0.0))),
        lambda t, s: 0.5 if s is None else 0.5 * np.exp(-0.1 * abs(s.get('rate', 0.0))),
        lambda t, s: 0.3 if s is None else 0.3 * np.exp(-0.1 * abs(s.get('rate', 0.0)))
    ]
    corr = WaveletCorrelation(
        scales=scales,
        coefficients=coefficients,
        coefficient_functions=coefficient_functions
    )
    matrix_no_state = corr.get_correlation_matrix(t=0.0)
    matrix_with_state = corr.get_correlation_matrix(t=0.0, state={'rate': 0.1})
    assert not np.allclose(matrix_no_state, matrix_with_state)

def test_coefficient_computation():
    """Test coefficient computation."""
    # Test constant coefficients
    coefficient_functions = [
        lambda t, s: 0.8,
        lambda t, s: 0.5,
        lambda t, s: 0.3
    ]
    corr = WaveletCorrelation(coefficient_functions=coefficient_functions)
    coefficients = corr._compute_coefficients(t=0.0)
    assert np.allclose(coefficients, [0.8, 0.5, 0.3])
    
    # Test time-dependent coefficients
    coefficient_functions = [
        lambda t, s: np.exp(-0.1 * t),
        lambda t, s: 0.5 * np.exp(-0.1 * t),
        lambda t, s: 0.3 * np.exp(-0.1 * t)
    ]
    corr = WaveletCorrelation(coefficient_functions=coefficient_functions)
    coefficients_t0 = corr._compute_coefficients(t=0.0)
    coefficients_t1 = corr._compute_coefficients(t=1.0)
    assert not np.allclose(coefficients_t0, coefficients_t1)
    assert np.all(coefficients_t1 <= coefficients_t0)

def test_wavelet_matrix():
    """Test wavelet matrix computation."""
    corr = WaveletCorrelation()
    W = corr._compute_wavelet_matrix()
    
    # Check shape and properties
    assert W.shape == (3, 3)
    assert np.all(np.diag(W) == 1.0)
    assert np.all(np.abs(W) <= 1.0)

def test_create_wavelet_correlation():
    """Test factory function for creating wavelet correlations."""
    # Test constant coefficients
    corr = create_wavelet_correlation(scale_type='constant')
    assert isinstance(corr, WaveletCorrelation)
    assert corr.n_factors == 3
    matrix_t0 = corr.get_correlation_matrix(t=0.0)
    matrix_t1 = corr.get_correlation_matrix(t=1.0)
    assert np.allclose(matrix_t0, matrix_t1)
    
    # Test decaying coefficients
    corr = create_wavelet_correlation(scale_type='decay', decay_rate=0.1)
    matrix_t0 = corr.get_correlation_matrix(t=0.0)
    matrix_t1 = corr.get_correlation_matrix(t=1.0)
    assert not np.allclose(matrix_t0, matrix_t1)
    
    # Test oscillating coefficients
    corr = create_wavelet_correlation(scale_type='oscillate', decay_rate=0.1)
    matrix_t0 = corr.get_correlation_matrix(t=0.0)
    matrix_t1 = corr.get_correlation_matrix(t=1.0)
    assert not np.allclose(matrix_t0, matrix_t1)
    
    # Test invalid scale type
    with pytest.raises(ValueError):
        create_wavelet_correlation(scale_type='invalid')

def test_string_representations():
    """Test string representations of wavelet correlation."""
    corr = WaveletCorrelation()
    
    # Test string representation
    assert str(corr) == "WaveletCorrelation(n_factors=3, wavelet_type='haar')"
    
    # Test detailed representation
    repr_str = repr(corr)
    assert "WaveletCorrelation" in repr_str
    assert "scales=" in repr_str
    assert "coefficients=" in repr_str
    assert "coefficient_functions=" in repr_str
    assert "wavelet_type='haar'" in repr_str
    assert "n_factors=3" in repr_str 