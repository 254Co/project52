"""
Unit tests for the base correlation module.
"""

import numpy as np
import pytest
from ..core.base import BaseCorrelation
from ..utils.exceptions import CorrelationError

class TestCorrelation(BaseCorrelation):
    """Test implementation of BaseCorrelation."""
    
    def get_correlation_matrix(
        self,
        t: float = 0.0,
        state: dict = None
    ) -> np.ndarray:
        """Return a simple correlation matrix for testing."""
        return np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.2],
            [0.3, 0.2, 1.0]
        ])

def test_initialization():
    """Test correlation initialization."""
    # Test default initialization
    corr = TestCorrelation()
    assert corr.n_factors == 3
    assert corr.name == "TestCorrelation"
    
    # Test custom initialization
    corr = TestCorrelation(n_factors=4, name="CustomCorr")
    assert corr.n_factors == 4
    assert corr.name == "CustomCorr"
    
    # Test invalid initialization
    with pytest.raises(CorrelationError):
        TestCorrelation(n_factors=1)

def test_correlation_matrix_validation():
    """Test correlation matrix validation."""
    corr = TestCorrelation()
    
    # Test valid matrix
    valid_matrix = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.2],
        [0.3, 0.2, 1.0]
    ])
    corr._validate_correlation_matrix(valid_matrix)
    
    # Test invalid matrix types
    with pytest.raises(CorrelationError):
        corr._validate_correlation_matrix([[1.0, 0.5], [0.5, 1.0]])
    
    # Test invalid matrix shape
    with pytest.raises(CorrelationError):
        corr._validate_correlation_matrix(np.array([[1.0, 0.5], [0.5, 1.0]]))
    
    # Test non-symmetric matrix
    with pytest.raises(CorrelationError):
        corr._validate_correlation_matrix(np.array([
            [1.0, 0.5, 0.3],
            [0.6, 1.0, 0.2],
            [0.3, 0.2, 1.0]
        ]))
    
    # Test invalid diagonal
    with pytest.raises(CorrelationError):
        corr._validate_correlation_matrix(np.array([
            [1.0, 0.5, 0.3],
            [0.5, 0.9, 0.2],
            [0.3, 0.2, 1.0]
        ]))
    
    # Test invalid correlation values
    with pytest.raises(CorrelationError):
        corr._validate_correlation_matrix(np.array([
            [1.0, 1.5, 0.3],
            [1.5, 1.0, 0.2],
            [0.3, 0.2, 1.0]
        ]))
    
    # Test non-positive definite matrix
    with pytest.raises(CorrelationError):
        corr._validate_correlation_matrix(np.array([
            [1.0, 0.9, 0.9],
            [0.9, 1.0, 0.9],
            [0.9, 0.9, 1.0]
        ]))

def test_cholesky_decomposition():
    """Test Cholesky decomposition."""
    corr = TestCorrelation()
    
    # Test valid decomposition
    matrix = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.2],
        [0.3, 0.2, 1.0]
    ])
    L = corr._compute_cholesky(matrix)
    assert np.allclose(matrix, L @ L.T)
    
    # Test invalid decomposition
    with pytest.raises(CorrelationError):
        corr._compute_cholesky(np.array([
            [1.0, 0.9, 0.9],
            [0.9, 1.0, 0.9],
            [0.9, 0.9, 1.0]
        ]))

def test_correlated_random_generation():
    """Test generation of correlated random variables."""
    corr = TestCorrelation()
    
    # Test generation
    n_paths = 1000
    matrix = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.2],
        [0.3, 0.2, 1.0]
    ])
    X = corr._generate_correlated_random(n_paths, matrix)
    
    # Check shape
    assert X.shape == (n_paths, 3)
    
    # Check correlation
    empirical_corr = np.corrcoef(X.T)
    assert np.allclose(empirical_corr, matrix, atol=0.1)
    
    # Test invalid generation
    with pytest.raises(CorrelationError):
        corr._generate_correlated_random(n_paths, np.array([
            [1.0, 0.9, 0.9],
            [0.9, 1.0, 0.9],
            [0.9, 0.9, 1.0]
        ]))

def test_string_representations():
    """Test string representations."""
    corr = TestCorrelation()
    assert str(corr) == "TestCorrelation(n_factors=3)"
    assert repr(corr) == "TestCorrelation(n_factors=3, name='TestCorrelation')"
    
    corr = TestCorrelation(name="CustomCorr")
    assert str(corr) == "CustomCorr(n_factors=3)"
    assert repr(corr) == "TestCorrelation(n_factors=3, name='CustomCorr')" 