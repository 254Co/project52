"""
Tests for the network correlation model.
"""

import numpy as np
import pytest
from ..models.network import NetworkCorrelation, create_network_correlation
from ..utils.exceptions import CorrelationValidationError, CorrelationError

def test_initialization():
    """Test initialization of network correlation."""
    # Test default initialization
    corr = NetworkCorrelation()
    assert corr.n_factors == 3
    assert corr.decay_param == 1.0
    assert np.all(np.diag(corr.edge_weights) == 0.0)
    assert np.all(corr.edge_weights >= 0.0)
    
    # Test custom initialization
    edge_weights = np.array([
        [0.0, 0.5, 0.3],
        [0.5, 0.0, 0.2],
        [0.3, 0.2, 0.0]
    ])
    weight_function = lambda t, s: edge_weights
    corr = NetworkCorrelation(
        edge_weights=edge_weights,
        weight_function=weight_function,
        decay_param=0.5,
        n_factors=3,
        name="CustomNetwork"
    )
    assert corr.name == "CustomNetwork"
    assert corr.decay_param == 0.5
    assert np.allclose(corr.edge_weights, edge_weights)
    
    # Test invalid initialization
    with pytest.raises(CorrelationValidationError):
        NetworkCorrelation(edge_weights=np.array([[0.0, -0.5], [0.5, 0.0]]))  # Negative weights
    
    with pytest.raises(CorrelationValidationError):
        NetworkCorrelation(edge_weights=np.array([[1.0, 0.5], [0.5, 0.0]]))  # Non-zero diagonal
    
    with pytest.raises(CorrelationValidationError):
        NetworkCorrelation(edge_weights=np.array([[0.0, 0.5]]))  # Wrong shape

def test_correlation_matrix():
    """Test correlation matrix computation."""
    # Test constant weights
    edge_weights = np.array([
        [0.0, 0.5, 0.3],
        [0.5, 0.0, 0.2],
        [0.3, 0.2, 0.0]
    ])
    weight_function = lambda t, s: edge_weights
    corr = NetworkCorrelation(
        edge_weights=edge_weights,
        weight_function=weight_function
    )
    matrix = corr.get_correlation_matrix()
    
    # Check shape and properties
    assert matrix.shape == (3, 3)
    assert np.all(np.diag(matrix) == 1.0)
    assert np.all(np.abs(matrix) <= 1.0)
    
    # Test time-dependent weights
    weight_function = lambda t, s: edge_weights * np.exp(-0.1 * t)
    corr = NetworkCorrelation(
        edge_weights=edge_weights,
        weight_function=weight_function
    )
    matrix_t0 = corr.get_correlation_matrix(t=0.0)
    matrix_t1 = corr.get_correlation_matrix(t=1.0)
    assert not np.allclose(matrix_t0, matrix_t1)
    
    # Test state-dependent weights
    weight_function = lambda t, s: edge_weights if s is None else edge_weights * np.exp(-0.1 * abs(s.get('rate', 0.0)))
    corr = NetworkCorrelation(
        edge_weights=edge_weights,
        weight_function=weight_function
    )
    matrix_no_state = corr.get_correlation_matrix(t=0.0)
    matrix_with_state = corr.get_correlation_matrix(t=0.0, state={'rate': 0.1})
    assert not np.allclose(matrix_no_state, matrix_with_state)

def test_shortest_paths():
    """Test shortest path computation."""
    # Test direct connections
    edge_weights = np.array([
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.2],
        [0.0, 0.2, 0.0]
    ])
    corr = NetworkCorrelation(edge_weights=edge_weights)
    paths = corr._compute_shortest_paths(edge_weights)
    
    # Check direct paths
    assert paths[0, 1] == 2.0  # 1/0.5
    assert paths[1, 2] == 5.0  # 1/0.2
    
    # Check indirect paths
    assert paths[0, 2] == 7.0  # 1/0.5 + 1/0.2
    
    # Test disconnected nodes
    edge_weights = np.array([
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    corr = NetworkCorrelation(edge_weights=edge_weights)
    paths = corr._compute_shortest_paths(edge_weights)
    assert np.isinf(paths[0, 2])  # No path exists

def test_create_network_correlation():
    """Test factory function for creating network correlations."""
    # Test constant weights
    corr = create_network_correlation(weight_type='constant')
    assert isinstance(corr, NetworkCorrelation)
    assert corr.n_factors == 3
    matrix_t0 = corr.get_correlation_matrix(t=0.0)
    matrix_t1 = corr.get_correlation_matrix(t=1.0)
    assert np.allclose(matrix_t0, matrix_t1)
    
    # Test decaying weights
    corr = create_network_correlation(weight_type='decay', decay_rate=0.1)
    matrix_t0 = corr.get_correlation_matrix(t=0.0)
    matrix_t1 = corr.get_correlation_matrix(t=1.0)
    assert not np.allclose(matrix_t0, matrix_t1)
    
    # Test oscillating weights
    corr = create_network_correlation(weight_type='oscillate', decay_rate=0.1)
    matrix_t0 = corr.get_correlation_matrix(t=0.0)
    matrix_t1 = corr.get_correlation_matrix(t=1.0)
    assert not np.allclose(matrix_t0, matrix_t1)
    
    # Test invalid weight type
    with pytest.raises(ValueError):
        create_network_correlation(weight_type='invalid')

def test_string_representations():
    """Test string representations of network correlation."""
    corr = NetworkCorrelation()
    
    # Test string representation
    assert str(corr) == "NetworkCorrelation(n_factors=3, decay_param=1.00)"
    
    # Test detailed representation
    repr_str = repr(corr)
    assert "NetworkCorrelation" in repr_str
    assert "edge_weights=" in repr_str
    assert "weight_function=" in repr_str
    assert "decay_param=1.0" in repr_str
    assert "n_factors=3" in repr_str 