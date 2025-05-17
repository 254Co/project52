"""
Tests for the dynamic correlation model.
"""

import numpy as np
import pytest
from ..models.dynamic import DynamicCorrelation, create_dynamic_correlation
from ..models.constant import ConstantCorrelation
from chen3.correlation import TimeDependentCorrelation
from ..utils.exceptions import CorrelationValidationError, CorrelationError

def test_initialization():
    """Test initialization of dynamic correlation."""
    # Test default initialization
    corr1 = ConstantCorrelation(n_factors=3)
    corr2 = TimeDependentCorrelation(n_factors=3)
    dynamic_corr = DynamicCorrelation([corr1, corr2])
    assert dynamic_corr.n_factors == 3
    assert len(dynamic_corr.correlation_structures) == 2
    assert len(dynamic_corr.weight_functions) == 2
    
    # Test custom initialization
    weight_funcs = [
        lambda t, s: 0.7,
        lambda t, s: 0.3
    ]
    dynamic_corr = DynamicCorrelation(
        [corr1, corr2],
        weight_functions=weight_funcs,
        n_factors=3,
        name="CustomDynamic"
    )
    assert dynamic_corr.name == "CustomDynamic"
    
    # Test invalid initialization
    with pytest.raises(CorrelationValidationError):
        DynamicCorrelation([])  # Empty structures
    
    with pytest.raises(CorrelationValidationError):
        DynamicCorrelation([corr1], weight_functions=[lambda t, s: 0.5, lambda t, s: 0.5])
    
    with pytest.raises(CorrelationValidationError):
        DynamicCorrelation([corr1, ConstantCorrelation(n_factors=2)])

def test_correlation_matrix():
    """Test correlation matrix computation."""
    # Create test structures
    corr1 = ConstantCorrelation(n_factors=3)
    corr2 = TimeDependentCorrelation(n_factors=3)
    
    # Test equal weights
    dynamic_corr = DynamicCorrelation([corr1, corr2])
    corr = dynamic_corr.get_correlation_matrix(t=0.0)
    assert corr.shape == (3, 3)
    assert np.all(np.diag(corr) == 1.0)
    assert np.all(np.abs(corr) <= 1.0)
    
    # Test time-dependent weights
    weight_funcs = [
        lambda t, s: np.exp(-0.1 * t),
        lambda t, s: 1.0 - np.exp(-0.1 * t)
    ]
    dynamic_corr = DynamicCorrelation([corr1, corr2], weight_functions=weight_funcs)
    corr_t0 = dynamic_corr.get_correlation_matrix(t=0.0)
    corr_t1 = dynamic_corr.get_correlation_matrix(t=1.0)
    assert not np.allclose(corr_t0, corr_t1)
    
    # Test state-dependent weights
    weight_funcs = [
        lambda t, s: 0.7 if s is None else np.exp(-0.1 * abs(s.get('rate', 0.0))),
        lambda t, s: 0.3 if s is None else 1.0 - np.exp(-0.1 * abs(s.get('rate', 0.0)))
    ]
    dynamic_corr = DynamicCorrelation([corr1, corr2], weight_functions=weight_funcs)
    corr_no_state = dynamic_corr.get_correlation_matrix(t=0.0)
    corr_with_state = dynamic_corr.get_correlation_matrix(t=0.0, state={'rate': 0.1})
    assert not np.allclose(corr_no_state, corr_with_state)

def test_weight_computation():
    """Test weight computation."""
    corr1 = ConstantCorrelation(n_factors=3)
    corr2 = TimeDependentCorrelation(n_factors=3)
    
    # Test equal weights
    dynamic_corr = DynamicCorrelation([corr1, corr2])
    weights = dynamic_corr._compute_weights(t=0.0)
    assert np.allclose(weights, [0.5, 0.5])
    
    # Test time-dependent weights
    weight_funcs = [
        lambda t, s: np.exp(-0.1 * t),
        lambda t, s: 1.0 - np.exp(-0.1 * t)
    ]
    dynamic_corr = DynamicCorrelation([corr1, corr2], weight_functions=weight_funcs)
    weights_t0 = dynamic_corr._compute_weights(t=0.0)
    weights_t1 = dynamic_corr._compute_weights(t=1.0)
    assert not np.allclose(weights_t0, weights_t1)
    assert np.allclose(weights_t0 + weights_t1, 1.0)

def test_create_dynamic_correlation():
    """Test factory function for creating dynamic correlations."""
    corr1 = ConstantCorrelation(n_factors=3)
    corr2 = TimeDependentCorrelation(n_factors=3)
    
    # Test equal weights
    dynamic_corr = create_dynamic_correlation([corr1, corr2])
    assert isinstance(dynamic_corr, DynamicCorrelation)
    assert len(dynamic_corr.correlation_structures) == 2
    
    # Test time-dependent weights
    dynamic_corr = create_dynamic_correlation([corr1, corr2], weight_type='time')
    corr_t0 = dynamic_corr.get_correlation_matrix(t=0.0)
    corr_t1 = dynamic_corr.get_correlation_matrix(t=1.0)
    assert not np.allclose(corr_t0, corr_t1)
    
    # Test state-dependent weights
    dynamic_corr = create_dynamic_correlation([corr1, corr2], weight_type='state')
    corr_no_state = dynamic_corr.get_correlation_matrix(t=0.0)
    corr_with_state = dynamic_corr.get_correlation_matrix(t=0.0, state={'rate': 0.1})
    assert not np.allclose(corr_no_state, corr_with_state)
    
    # Test invalid weight type
    with pytest.raises(ValueError):
        create_dynamic_correlation([corr1, corr2], weight_type='invalid')

def test_string_representations():
    """Test string representations of dynamic correlation."""
    corr1 = ConstantCorrelation(n_factors=3)
    corr2 = TimeDependentCorrelation(n_factors=3)
    dynamic_corr = DynamicCorrelation([corr1, corr2])
    
    # Test string representation
    assert str(dynamic_corr) == "DynamicCorrelation(n_structures=2)"
    
    # Test detailed representation
    repr_str = repr(dynamic_corr)
    assert "DynamicCorrelation" in repr_str
    assert "correlation_structures" in repr_str
    assert "weight_functions" in repr_str
    assert "n_factors=3" in repr_str 