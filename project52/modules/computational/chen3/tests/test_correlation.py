"""
Tests for correlation structures in the Chen3 model.
"""
import pytest
import numpy as np
from chen3.correlation import TimeDependentCorrelation

def test_time_dependent_correlation(time_correlation):
    """Test time-dependent correlation structure."""
    # Test correlation at specific time points
    t0 = time_correlation.time_points[0]
    t1 = time_correlation.time_points[1]
    t2 = time_correlation.time_points[2]
    
    corr0 = time_correlation.get_correlation(t0)
    corr1 = time_correlation.get_correlation(t1)
    corr2 = time_correlation.get_correlation(t2)
    
    # Check that correlations match at time points
    assert np.allclose(corr0, time_correlation.correlation_matrices[0])
    assert np.allclose(corr1, time_correlation.correlation_matrices[1])
    assert np.allclose(corr2, time_correlation.correlation_matrices[2])
    
    # Test interpolation at intermediate point
    t_mid = (t0 + t1) / 2
    corr_mid = time_correlation.get_correlation(t_mid)
    
    # Check that interpolated correlation is valid
    assert np.all(np.diag(corr_mid) == 1.0)  # Diagonal elements are 1
    assert np.all(np.abs(corr_mid) <= 1.0)   # All elements are between -1 and 1
    assert np.allclose(corr_mid, corr_mid.T)  # Matrix is symmetric
    
    # Check positive definiteness
    eigenvalues = np.linalg.eigvals(corr_mid)
    assert np.all(eigenvalues > 0)

def test_invalid_correlation_matrix():
    """Test that invalid correlation matrices are rejected."""
    time_points = np.array([0.0, 1.0])
    # Non-positive definite matrix
    invalid_matrix = np.array([[1.0, 0.9, 0.9],
                             [0.9, 1.0, 0.9],
                             [0.9, 0.9, 1.0]])
    
    with pytest.raises(ValueError):
        TimeDependentCorrelation(
            time_points=time_points,
            correlation_matrices=[invalid_matrix, invalid_matrix]
        )

@pytest.mark.slow
def test_state_dependent_correlation(model_params):
    """Test that correlation changes with different states."""
    from chen3 import Chen3Model
    
    model = Chen3Model(model_params)
    T = 1.0
    n_steps = 100
    n_paths = 1000
    
    # Simulate paths
    paths = model.simulate(T, n_steps, n_paths)
    
    # Calculate realized correlations for different time windows
    window_size = 10
    for i in range(0, n_steps - window_size, window_size):
        window_paths = paths[:, i:i+window_size, :]
        
        # Calculate realized correlations
        rate_returns = np.diff(window_paths[:, :, 0], axis=1)
        equity_returns = np.diff(window_paths[:, :, 1], axis=1)
        var_returns = np.diff(window_paths[:, :, 2], axis=1)
        
        realized_corr = np.corrcoef(
            np.column_stack([
                rate_returns.flatten(),
                equity_returns.flatten(),
                var_returns.flatten()
            ])
        )
        
        # Check that realized correlation is valid
        assert np.all(np.diag(realized_corr) == 1.0)
        assert np.all(np.abs(realized_corr) <= 1.0)
        assert np.allclose(realized_corr, realized_corr.T) 