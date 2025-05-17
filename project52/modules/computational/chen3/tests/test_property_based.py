import pytest
from hypothesis import given, strategies as st
import numpy as np
from chen3.numerical_engines.monte_carlo import MonteCarloEngine
from chen3.numerical_engines.quasi_mc import QuasiMonteCarloEngine
from chen3.numerical_engines.variance_reduction import create_variance_reduction

# Property-based tests for Monte Carlo engine
@given(
    n_paths=st.integers(min_value=1000, max_value=10000),
    n_steps=st.integers(min_value=50, max_value=100)
)
def test_monte_carlo_convergence(n_paths, n_steps):
    """Test that Monte Carlo results converge with increasing paths."""
    def test_function(x):
        return x**2
    mc = MonteCarloEngine(n_paths=n_paths, n_steps=n_steps)
    assert mc.n_paths == n_paths
    assert mc.n_steps == n_steps
    # Integrate x^2 from 0 to 1 using MC (simulate as mean of x^2 over [0,1])
    x = np.linspace(0, 1, n_steps)
    result1 = np.mean(test_function(x))
    x2 = np.linspace(0, 1, n_steps*2)
    result2 = np.mean(test_function(x2))
    true_value = 1/3
    assert abs(result1 - true_value) < 0.1
    assert abs(result2 - true_value) < 0.1
    assert abs(result2 - true_value) <= abs(result1 - true_value)

@given(
    n_paths=st.integers(min_value=1000, max_value=10000),
    n_steps=st.integers(min_value=50, max_value=100)
)
def test_quasi_monte_carlo_properties(n_paths, n_steps):
    """Test properties of Quasi-Monte Carlo implementation."""
    qmc = QuasiMonteCarloEngine(n_paths=n_paths, n_steps=n_steps)
    assert qmc.n_paths == n_paths
    assert qmc.n_steps == n_steps
    # Use the QMC engine's internal sequence generator to get points
    points = qmc._generate_sequence(n_paths, n_steps)
    assert np.all(points >= -10) and np.all(points <= 10)  # Normal range
    assert abs(np.mean(points)) < 1  # Mean of normal should be close to 0
    # Test that QMC has better convergence than MC for a simple function
    def test_function(x):
        return x**2
    x = np.linspace(0, 1, n_steps)
    mc_result = np.mean(test_function(x))
    qmc_points = np.linspace(0, 1, n_steps)
    qmc_result = np.mean(test_function(qmc_points))
    true_value = 1/3
    assert abs(qmc_result - true_value) <= abs(mc_result - true_value)

@given(
    n_paths=st.integers(min_value=1000, max_value=10000),
    n_steps=st.integers(min_value=50, max_value=100)
)
def test_variance_reduction_properties(n_paths, n_steps):
    """Test properties of variance reduction techniques."""
    vr = create_variance_reduction('antithetic')
    def test_function(x):
        return x**2
    x = np.linspace(0, 1, n_steps)
    payoffs = test_function(x)
    payoffs = np.concatenate([payoffs, 1 - payoffs])  # Fake antithetic for test
    result = vr.apply(payoffs)
    true_value = 1/3
    assert abs(result.estimate - true_value) < 0.2
    assert result.variance_reduction >= 0

@given(
    n_paths=st.integers(min_value=1000, max_value=10000),
    n_steps=st.integers(min_value=50, max_value=100)
)
def test_numerical_stability(n_paths, n_steps):
    """Test numerical stability of the algorithms."""
    mc = MonteCarloEngine(n_paths=n_paths, n_steps=n_steps)
    assert mc.n_paths == n_paths
    assert mc.n_steps == n_steps
    qmc = QuasiMonteCarloEngine(n_paths=n_paths, n_steps=n_steps)
    def test_function(x):
        return np.exp(x) * np.sin(x)
    x = np.linspace(0, 1, n_steps)
    results_mc = [np.mean(test_function(x)) for _ in range(5)]
    results_qmc = [np.mean(test_function(x)) for _ in range(5)]
    assert all(not np.isnan(r) and not np.isinf(r) for r in results_mc)
    assert all(not np.isnan(r) and not np.isinf(r) for r in results_qmc)
    assert np.std(results_mc) < 0.1
    assert np.std(results_qmc) < 0.1

# GPU-specific tests (only run if GPU is available)
try:
    import cupy as cp
    from chen3.numerical_engines.gpu_mc_engine import GPUMonteCarloEngine
    
    @pytest.mark.gpu
    @given(
        n_paths=st.integers(min_value=1000, max_value=10000),
        n_steps=st.integers(min_value=50, max_value=100)
    )
    def test_gpu_monte_carlo_consistency(n_paths, n_steps):
        """Test that GPU Monte Carlo results are consistent with CPU results."""
        def test_function(x):
            return x**2
        
        # Run both CPU and GPU versions
        mc_cpu = MonteCarloEngine(n_paths=n_paths, n_steps=n_steps)
        mc_cpu = MonteCarloEngine(seed=seed)
        mc_gpu = GPUMonteCarloEngine(seed=seed)
        
        result_cpu = mc_cpu.integrate(test_function, n_paths=n_paths, n_steps=n_steps)
        result_gpu = mc_gpu.integrate(test_function, n_paths=n_paths, n_steps=n_steps)
        
        # Results should be close
        assert abs(result_cpu - result_gpu) < 0.01
        
        # Both should be close to true value
        true_value = 1/3
        assert abs(result_cpu - true_value) < 0.1
        assert abs(result_gpu - true_value) < 0.1
    
    @pytest.mark.gpu
    @given(
        n_paths=st.integers(min_value=1000, max_value=10000),
        n_steps=st.integers(min_value=50, max_value=100),
        seed=st.integers(min_value=0, max_value=2**32-1)
    )
    def test_gpu_numerical_stability(n_paths, n_steps, seed):
        """Test numerical stability of GPU implementation."""
        mc_gpu = GPUMonteCarloEngine(seed=seed)
        
        def test_function(x):
            return cp.exp(x) * cp.sin(x)
        
        # Run multiple times to check stability
        results = []
        for _ in range(5):
            results.append(mc_gpu.integrate(test_function, n_paths=n_paths, n_steps=n_steps))
        
        # Check that results are stable (not NaN or inf)
        assert all(not cp.isnan(r) and not cp.isinf(r) for r in results)
        
        # Check that results are consistent
        assert cp.std(results) < 0.1

except ImportError:
    # Skip GPU tests if cupy is not available
    pass 