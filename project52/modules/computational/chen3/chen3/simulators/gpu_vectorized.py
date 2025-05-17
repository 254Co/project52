# File: chen3/simulators/gpu_vectorized.py
"""
GPU Vectorized Path Simulator Implementation

This module implements a GPU-accelerated path simulator for the three-factor
Chen model using CuPy. The simulator leverages GPU parallelization to efficiently
generate Monte Carlo paths for the coupled stochastic processes:
- Short rate (CIR process)
- Variance (CIR process)
- Stock price (Geometric Brownian motion with stochastic volatility)

The implementation uses the Euler-Maruyama scheme for numerical integration
and handles the correlation structure between the processes using Cholesky
decomposition, all while performing computations on the GPU.
"""

import cupy as cp
from .core import PathGenerator
from chen3.correlation import cholesky_correlation


class GPUVectorizedSimulator(PathGenerator):
    """
    GPU-accelerated path simulator for the three-factor Chen model.
    
    This class implements a highly efficient path generator that leverages
    GPU parallelization through CuPy to simulate the three-factor model.
    The implementation is optimized for performance on NVIDIA GPUs while
    maintaining numerical stability and accuracy.
    
    The simulator generates paths for the coupled system:
        dr = κ_r(θ_r - r)dt + σ_r√r dW₁
        dv = κ_v(θ_v - v)dt + σ_v√v dW₂
        dS = (r - q)S dt + √v S dW₃
    
    where the Brownian motions are correlated according to the model's
    correlation matrix.
    
    Attributes:
        model: The three-factor Chen model instance
        params: Simulation parameters including:
            - n_paths: Number of Monte Carlo paths
            - n_steps: Number of time steps
            - dt: Time step size
        rng: CuPy random number generator instance
    
    Example:
        >>> from chen3.config import Settings
        >>> settings = Settings(backend="gpu", n_paths=1000000, n_steps=252)
        >>> simulator = GPUVectorizedSimulator(model, settings)
        >>> paths = simulator.generate()  # shape: (n_paths, n_steps+1, 3)
    
    Notes:
        - Requires CUDA-capable GPU and CuPy installation
        - All computations are performed on the GPU
        - Random number generation is also performed on the GPU
    """
    
    def __init__(self, model, settings):
        """
        Initialize the GPU-accelerated path simulator.
        
        Args:
            model: The three-factor Chen model instance
            settings: Simulation settings including:
                - seed: Random number generator seed
                - n_paths: Number of Monte Carlo paths
                - n_steps: Number of time steps
                - dt: Time step size
        """
        super().__init__(model, settings)
        self.rng = cp.random.default_rng(settings.seed)

    def generate(self):
        """
        Generate Monte Carlo paths for the three-factor model on GPU.
        
        This method simulates paths for the coupled system of stochastic
        processes using the Euler-Maruyama scheme. The implementation is
        fully vectorized and executed on the GPU for maximum performance.
        
        The method:
        1. Initializes the state vectors on GPU
        2. Generates correlated random increments on GPU
        3. Updates each process using the Euler-Maruyama scheme
        4. Ensures non-negativity of rates and variance
        5. Stores the complete path history
        
        Returns:
            cp.ndarray: Array of simulated paths with shape (n_paths, n_steps+1, 3)
                       where the last dimension represents [S, v, r]
        
        Notes:
            - All computations are performed on the GPU
            - The implementation uses CuPy's vectorized operations
            - Non-negativity is enforced for rates and variance
            - The correlation structure is preserved through Cholesky decomposition
            - The time grid includes the initial state (n_steps+1 points)
        """
        # Unpack model parameters
        rp = self.model.params.rate
        ep = self.model.params.equity
        L_cp = cp.asarray(cholesky_correlation(self.model.params.corr_matrix))

        # Simulation parameters
        N, M = self.params.n_paths, self.params.n_steps
        dt = self.params.dt
        sqrt_dt = cp.sqrt(dt)

        # Initialize state vectors on GPU
        S = cp.full(N, ep.S0, dtype=float)
        v = cp.full(N, ep.v0, dtype=float)
        r = cp.full(N, rp.r0, dtype=float)

        # Initialize path array on GPU
        paths = cp.empty((N, M+1, 3), dtype=float)
        paths[:, 0, :] = cp.stack([S, v, r], axis=1)

        # Time stepping
        for t in range(1, M+1):
            # Generate correlated random increments on GPU
            Z = self.rng.standard_normal((N, 3))
            dW = Z @ L_cp.T * sqrt_dt

            # Update processes using Euler-Maruyama scheme
            r = r + rp.kappa*(rp.theta - r)*dt + rp.sigma*cp.sqrt(cp.maximum(r,0))*dW[:,0]
            v = cp.maximum(v + ep.kappa_v*(ep.theta_v - v)*dt + ep.sigma_v*cp.sqrt(cp.maximum(v,0))*dW[:,1], 0)
            S = S + (r - ep.q)*S*dt + cp.sqrt(cp.maximum(v,0))*S*dW[:,2]

            # Store current state
            paths[:, t, :] = cp.stack([S, v, r], axis=1)

        return paths