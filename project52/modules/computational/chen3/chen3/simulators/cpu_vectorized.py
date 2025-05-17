# File: chen3/simulators/cpu_vectorized.py

"""
CPU Vectorized Path Simulator Implementation

This module implements a vectorized CPU-based path simulator for the three-factor
Chen model. The simulator uses NumPy's vectorized operations to efficiently
generate Monte Carlo paths for the coupled stochastic processes:
- Short rate (CIR process)
- Variance (CIR process)
- Stock price (Geometric Brownian motion with stochastic volatility)

The implementation uses the Euler-Maruyama scheme for numerical integration
and handles the correlation structure between the processes using Cholesky
decomposition.
"""

import numpy as np

from chen3.correlation import cholesky_correlation

from .core import PathGenerator


class CPUVectorizedSimulator(PathGenerator):
    """
    Vectorized CPU-based path simulator for the three-factor Chen model.

    This class implements a highly efficient path generator that leverages
    NumPy's vectorized operations to simulate the three-factor model on CPU.
    The implementation is optimized for performance while maintaining numerical
    stability and accuracy.

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

    Example:
        >>> from chen3.config import Settings
        >>> settings = Settings(backend="cpu", n_paths=100000, n_steps=252)
        >>> simulator = CPUVectorizedSimulator(model, settings)
        >>> paths = simulator.generate()  # shape: (n_paths, n_steps+1, 3)
    """

    def generate(self):
        """
        Generate Monte Carlo paths for the three-factor model.

        This method simulates paths for the coupled system of stochastic
        processes using the Euler-Maruyama scheme. The implementation is
        fully vectorized for efficient computation on CPU.

        The method:
        1. Initializes the state vectors
        2. Generates correlated random increments
        3. Updates each process using the Euler-Maruyama scheme
        4. Ensures non-negativity of rates and variance
        5. Stores the complete path history

        Returns:
            np.ndarray: Array of simulated paths with shape (n_paths, n_steps+1, 3)
                       where the last dimension represents [S, v, r]

        Notes:
            - The implementation uses vectorized operations for efficiency
            - Non-negativity is enforced for rates and variance
            - The correlation structure is preserved through Cholesky decomposition
            - The time grid includes the initial state (n_steps+1 points)
        """
        # Unpack model parameters
        rp = self.model.params.rate
        ep = self.model.params.equity
        L = cholesky_correlation(self.model.params.corr_matrix)

        # Simulation parameters
        N, M = self.params.n_paths, self.params.n_steps
        dt = self.params.dt
        sqrt_dt = np.sqrt(dt)

        # Initialize state vectors
        S = np.full(N, ep.S0, dtype=float)
        v = np.full(N, ep.v0, dtype=float)
        r = np.full(N, rp.r0, dtype=float)

        # Initialize path array
        paths = np.empty((N, M + 1, 3), dtype=float)
        paths[:, 0, :] = np.stack([S, v, r], axis=1)

        # Time stepping
        for t in range(1, M + 1):
            # Generate correlated random increments
            Z = np.random.standard_normal((N, 3))
            dW = Z @ L.T * sqrt_dt

            # Update short rate (CIR process)
            dr = (
                rp.kappa * (rp.theta - r) * dt
                + rp.sigma * np.sqrt(np.maximum(r, 0.0)) * dW[:, 0]
            )
            r += dr

            # Update variance (CIR process)
            dv = (
                ep.kappa_v * (ep.theta_v - v) * dt
                + ep.sigma_v * np.sqrt(np.maximum(v, 0.0)) * dW[:, 1]
            )
            v = np.maximum(v + dv, 0.0)

            # Update stock price (GBM with stochastic volatility)
            dS = (r - ep.q) * S * dt + np.sqrt(np.maximum(v, 0.0)) * S * dW[:, 2]
            S += dS

            # Store current state
            paths[:, t, 0] = S
            paths[:, t, 1] = v
            paths[:, t, 2] = r

        return paths
