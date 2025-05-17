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

from typing import Dict, Optional, Tuple

import cupy as cp

from chen3.correlation import cholesky_correlation

from .core import PathGenerator


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
        current_step: Current simulation step
        total_steps: Total number of steps
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
        self.current_step = 0
        self.total_steps = settings.n_steps

    def generate(self) -> cp.ndarray:
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
        paths = cp.empty((N, M + 1, 3), dtype=float)
        paths[:, 0, :] = cp.stack([S, v, r], axis=1)

        # Time stepping
        for t in range(1, M + 1):
            self.current_step = t

            # Generate correlated random increments on GPU
            Z = self.rng.standard_normal((N, 3))
            dW = Z @ L_cp.T * sqrt_dt

            # Update processes using Euler-Maruyama scheme
            r = (
                r
                + rp.kappa * (rp.theta - r) * dt
                + rp.sigma * cp.sqrt(cp.maximum(r, 0)) * dW[:, 0]
            )
            v = cp.maximum(
                v
                + ep.kappa_v * (ep.theta_v - v) * dt
                + ep.sigma_v * cp.sqrt(cp.maximum(v, 0)) * dW[:, 1],
                0,
            )
            S = S + (r - ep.q) * S * dt + cp.sqrt(cp.maximum(v, 0)) * S * dW[:, 2]

            # Store current state
            paths[:, t, :] = cp.stack([S, v, r], axis=1)

        return paths

    def get_state(self) -> Dict[str, cp.ndarray]:
        """
        Get current simulation state.

        Returns:
            Dict[str, cp.ndarray]: Current state including:
                - paths: Current path array
                - S: Current stock prices
                - v: Current variance values
                - r: Current interest rates
                - rng_state: Random number generator state
        """
        return {
            "paths": self.paths,
            "S": self.S,
            "v": self.v,
            "r": self.r,
            "rng_state": self.rng.get_state(),
        }

    def resume_from_state(self, state: Dict[str, cp.ndarray]) -> cp.ndarray:
        """
        Resume simulation from a saved state.

        Args:
            state (Dict[str, cp.ndarray]): Saved state

        Returns:
            cp.ndarray: Completed paths

        Raises:
            ValueError: If state is invalid
        """
        # Validate state
        required_keys = {"paths", "S", "v", "r", "rng_state"}
        if not all(key in state for key in required_keys):
            raise ValueError("Invalid state: missing required keys")

        # Restore state
        self.paths = state["paths"]
        self.S = state["S"]
        self.v = state["v"]
        self.r = state["r"]
        self.rng.set_state(state["rng_state"])

        # Continue simulation
        return self.generate()

    def get_progress(self) -> float:
        """
        Get simulation progress.

        Returns:
            float: Progress as a fraction between 0 and 1
        """
        return self.current_step / self.total_steps if self.total_steps > 0 else 0.0
