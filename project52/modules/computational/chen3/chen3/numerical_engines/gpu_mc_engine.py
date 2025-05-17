# -------------------- chen3/numerical_engines/gpu_mc_engine.py --------------------
"""
GPU-accelerated Monte Carlo Simulation Engine for the Chen3 Model

This module implements a GPU-accelerated version of the Monte Carlo simulation engine
for the Chen3 model, providing efficient parallel path generation and pricing capabilities.
"""

from typing import Callable, Dict, Optional, Tuple

import cupy as cp
import numpy as np

from ..utils.exceptions import NumericalError
from ..utils.logging_config import logger
from .integration import IntegrationScheme
from .monte_carlo import MonteCarloEngineBase
from .path_generator import PathGenerator
from .random import RandomNumberGenerator
from .variance_reduction import VarianceReductionTechnique


class GPUMonteCarloEngine(MonteCarloEngineBase):
    """
    GPU-accelerated Monte Carlo simulation engine for the Chen3 model.

    This class provides GPU-accelerated functionality for Monte Carlo simulation,
    including parallel path generation, payoff evaluation, and statistical analysis.
    """

    def __init__(
        self,
        rng: Optional[RandomNumberGenerator] = None,
        path_generator: Optional[PathGenerator] = None,
        integration_scheme: Optional[IntegrationScheme] = None,
        n_paths: int = 10000,
        n_steps: int = 100,
        dt: float = 0.01,
        use_antithetic: bool = True,
        use_control_variate: bool = True,
        block_size: int = 256,
        variance_reduction: Optional[VarianceReductionTechnique] = None,
    ):
        """Initialize the GPU Monte Carlo engine."""
        super().__init__(
            rng=rng,
            path_generator=path_generator,
            integration_scheme=integration_scheme,
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            use_antithetic=use_antithetic,
            use_control_variate=use_control_variate,
            variance_reduction=variance_reduction,
        )
        self.block_size = block_size

        # Initialize CUDA stream
        self.stream = cp.cuda.Stream()

        # Load CUDA kernels
        self._load_kernels()

    def _load_kernels(self):
        """Load and compile CUDA kernels."""
        try:
            # Path simulation kernel
            self.simulate_paths_kernel = cp.RawKernel(
                r"""
            extern "C" __global__
            void simulate_paths(
                float* rate_paths,
                float* equity_paths,
                float* variance_paths,
                const float* dW,
                const float* initial_state,
                const float dt,
                const int n_steps
            ) {
                int tid = blockDim.x * blockIdx.x + threadIdx.x;
                int stride = blockDim.x * gridDim.x;

                for (int i = tid; i < n_steps; i += stride) {
                    // Initialize paths
                    rate_paths[i] = initial_state[0];
                    equity_paths[i] = initial_state[1];
                    variance_paths[i] = initial_state[2];

                    // Simulate paths
                    for (int t = 0; t < n_steps; t++) {
                        // Update paths using Euler-Maruyama scheme
                        rate_paths[i] += drift_r * dt + diffusion_r * dW[i * n_steps + t];
                        equity_paths[i] *= exp((drift_S - 0.5 * diffusion_S * diffusion_S) * dt
                                             + diffusion_S * dW[i * n_steps + t]);
                        variance_paths[i] = max(variance_paths[i] + drift_v * dt
                                             + diffusion_v * dW[i * n_steps + t], 0.0f);
                    }
                }
            }
            """,
                "simulate_paths",
            )

            # Payoff computation kernel
            self.compute_payoff_kernel = cp.RawKernel(
                r"""
            extern "C" __global__
            void compute_payoff(
                float* payoffs,
                const float* rate_paths,
                const float* equity_paths,
                const float* variance_paths,
                const int n_paths
            ) {
                int tid = blockDim.x * blockIdx.x + threadIdx.x;
                int stride = blockDim.x * gridDim.x;

                for (int i = tid; i < n_paths; i += stride) {
                    // Compute payoff for each path
                    payoffs[i] = payoff_function(
                        rate_paths[i],
                        equity_paths[i],
                        variance_paths[i]
                    );
                }
            }
            """,
                "compute_payoff",
            )

            # Variance reduction kernel
            self.variance_reduction_kernel = cp.RawKernel(
                r"""
            extern "C" __global__
            void apply_variance_reduction(
                float* adjusted_payoffs,
                const float* payoffs,
                const float* control_values,
                const float control_expectation,
                const float beta,
                const int n_paths
            ) {
                int tid = blockDim.x * blockIdx.x + threadIdx.x;
                int stride = blockDim.x * gridDim.x;

                for (int i = tid; i < n_paths; i += stride) {
                    adjusted_payoffs[i] = payoffs[i] - beta * (control_values[i] - control_expectation);
                }
            }
            """,
                "apply_variance_reduction",
            )

        except Exception as e:
            raise NumericalError(f"Failed to load CUDA kernels: {str(e)}")

    def simulate_paths(
        self,
        initial_state: Dict[str, float],
        drift_function: Callable,
        diffusion_function: Callable,
        correlation_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate paths for the three-factor model using GPU.

        Args:
            initial_state: Initial state variables
            drift_function: Function computing drift terms
            diffusion_function: Function computing diffusion terms
            correlation_matrix: Correlation matrix between factors

        Returns:
            Tuple of (rate_paths, equity_paths, variance_paths)

        Raises:
            NumericalError: If simulation fails
        """
        try:
            # Generate random numbers on CPU and transfer to GPU
            n_total = self.n_paths * (2 if self.use_antithetic else 1)
            dW = self.rng.generate_normal(
                size=(n_total, self.n_steps, 3), correlation_matrix=correlation_matrix
            )
            dW_gpu = cp.asarray(dW)

            # Initialize GPU arrays
            rate_paths_gpu = cp.zeros((n_total, self.n_steps + 1))
            equity_paths_gpu = cp.zeros((n_total, self.n_steps + 1))
            variance_paths_gpu = cp.zeros((n_total, self.n_steps + 1))

            # Set initial values
            rate_paths_gpu[:, 0] = initial_state["r"]
            equity_paths_gpu[:, 0] = initial_state["S"]
            variance_paths_gpu[:, 0] = initial_state["v"]

            # Compute grid and block dimensions
            grid_size = (n_total + self.block_size - 1) // self.block_size

            # Launch kernel
            self.simulate_paths_kernel(
                (grid_size,),
                (self.block_size,),
                (
                    rate_paths_gpu,
                    equity_paths_gpu,
                    variance_paths_gpu,
                    dW_gpu,
                    cp.array(
                        [initial_state["r"], initial_state["S"], initial_state["v"]]
                    ),
                    self.dt,
                    self.n_steps,
                ),
            )

            # Apply antithetic variates if enabled
            if self.use_antithetic:
                rate_paths_gpu = (
                    rate_paths_gpu[: self.n_paths] + rate_paths_gpu[self.n_paths :]
                ) / 2
                equity_paths_gpu = (
                    equity_paths_gpu[: self.n_paths] + equity_paths_gpu[self.n_paths :]
                ) / 2
                variance_paths_gpu = (
                    variance_paths_gpu[: self.n_paths]
                    + variance_paths_gpu[self.n_paths :]
                ) / 2

            # Transfer results back to CPU
            rate_paths = cp.asnumpy(rate_paths_gpu)
            equity_paths = cp.asnumpy(equity_paths_gpu)
            variance_paths = cp.asnumpy(variance_paths_gpu)

            return rate_paths, equity_paths, variance_paths

        except Exception as e:
            raise NumericalError(f"GPU path simulation failed: {str(e)}")

    def compute_expectation(
        self,
        payoff_function: Callable,
        paths: Tuple[np.ndarray, ...],
        control_variate: Optional[Callable] = None,
    ) -> Tuple[float, float]:
        """
        Compute expectation of a payoff function using GPU.

        Args:
            payoff_function: Function computing payoff
            paths: Simulated paths
            control_variate: Optional control variate function

        Returns:
            Tuple of (price, standard_error)
        """
        try:
            # Transfer paths to GPU
            rate_paths_gpu = cp.asarray(paths[0])
            equity_paths_gpu = cp.asarray(paths[1])
            variance_paths_gpu = cp.asarray(paths[2])

            # Initialize GPU array for payoffs
            payoffs_gpu = cp.zeros(self.n_paths)

            # Compute grid and block dimensions
            grid_size = (self.n_paths + self.block_size - 1) // self.block_size

            # Launch kernel
            self.compute_payoff_kernel(
                (grid_size,),
                (self.block_size,),
                (
                    payoffs_gpu,
                    rate_paths_gpu,
                    equity_paths_gpu,
                    variance_paths_gpu,
                    self.n_paths,
                ),
            )

            # Apply variance reduction if enabled
            if self.variance_reduction is not None:
                # Compute control variate values if provided
                if control_variate is not None:
                    control_values_gpu = cp.asarray(control_variate(*paths))
                    control_expectation = float(cp.mean(control_values_gpu))

                    # Compute optimal coefficient
                    cov = cp.cov(payoffs_gpu, control_values_gpu)[0, 1]
                    var_cv = cp.var(control_values_gpu)
                    beta = float(cov / var_cv if var_cv > 0 else 0.0)

                    # Apply variance reduction
                    adjusted_payoffs_gpu = cp.zeros_like(payoffs_gpu)
                    self.variance_reduction_kernel(
                        (grid_size,),
                        (self.block_size,),
                        (
                            adjusted_payoffs_gpu,
                            payoffs_gpu,
                            control_values_gpu,
                            control_expectation,
                            beta,
                            self.n_paths,
                        ),
                    )
                    payoffs_gpu = adjusted_payoffs_gpu

            # Compute statistics on GPU
            price = float(cp.mean(payoffs_gpu))
            std_error = float(cp.std(payoffs_gpu) / cp.sqrt(self.n_paths))

            return price, std_error

        except Exception as e:
            raise NumericalError(f"GPU expectation computation failed: {str(e)}")

    def __str__(self) -> str:
        """String representation of the GPU Monte Carlo engine."""
        return (
            f"GPUMonteCarloEngine(n_paths={self.n_paths}, n_steps={self.n_steps}, "
            f"dt={self.dt}, use_antithetic={self.use_antithetic}, "
            f"use_control_variate={self.use_control_variate}, "
            f"block_size={self.block_size})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the GPU Monte Carlo engine."""
        return self.__str__()
