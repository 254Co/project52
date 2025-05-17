# File: chen3/schemes/euler.py
"""
Euler-Maruyama Numerical Scheme Implementation

This module implements the Euler-Maruyama scheme for numerical integration of
stochastic differential equations (SDEs). The scheme is a first-order method
that provides a simple and efficient way to simulate stochastic processes.

The Euler-Maruyama scheme approximates the solution of the SDE:
    dX_t = μ(X_t)dt + σ(X_t)dW_t

using the discrete-time approximation:
    X_{t+Δt} = X_t + μ(X_t)Δt + σ(X_t)√Δt Z_t

where Z_t is a standard normal random variable.
"""

from typing import Callable, Optional, Tuple, Union
import numpy as np
from numba import njit, float64, types
from numba.experimental import jitclass

@njit
def _euler_step(state: float64, z: float64, dt: float64, drift: float64, diffusion: float64) -> float64:
    """
    Compute one step of the Euler-Maruyama scheme.

    Args:
        state: Current state of the process
        z: Standard normal random variable
        dt: Time step size
        drift: Drift term value
        diffusion: Diffusion term value

    Returns:
        Next state according to the Euler-Maruyama scheme
    """
    return state + drift * dt + diffusion * np.sqrt(dt) * z

def euler_scheme(drift_fn: Callable, diffusion_fn: Callable, dt: float):
    """
    Create an Euler-Maruyama step function for SDE integration.

    This function returns a Numba-compiled step function that implements the
    Euler-Maruyama scheme for numerical integration of stochastic differential
    equations. The returned function is optimized for performance using Numba's
    just-in-time compilation.

    Args:
        drift_fn (callable): Function computing the drift term μ(X_t)
                            Must accept a state array and return drift values
        diffusion_fn (callable): Function computing the diffusion term σ(X_t)
                                Must accept a state array and return diffusion values
        dt (float): Time step size for the numerical scheme (dt > 0)

    Returns:
        callable: A Numba-compiled step function that takes (state, z) as arguments
                 and returns the next state according to the Euler-Maruyama scheme

    Example:
        >>> def drift(state): return 0.1 * state
        >>> def diffusion(state): return 0.2 * state
        >>> step = euler_scheme(drift, diffusion, dt=0.01)
        >>> next_state = step(current_state, random_normal)

    Notes:
        - The returned function is Numba-compiled for performance
        - The scheme is first-order accurate in the strong sense
        - The scheme is first-order accurate in the weak sense
        - The scheme may not preserve positivity of the solution
    """
    @njit
    def step(state: float64, z: float64) -> float64:
        """
        Compute one step of the Euler-Maruyama scheme.

        Args:
            state: Current state of the process
            z: Standard normal random variable

        Returns:
            Next state according to the Euler-Maruyama scheme
        """
        drift = drift_fn(state)
        diffusion = diffusion_fn(state)
        return _euler_step(state, z, dt, drift, diffusion)

    return step


class EulerMaruyama:
    """
    Euler-Maruyama scheme for numerical integration of SDEs.

    This class implements the Euler-Maruyama scheme for numerical integration
    of stochastic differential equations. It provides methods for simulating
    paths of the solution.

    Attributes:
        mu (float): Drift coefficient
        sigma (float): Diffusion coefficient
    """

    def __init__(self, mu: Callable, sigma: Callable):
        """
        Initialize the Euler-Maruyama scheme.

        Args:
            mu: Drift coefficient function
            sigma: Diffusion coefficient function

        Raises:
            ValueError: If sigma is negative
        """
        self.mu = mu
        self.sigma = sigma

        if sigma(0) < 0:
            raise ValueError("Diffusion coefficient must be non-negative")

    def simulate(self, S0: float, dt: float, n_steps: int, n_paths: int) -> np.ndarray:
        """
        Simulate paths using the Euler-Maruyama scheme.

        Args:
            S0: Initial value
            dt: Time step size
            n_steps: Number of time steps
            n_paths: Number of paths to simulate

        Returns:
            Array of simulated paths
        """
        # Initialize paths
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0

        # Generate random increments
        dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))

        # Simulate paths
        for i in range(n_steps):
            t = i * dt
            paths[:, i + 1] = paths[:, i] + self.mu(t) * dt + self.sigma(t) * dW[:, i]

        return paths
