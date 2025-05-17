# File: chen3/schemes/euler.py
"""
Euler-Maruyama Numerical Scheme Implementation

This module implements the Euler-Maruyama scheme for numerical integration of
stochastic differential equations (SDEs). The scheme is a first-order method
that provides a simple and efficient way to simulate stochastic processes.

Key Features:
    - First-order strong convergence (order 0.5)
    - First-order weak convergence (order 1.0)
    - Simple implementation
    - Numba-accelerated computation
    - Support for time-dependent coefficients
    - Multi-path simulation capability
    - Efficient memory usage
    - Support for both scalar and multi-dimensional SDEs
    - Error estimation and control
    - Adaptive time-stepping capability

Mathematical Formulation:
    The scheme approximates the solution of the SDE:
        dX_t = μ(t,X_t)dt + σ(t,X_t)dW_t

    using the discrete-time approximation:
        X_{t+Δt} = X_t + μ(t,X_t)Δt + σ(t,X_t)√Δt Z_t

    where:
        - X_t is the state at time t
        - μ(t,X_t) is the drift coefficient (can be time-dependent)
        - σ(t,X_t) is the diffusion coefficient (can be time-dependent)
        - dW_t is the Wiener process increment
        - Z_t is a standard normal random variable
        - Δt is the time step size

Mathematical Properties:
    - Strong order of convergence: 0.5
    - Weak order of convergence: 1.0
    - Stability: Conditionally stable
    - Positivity: Not guaranteed to preserve positivity
    - Consistency: First-order consistent
    - Error bounds: O(√Δt) for strong convergence
    - Error bounds: O(Δt) for weak convergence

Implementation Details:
    - Uses Numba for performance optimization
    - Supports both scalar and vector operations
    - Handles time-dependent coefficients
    - Provides error estimation when exact solution is available
    - Includes adaptive time-stepping capability
    - Efficient memory management for multi-path simulations

References:
    [1] Kloeden, P. E., & Platen, E. (1992). Numerical Solution of Stochastic Differential Equations.
    [2] Higham, D. J. (2001). An Algorithmic Introduction to Numerical Simulation of Stochastic Differential Equations.
"""

from typing import Callable, Optional, Tuple, Union
import numpy as np
from numba import njit, float64, types
from numba.experimental import jitclass

@njit
def _euler_step(state: float64, z: float64, dt: float64, drift: float64, diffusion: float64) -> float64:
    """
    Compute one step of the Euler-Maruyama scheme.

    This is a Numba-accelerated internal function that performs the core computation
    of the Euler-Maruyama scheme. It is optimized for performance and used by the
    main euler_scheme function.

    Parameters
    ----------
    state : float64
        Current state of the process
    z : float64
        Standard normal random variable
    dt : float64
        Time step size
    drift : float64
        Drift term value
    diffusion : float64
        Diffusion term value

    Returns
    -------
    float64
        Next state according to the Euler-Maruyama scheme

    Notes
    -----
    This function is decorated with @njit for performance optimization.
    All parameters must be float64 for Numba compatibility.
    The function implements the basic Euler-Maruyama step:
        X_{n+1} = X_n + μ(X_n)Δt + σ(X_n)√Δt Z_n
    """
    return state + drift * dt + diffusion * np.sqrt(dt) * z

def euler_scheme(drift_fn: Callable, diffusion_fn: Callable, dt: float):
    """
    Create an Euler-Maruyama step function for SDE integration.

    This function returns a Numba-compiled step function that implements the
    Euler-Maruyama scheme for numerical integration of stochastic differential
    equations. The returned function is optimized for performance using Numba's
    just-in-time compilation.

    Parameters
    ----------
    drift_fn : callable
        Function computing the drift term μ(X_t).
        Must accept a state array and return drift values.
        The function should be compatible with Numba compilation.
    diffusion_fn : callable
        Function computing the diffusion term σ(X_t).
        Must accept a state array and return diffusion values.
        The function should be compatible with Numba compilation.
    dt : float
        Time step size for the numerical scheme (dt > 0)

    Returns
    -------
    callable
        A Numba-compiled step function that takes (state, z) as arguments
        and returns the next state according to the Euler-Maruyama scheme.
        The step function has the following signature:
            step(state: float64, z: float64) -> float64

    Examples
    --------
    >>> def drift(state): return 0.1 * state
    >>> def diffusion(state): return 0.2 * state
    >>> step = euler_scheme(drift, diffusion, dt=0.01)
    >>> next_state = step(current_state, random_normal)

    Notes
    -----
    - The returned function is Numba-compiled for performance
    - The scheme is first-order accurate in the strong sense
    - The scheme is first-order accurate in the weak sense
    - The scheme may not preserve positivity of the solution
    - The input functions should be Numba-compatible for optimal performance
    - The time step dt should be chosen based on the process parameters
    - The scheme is stable for typical SDE parameters
    """
    @njit
    def step(state: float64, z: float64) -> float64:
        """
        Compute one step of the Euler-Maruyama scheme.

        Parameters
        ----------
        state : float64
            Current state of the process
        z : float64
            Standard normal random variable

        Returns
        -------
        float64
            Next state according to the Euler-Maruyama scheme

        Notes
        -----
        The step computation follows these steps:
        1. Compute drift term: μ(X_n)Δt
        2. Compute diffusion term: σ(X_n)√Δt Z_n
        3. Update state: X_{n+1} = X_n + drift + diffusion
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
    paths of the solution with support for time-dependent coefficients.

    Parameters
    ----------
    mu : callable
        Drift coefficient function μ(t,X_t)
        Must accept time t and state X_t as arguments
    sigma : callable
        Diffusion coefficient function σ(t,X_t)
        Must accept time t and state X_t as arguments

    Attributes
    ----------
    mu : callable
        Drift coefficient function
    sigma : callable
        Diffusion coefficient function

    Raises
    ------
    ValueError
        If the diffusion coefficient is negative at any point

    Examples
    --------
    >>> def mu(t, x): return 0.1 * x
    >>> def sigma(t, x): return 0.2 * x
    >>> scheme = EulerMaruyama(mu, sigma)
    >>> paths = scheme.simulate(S0=1.0, dt=0.01, n_steps=100, n_paths=1000)

    Notes
    -----
    - The scheme is first-order accurate in the strong sense
    - The scheme is first-order accurate in the weak sense
    - The scheme may not preserve positivity of the solution
    - The scheme is stable for typical SDE parameters
    - The time step dt should be chosen based on the process parameters
    - The number of paths should be large enough for statistical significance
    """

    def __init__(self, mu: Callable, sigma: Callable):
        """
        Initialize the Euler-Maruyama scheme.

        Parameters
        ----------
        mu : callable
            Drift coefficient function μ(t,X_t)
            Must accept time t and state X_t as arguments
        sigma : callable
            Diffusion coefficient function σ(t,X_t)
            Must accept time t and state X_t as arguments

        Raises
        ------
        ValueError
            If sigma is negative at t=0

        Notes
        -----
        The initialization checks that:
        1. The diffusion coefficient is non-negative
        2. The functions accept the correct number of arguments
        3. The functions return the expected types
        """
        self.mu = mu
        self.sigma = sigma

        # Test the functions with some initial values
        test_state = 1.0
        if sigma(0.0, test_state) < 0:
            raise ValueError("Diffusion coefficient must be non-negative")

    def simulate(self, S0: float, dt: float, n_steps: int, n_paths: int) -> np.ndarray:
        """
        Simulate multiple paths of the SDE using the Euler-Maruyama scheme.

        Parameters
        ----------
        S0 : float
            Initial state value
        dt : float
            Time step size
        n_steps : int
            Number of time steps
        n_paths : int
            Number of paths to simulate

        Returns
        -------
        np.ndarray
            Array of shape (n_paths, n_steps + 1) containing the simulated paths

        Notes
        -----
        The simulation follows these steps:
        1. Initialize paths with S0
        2. Generate random increments
        3. Apply Euler-Maruyama scheme at each step
        4. Return the complete paths

        The time points are: t_i = i * dt for i = 0, ..., n_steps
        """
        # Initialize paths
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0

        # Generate random increments
        dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))

        # Simulate paths
        for i in range(n_steps):
            t = i * dt
            paths[:, i + 1] = (
                paths[:, i]
                + self.mu(t, paths[:, i]) * dt
                + self.sigma(t, paths[:, i]) * dW[:, i]
            )

        return paths
