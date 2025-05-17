"""
Milstein scheme implementation for stochastic differential equations.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np


@dataclass
class MilsteinResult:
    """Container for Milstein scheme simulation results."""

    paths: np.ndarray
    times: np.ndarray
    errors: Optional[np.ndarray] = None
    strong_error: Optional[float] = None
    weak_error: Optional[float] = None


def milstein_step(
    x: float,
    t: float,
    dt: float,
    drift: Callable,
    diffusion: Callable,
    diffusion_derivative: Callable,
    dW: float,
) -> float:
    """
    Single step of the Milstein scheme.

    Parameters
    ----------
    x : float
        Current state
    t : float
        Current time
    dt : float
        Time step
    drift : Callable
        Drift function f(t,x)
    diffusion : Callable
        Diffusion function g(t,x)
    diffusion_derivative : Callable
        Derivative of diffusion function g'(t,x)
    dW : float
        Brownian increment

    Returns
    -------
    float
        Next state
    """
    a = drift(t, x)
    b = diffusion(t, x)
    b_prime = diffusion_derivative(t, x)
    return x + a * dt + b * dW + 0.5 * b * b_prime * (dW**2 - dt)


def milstein_scheme(
    x0: float,
    t0: float,
    T: float,
    n_steps: int,
    drift: Callable,
    diffusion: Callable,
    diffusion_derivative: Callable,
    rng: np.random.Generator,
    exact_solution: Optional[Callable] = None,
) -> MilsteinResult:
    """
    Milstein scheme for SDE integration.

    Parameters
    ----------
    x0 : float
        Initial state
    t0 : float
        Initial time
    T : float
        Final time
    n_steps : int
        Number of time steps
    drift : Callable
        Drift function f(t,x)
    diffusion : Callable
        Diffusion function g(t,x)
    diffusion_derivative : Callable
        Derivative of diffusion function g'(t,x)
    rng : np.random.Generator
        Random number generator
    exact_solution : Optional[Callable], optional
        Exact solution for error computation, by default None

    Returns
    -------
    MilsteinResult
        Container with simulation results
    """
    dt = (T - t0) / n_steps
    times = np.linspace(t0, T, n_steps + 1)
    paths = np.zeros(n_steps + 1)
    paths[0] = x0

    for i in range(n_steps):
        dW = rng.normal(0, np.sqrt(dt))
        paths[i + 1] = milstein_step(
            paths[i], times[i], dt, drift, diffusion, diffusion_derivative, dW
        )

    result = MilsteinResult(paths=paths, times=times)

    if exact_solution is not None:
        exact_paths = exact_solution(times)
        errors = np.abs(paths - exact_paths)
        result.errors = errors
        result.strong_error = np.sqrt(np.mean(errors**2))
        result.weak_error = np.abs(np.mean(paths) - np.mean(exact_paths))

    return result


def multi_dimensional_milstein(
    x0: np.ndarray,
    t0: float,
    T: float,
    n_steps: int,
    drift: Callable,
    diffusion: Callable,
    diffusion_jacobian: Callable,
    rng: np.random.Generator,
    exact_solution: Optional[Callable] = None,
) -> MilsteinResult:
    """
    Multi-dimensional Milstein scheme for SDE integration.

    Parameters
    ----------
    x0 : np.ndarray
        Initial state vector
    t0 : float
        Initial time
    T : float
        Final time
    n_steps : int
        Number of time steps
    drift : Callable
        Drift function f(t,x)
    diffusion : Callable
        Diffusion function g(t,x)
    diffusion_jacobian : Callable
        Jacobian of diffusion function
    rng : np.random.Generator
        Random number generator
    exact_solution : Optional[Callable], optional
        Exact solution for error computation, by default None

    Returns
    -------
    MilsteinResult
        Container with simulation results
    """
    dt = (T - t0) / n_steps
    times = np.linspace(t0, T, n_steps + 1)
    n_dims = len(x0)
    paths = np.zeros((n_steps + 1, n_dims))
    paths[0] = x0

    for i in range(n_steps):
        dW = rng.normal(0, np.sqrt(dt), size=n_dims)
        a = drift(times[i], paths[i])
        b = diffusion(times[i], paths[i])
        b_jac = diffusion_jacobian(times[i], paths[i])

        # Milstein correction term
        correction = np.zeros(n_dims)
        for j in range(n_dims):
            for k in range(n_dims):
                correction[j] += 0.5 * b[j, k] * np.sum(b_jac[j, :, k] * dW)

        paths[i + 1] = paths[i] + a * dt + b @ dW + correction

    result = MilsteinResult(paths=paths, times=times)

    if exact_solution is not None:
        exact_paths = exact_solution(times)
        errors = np.abs(paths - exact_paths)
        result.errors = errors
        result.strong_error = np.sqrt(np.mean(errors**2))
        result.weak_error = np.abs(
            np.mean(paths, axis=0) - np.mean(exact_paths, axis=0)
        )

    return result


def adaptive_milstein(
    x0: Union[float, np.ndarray],
    t0: float,
    T: float,
    drift: Callable,
    diffusion: Callable,
    diffusion_derivative: Callable,
    rng: np.random.Generator,
    tol: float = 1e-6,
    max_steps: int = 1000,
    min_steps: int = 10,
    exact_solution: Optional[Callable] = None,
) -> MilsteinResult:
    """
    Adaptive Milstein scheme with error control.

    Parameters
    ----------
    x0 : Union[float, np.ndarray]
        Initial state
    t0 : float
        Initial time
    T : float
        Final time
    drift : Callable
        Drift function f(t,x)
    diffusion : Callable
        Diffusion function g(t,x)
    diffusion_derivative : Callable
        Derivative of diffusion function g'(t,x)
    rng : np.random.Generator
        Random number generator
    tol : float, optional
        Error tolerance, by default 1e-6
    max_steps : int, optional
        Maximum number of steps, by default 1000
    min_steps : int, optional
        Minimum number of steps, by default 10
    exact_solution : Optional[Callable], optional
        Exact solution for error computation, by default None

    Returns
    -------
    MilsteinResult
        Container with simulation results
    """
    n_steps = min_steps
    while n_steps <= max_steps:
        if isinstance(x0, np.ndarray):
            result = multi_dimensional_milstein(
                x0,
                t0,
                T,
                n_steps,
                drift,
                diffusion,
                diffusion_derivative,
                rng,
                exact_solution,
            )
        else:
            result = milstein_scheme(
                x0,
                t0,
                T,
                n_steps,
                drift,
                diffusion,
                diffusion_derivative,
                rng,
                exact_solution,
            )

        if result.strong_error is not None and result.strong_error <= tol:
            break

        n_steps *= 2

    return result


class Milstein:
    """
    Milstein scheme for numerical integration of SDEs.

    This class implements the Milstein scheme for simulating stochastic
    processes. It provides a simple interface for simulating paths of various
    stochastic differential equations.

    Example:
        >>> scheme = Milstein(kappa=2.0, theta=0.04, sigma=0.3)
        >>> paths = scheme.simulate(v0=0.04, dt=0.01, n_steps=100, n_paths=1000)
    """

    def __init__(self, kappa: float, theta: float, sigma: float):
        """
        Initialize the Milstein scheme.

        Args:
            kappa: Mean reversion speed
            theta: Long-term mean
            sigma: Volatility
        """
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma

    def simulate(self, v0: float, dt: float, n_steps: int, n_paths: int) -> np.ndarray:
        """
        Simulate paths using the Milstein scheme.

        Args:
            v0: Initial value
            dt: Time step size
            n_steps: Number of time steps
            n_paths: Number of paths to simulate

        Returns:
            np.ndarray: Simulated paths (n_paths x (n_steps + 1))
        """
        # Initialize paths
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = v0

        # Generate random increments
        dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))

        # Simulate paths
        for i in range(n_steps):
            # Drift term
            drift = self.kappa * (self.theta - paths[:, i])

            # Diffusion term
            diffusion = self.sigma * np.sqrt(paths[:, i])

            # Milstein correction term
            correction = 0.25 * self.sigma**2 * (dW[:, i] ** 2 - dt)

            # Update paths
            paths[:, i + 1] = (
                paths[:, i] + drift * dt + diffusion * dW[:, i] + correction
            )

            # Ensure positivity
            paths[:, i + 1] = np.maximum(paths[:, i + 1], 0)

        return paths
