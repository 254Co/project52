"""
Platen's explicit order 2.0 scheme for stochastic differential equations.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np


@dataclass
class PlatenResult:
    """Container for Platen scheme simulation results."""

    paths: np.ndarray
    times: np.ndarray
    errors: Optional[np.ndarray] = None
    strong_error: Optional[float] = None
    weak_error: Optional[float] = None
    convergence_rate: Optional[float] = None


def platen_step(
    x: float,
    t: float,
    dt: float,
    drift: Callable,
    diffusion: Callable,
    dW: float,
    dZ: float,
) -> float:
    """
    Single step of Platen's explicit order 2.0 scheme.

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
    dW : float
        Brownian increment
    dZ : float
        Double integral of Brownian motion

    Returns
    -------
    float
        Next state
    """
    a = drift(t, x)
    b = diffusion(t, x)

    # Compute auxiliary terms
    x_aux = x + a * dt + b * dW
    a_aux = drift(t + dt, x_aux)
    b_aux = diffusion(t + dt, x_aux)

    # Compute correction terms
    correction = 0.5 * (a_aux - a) * dt
    correction += 0.5 * (b_aux - b) * dZ
    correction += 0.5 * (b_aux - b) * (dW**2 - dt) / np.sqrt(dt)

    return x + a * dt + b * dW + correction


def platen_scheme(
    x0: float,
    t0: float,
    T: float,
    n_steps: int,
    drift: Callable,
    diffusion: Callable,
    rng: np.random.Generator,
    exact_solution: Optional[Callable] = None,
) -> PlatenResult:
    """
    Platen's explicit order 2.0 scheme for SDE integration.

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
    rng : np.random.Generator
        Random number generator
    exact_solution : Optional[Callable], optional
        Exact solution for error computation, by default None

    Returns
    -------
    PlatenResult
        Container with simulation results
    """
    dt = (T - t0) / n_steps
    times = np.linspace(t0, T, n_steps + 1)
    paths = np.zeros(n_steps + 1)
    paths[0] = x0

    for i in range(n_steps):
        dW = rng.normal(0, np.sqrt(dt))
        dZ = 0.5 * dt * (dW + rng.normal(0, np.sqrt(dt / 3)))
        paths[i + 1] = platen_step(paths[i], times[i], dt, drift, diffusion, dW, dZ)

    result = PlatenResult(paths=paths, times=times)

    if exact_solution is not None:
        exact_paths = exact_solution(times)
        errors = np.abs(paths - exact_paths)
        result.errors = errors
        result.strong_error = np.sqrt(np.mean(errors**2))
        result.weak_error = np.abs(np.mean(paths) - np.mean(exact_paths))

    return result


def multi_dimensional_platen(
    x0: np.ndarray,
    t0: float,
    T: float,
    n_steps: int,
    drift: Callable,
    diffusion: Callable,
    rng: np.random.Generator,
    exact_solution: Optional[Callable] = None,
) -> PlatenResult:
    """
    Multi-dimensional Platen's explicit order 2.0 scheme.

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
    rng : np.random.Generator
        Random number generator
    exact_solution : Optional[Callable], optional
        Exact solution for error computation, by default None

    Returns
    -------
    PlatenResult
        Container with simulation results
    """
    dt = (T - t0) / n_steps
    times = np.linspace(t0, T, n_steps + 1)
    n_dims = len(x0)
    paths = np.zeros((n_steps + 1, n_dims))
    paths[0] = x0

    for i in range(n_steps):
        dW = rng.normal(0, np.sqrt(dt), size=n_dims)
        dZ = 0.5 * dt * (dW + rng.normal(0, np.sqrt(dt / 3), size=n_dims))

        a = drift(times[i], paths[i])
        b = diffusion(times[i], paths[i])

        # Compute auxiliary terms
        x_aux = paths[i] + a * dt + b @ dW
        a_aux = drift(times[i] + dt, x_aux)
        b_aux = diffusion(times[i] + dt, x_aux)

        # Compute correction terms
        correction = 0.5 * (a_aux - a) * dt
        correction += 0.5 * (b_aux - b) @ dZ
        correction += 0.5 * (b_aux - b) @ (dW**2 - dt) / np.sqrt(dt)

        paths[i + 1] = paths[i] + a * dt + b @ dW + correction

    result = PlatenResult(paths=paths, times=times)

    if exact_solution is not None:
        exact_paths = exact_solution(times)
        errors = np.abs(paths - exact_paths)
        result.errors = errors
        result.strong_error = np.sqrt(np.mean(errors**2))
        result.weak_error = np.abs(
            np.mean(paths, axis=0) - np.mean(exact_paths, axis=0)
        )

    return result


def adaptive_platen(
    x0: Union[float, np.ndarray],
    t0: float,
    T: float,
    drift: Callable,
    diffusion: Callable,
    rng: np.random.Generator,
    tol: float = 1e-6,
    max_steps: int = 1000,
    min_steps: int = 10,
    exact_solution: Optional[Callable] = None,
) -> PlatenResult:
    """
    Adaptive Platen scheme with error control.

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
    PlatenResult
        Container with simulation results
    """
    n_steps = min_steps
    errors = []
    steps = []

    while n_steps <= max_steps:
        if isinstance(x0, np.ndarray):
            result = multi_dimensional_platen(
                x0, t0, T, n_steps, drift, diffusion, rng, exact_solution
            )
        else:
            result = platen_scheme(
                x0, t0, T, n_steps, drift, diffusion, rng, exact_solution
            )

        if result.strong_error is not None:
            errors.append(result.strong_error)
            steps.append(n_steps)

            if result.strong_error <= tol:
                break

        n_steps *= 2

    # Compute convergence rate if we have multiple error estimates
    if len(errors) > 1:
        log_errors = np.log(errors)
        log_steps = np.log(steps)
        result.convergence_rate = -np.polyfit(log_steps, log_errors, 1)[0]

    return result
