"""
Predictor-corrector schemes for stochastic differential equations.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np


@dataclass
class PCSchemeResult:
    """Container for predictor-corrector scheme simulation results."""

    paths: np.ndarray
    times: np.ndarray
    predictor_paths: Optional[np.ndarray] = None
    errors: Optional[np.ndarray] = None
    strong_error: Optional[float] = None
    weak_error: Optional[float] = None


def euler_predictor(
    x: float, t: float, dt: float, drift: Callable, diffusion: Callable, dW: float
) -> float:
    """
    Euler predictor step.

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

    Returns
    -------
    float
        Predicted state
    """
    return x + drift(t, x) * dt + diffusion(t, x) * dW


def trapezoidal_corrector(
    x: float,
    x_pred: float,
    t: float,
    dt: float,
    drift: Callable,
    diffusion: Callable,
    dW: float,
    theta: float = 0.5,
) -> float:
    """
    Trapezoidal corrector step.

    Parameters
    ----------
    x : float
        Current state
    x_pred : float
        Predicted state
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
    theta : float, optional
        Weighting parameter, by default 0.5

    Returns
    -------
    float
        Corrected state
    """
    a_old = drift(t, x)
    a_new = drift(t + dt, x_pred)
    b_old = diffusion(t, x)
    b_new = diffusion(t + dt, x_pred)

    return (
        x
        + (theta * a_old + (1 - theta) * a_new) * dt
        + (theta * b_old + (1 - theta) * b_new) * dW
    )


def predictor_corrector_scheme(
    x0: float,
    t0: float,
    T: float,
    n_steps: int,
    drift: Callable,
    diffusion: Callable,
    rng: np.random.Generator,
    theta: float = 0.5,
    max_iter: int = 10,
    tol: float = 1e-6,
    exact_solution: Optional[Callable] = None,
) -> PCSchemeResult:
    """
    Predictor-corrector scheme for SDE integration.

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
    theta : float, optional
        Weighting parameter, by default 0.5
    max_iter : int, optional
        Maximum number of correction iterations, by default 10
    tol : float, optional
        Convergence tolerance, by default 1e-6
    exact_solution : Optional[Callable], optional
        Exact solution for error computation, by default None

    Returns
    -------
    PCSchemeResult
        Container with simulation results
    """
    dt = (T - t0) / n_steps
    times = np.linspace(t0, T, n_steps + 1)
    paths = np.zeros(n_steps + 1)
    predictor_paths = np.zeros(n_steps + 1)
    paths[0] = x0
    predictor_paths[0] = x0

    for i in range(n_steps):
        dW = rng.normal(0, np.sqrt(dt))

        # Predictor step
        x_pred = euler_predictor(paths[i], times[i], dt, drift, diffusion, dW)
        predictor_paths[i + 1] = x_pred

        # Corrector step with iteration
        x_corr = x_pred
        for _ in range(max_iter):
            x_new = trapezoidal_corrector(
                paths[i], x_corr, times[i], dt, drift, diffusion, dW, theta
            )
            if np.abs(x_new - x_corr) < tol:
                break
            x_corr = x_new

        paths[i + 1] = x_corr

    result = PCSchemeResult(paths=paths, times=times, predictor_paths=predictor_paths)

    if exact_solution is not None:
        exact_paths = exact_solution(times)
        errors = np.abs(paths - exact_paths)
        result.errors = errors
        result.strong_error = np.sqrt(np.mean(errors**2))
        result.weak_error = np.abs(np.mean(paths) - np.mean(exact_paths))

    return result


def multi_dimensional_predictor_corrector(
    x0: np.ndarray,
    t0: float,
    T: float,
    n_steps: int,
    drift: Callable,
    diffusion: Callable,
    rng: np.random.Generator,
    theta: float = 0.5,
    max_iter: int = 10,
    tol: float = 1e-6,
    exact_solution: Optional[Callable] = None,
) -> PCSchemeResult:
    """
    Multi-dimensional predictor-corrector scheme.

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
    theta : float, optional
        Weighting parameter, by default 0.5
    max_iter : int, optional
        Maximum number of correction iterations, by default 10
    tol : float, optional
        Convergence tolerance, by default 1e-6
    exact_solution : Optional[Callable], optional
        Exact solution for error computation, by default None

    Returns
    -------
    PCSchemeResult
        Container with simulation results
    """
    dt = (T - t0) / n_steps
    times = np.linspace(t0, T, n_steps + 1)
    n_dims = len(x0)
    paths = np.zeros((n_steps + 1, n_dims))
    predictor_paths = np.zeros((n_steps + 1, n_dims))
    paths[0] = x0
    predictor_paths[0] = x0

    for i in range(n_steps):
        dW = rng.normal(0, np.sqrt(dt), size=n_dims)

        # Predictor step
        a = drift(times[i], paths[i])
        b = diffusion(times[i], paths[i])
        x_pred = paths[i] + a * dt + b @ dW
        predictor_paths[i + 1] = x_pred

        # Corrector step with iteration
        x_corr = x_pred
        for _ in range(max_iter):
            a_old = drift(times[i], paths[i])
            a_new = drift(times[i] + dt, x_corr)
            b_old = diffusion(times[i], paths[i])
            b_new = diffusion(times[i] + dt, x_corr)

            x_new = (
                paths[i]
                + (theta * a_old + (1 - theta) * a_new) * dt
                + (theta * b_old + (1 - theta) * b_new) @ dW
            )

            if np.all(np.abs(x_new - x_corr) < tol):
                break
            x_corr = x_new

        paths[i + 1] = x_corr

    result = PCSchemeResult(paths=paths, times=times, predictor_paths=predictor_paths)

    if exact_solution is not None:
        exact_paths = exact_solution(times)
        errors = np.abs(paths - exact_paths)
        result.errors = errors
        result.strong_error = np.sqrt(np.mean(errors**2))
        result.weak_error = np.abs(
            np.mean(paths, axis=0) - np.mean(exact_paths, axis=0)
        )

    return result


def adaptive_predictor_corrector(
    x0: Union[float, np.ndarray],
    t0: float,
    T: float,
    drift: Callable,
    diffusion: Callable,
    rng: np.random.Generator,
    theta: float = 0.5,
    tol: float = 1e-6,
    max_steps: int = 1000,
    min_steps: int = 10,
    exact_solution: Optional[Callable] = None,
) -> PCSchemeResult:
    """
    Adaptive predictor-corrector scheme with error control.

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
    theta : float, optional
        Weighting parameter, by default 0.5
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
    PCSchemeResult
        Container with simulation results
    """
    n_steps = min_steps
    while n_steps <= max_steps:
        if isinstance(x0, np.ndarray):
            result = multi_dimensional_predictor_corrector(
                x0,
                t0,
                T,
                n_steps,
                drift,
                diffusion,
                rng,
                theta,
                tol=tol,
                exact_solution=exact_solution,
            )
        else:
            result = predictor_corrector_scheme(
                x0,
                t0,
                T,
                n_steps,
                drift,
                diffusion,
                rng,
                theta,
                tol=tol,
                exact_solution=exact_solution,
            )

        if result.strong_error is not None and result.strong_error <= tol:
            break

        n_steps *= 2

    return result
