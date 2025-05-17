"""
Runge-Kutta schemes for stochastic differential equations.
"""
import numpy as np
from typing import Tuple, Callable, Optional, Union, List, Dict
from dataclasses import dataclass

@dataclass
class RKResult:
    """Container for Runge-Kutta scheme simulation results."""
    paths: np.ndarray
    times: np.ndarray
    errors: Optional[np.ndarray] = None
    strong_error: Optional[float] = None
    weak_error: Optional[float] = None
    convergence_rate: Optional[float] = None
    stability_measure: Optional[float] = None
    local_errors: Optional[np.ndarray] = None
    step_sizes: Optional[np.ndarray] = None

def rk4_step(
    x: float,
    t: float,
    dt: float,
    drift: Callable,
    diffusion: Callable,
    dW: float,
    compute_local_error: bool = False
) -> Union[float, Tuple[float, float]]:
    """
    Single step of the RK4 scheme for SDEs.
    
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
    compute_local_error : bool, optional
        Whether to compute local error estimate, by default False
        
    Returns
    -------
    Union[float, Tuple[float, float]]
        Next state and optionally local error estimate
    """
    # Drift terms
    k1 = drift(t, x)
    k2 = drift(t + dt/2, x + k1*dt/2)
    k3 = drift(t + dt/2, x + k2*dt/2)
    k4 = drift(t + dt, x + k3*dt)
    
    # Diffusion terms
    l1 = diffusion(t, x)
    l2 = diffusion(t + dt/2, x + l1*dW/2)
    l3 = diffusion(t + dt/2, x + l2*dW/2)
    l4 = diffusion(t + dt, x + l3*dW)
    
    # Compute next state
    x_next = x + (k1 + 2*k2 + 2*k3 + k4)*dt/6 + (l1 + 2*l2 + 2*l3 + l4)*dW/6
    
    if compute_local_error:
        # Estimate local error using embedded method
        x_embedded = x + (k1 + k4)*dt/2 + (l1 + l4)*dW/2
        local_error = np.abs(x_next - x_embedded)
        return x_next, local_error
    
    return x_next

def compute_stability_measure(
    paths: np.ndarray,
    times: np.ndarray
) -> float:
    """
    Compute stability measure for the numerical scheme.
    
    Parameters
    ----------
    paths : np.ndarray
        Simulated paths
    times : np.ndarray
        Time points
        
    Returns
    -------
    float
        Stability measure
    """
    # Compute growth rate
    log_returns = np.log(np.abs(paths[1:] / paths[:-1]))
    growth_rate = np.mean(log_returns)
    
    # Compute variance of returns
    return_variance = np.var(log_returns)
    
    # Stability measure: negative growth rate and low variance indicate stability
    return -growth_rate / (1 + return_variance)

def rk4_scheme(
    x0: float,
    t0: float,
    T: float,
    n_steps: int,
    drift: Callable,
    diffusion: Callable,
    rng: np.random.Generator,
    exact_solution: Optional[Callable] = None,
    adaptive: bool = False,
    tol: float = 1e-6,
    min_dt: float = 1e-6,
    max_dt: float = 1.0
) -> RKResult:
    """
    RK4 scheme for SDE integration.
    
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
    adaptive : bool, optional
        Whether to use adaptive time-stepping, by default False
    tol : float, optional
        Error tolerance for adaptive stepping, by default 1e-6
    min_dt : float, optional
        Minimum time step, by default 1e-6
    max_dt : float, optional
        Maximum time step, by default 1.0
        
    Returns
    -------
    RKResult
        Container with simulation results
    """
    if adaptive:
        # Adaptive time-stepping
        times = [t0]
        paths = [x0]
        local_errors = []
        step_sizes = []
        t = t0
        x = x0
        
        while t < T:
            dt = min(max_dt, (T - t) / n_steps)
            dW = rng.normal(0, np.sqrt(dt))
            
            # Compute step with error estimate
            x_next, local_error = rk4_step(
                x, t, dt,
                drift, diffusion,
                dW, compute_local_error=True
            )
            
            # Adjust time step based on error
            if local_error > tol:
                dt = dt * (tol / local_error)**0.25
                dt = max(min_dt, min(dt, max_dt))
                continue
            
            # Accept step
            t += dt
            x = x_next
            times.append(t)
            paths.append(x)
            local_errors.append(local_error)
            step_sizes.append(dt)
        
        paths = np.array(paths)
        times = np.array(times)
        local_errors = np.array(local_errors)
        step_sizes = np.array(step_sizes)
    else:
        # Fixed time-stepping
        dt = (T - t0) / n_steps
        times = np.linspace(t0, T, n_steps + 1)
        paths = np.zeros(n_steps + 1)
        paths[0] = x0
        
        for i in range(n_steps):
            dW = rng.normal(0, np.sqrt(dt))
            paths[i+1] = rk4_step(
                paths[i],
                times[i],
                dt,
                drift,
                diffusion,
                dW
            )
    
    result = RKResult(
        paths=paths,
        times=times,
        local_errors=local_errors if adaptive else None,
        step_sizes=step_sizes if adaptive else None
    )
    
    if exact_solution is not None:
        exact_paths = exact_solution(times)
        errors = np.abs(paths - exact_paths)
        result.errors = errors
        result.strong_error = np.sqrt(np.mean(errors**2))
        result.weak_error = np.abs(np.mean(paths) - np.mean(exact_paths))
        
        # Compute convergence rate
        if len(errors) > 1:
            log_errors = np.log(errors)
            log_steps = np.log(np.diff(times))
            result.convergence_rate = -np.polyfit(log_steps, log_errors[:-1], 1)[0]
    
    # Compute stability measure
    result.stability_measure = compute_stability_measure(paths, times)
    
    return result

def multi_dimensional_rk4(
    x0: np.ndarray,
    t0: float,
    T: float,
    n_steps: int,
    drift: Callable,
    diffusion: Callable,
    rng: np.random.Generator,
    exact_solution: Optional[Callable] = None,
    adaptive: bool = False,
    tol: float = 1e-6,
    min_dt: float = 1e-6,
    max_dt: float = 1.0
) -> RKResult:
    """
    Multi-dimensional RK4 scheme for SDE integration.
    
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
    adaptive : bool, optional
        Whether to use adaptive time-stepping, by default False
    tol : float, optional
        Error tolerance for adaptive stepping, by default 1e-6
    min_dt : float, optional
        Minimum time step, by default 1e-6
    max_dt : float, optional
        Maximum time step, by default 1.0
        
    Returns
    -------
    RKResult
        Container with simulation results
    """
    if adaptive:
        # Adaptive time-stepping
        times = [t0]
        paths = [x0]
        local_errors = []
        step_sizes = []
        t = t0
        x = x0
        n_dims = len(x0)
        
        while t < T:
            dt = min(max_dt, (T - t) / n_steps)
            dW = rng.normal(0, np.sqrt(dt), size=n_dims)
            
            # Drift terms
            k1 = drift(t, x)
            k2 = drift(t + dt/2, x + k1*dt/2)
            k3 = drift(t + dt/2, x + k2*dt/2)
            k4 = drift(t + dt, x + k3*dt)
            
            # Diffusion terms
            l1 = diffusion(t, x)
            l2 = diffusion(t + dt/2, x + l1 @ dW/2)
            l3 = diffusion(t + dt/2, x + l2 @ dW/2)
            l4 = diffusion(t + dt, x + l3 @ dW)
            
            # Compute next state
            x_next = x + (
                (k1 + 2*k2 + 2*k3 + k4)*dt/6 +
                (l1 + 2*l2 + 2*l3 + l4) @ dW/6
            )
            
            # Estimate local error
            x_embedded = x + (k1 + k4)*dt/2 + (l1 + l4) @ dW/2
            local_error = np.linalg.norm(x_next - x_embedded)
            
            # Adjust time step based on error
            if local_error > tol:
                dt = dt * (tol / local_error)**0.25
                dt = max(min_dt, min(dt, max_dt))
                continue
            
            # Accept step
            t += dt
            x = x_next
            times.append(t)
            paths.append(x)
            local_errors.append(local_error)
            step_sizes.append(dt)
        
        paths = np.array(paths)
        times = np.array(times)
        local_errors = np.array(local_errors)
        step_sizes = np.array(step_sizes)
    else:
        # Fixed time-stepping
        dt = (T - t0) / n_steps
        times = np.linspace(t0, T, n_steps + 1)
        n_dims = len(x0)
        paths = np.zeros((n_steps + 1, n_dims))
        paths[0] = x0
        
        for i in range(n_steps):
            dW = rng.normal(0, np.sqrt(dt), size=n_dims)
            
            # Drift terms
            k1 = drift(times[i], paths[i])
            k2 = drift(times[i] + dt/2, paths[i] + k1*dt/2)
            k3 = drift(times[i] + dt/2, paths[i] + k2*dt/2)
            k4 = drift(times[i] + dt, paths[i] + k3*dt)
            
            # Diffusion terms
            l1 = diffusion(times[i], paths[i])
            l2 = diffusion(times[i] + dt/2, paths[i] + l1 @ dW/2)
            l3 = diffusion(times[i] + dt/2, paths[i] + l2 @ dW/2)
            l4 = diffusion(times[i] + dt, paths[i] + l3 @ dW)
            
            paths[i+1] = paths[i] + (
                (k1 + 2*k2 + 2*k3 + k4)*dt/6 +
                (l1 + 2*l2 + 2*l3 + l4) @ dW/6
            )
    
    result = RKResult(
        paths=paths,
        times=times,
        local_errors=local_errors if adaptive else None,
        step_sizes=step_sizes if adaptive else None
    )
    
    if exact_solution is not None:
        exact_paths = exact_solution(times)
        errors = np.abs(paths - exact_paths)
        result.errors = errors
        result.strong_error = np.sqrt(np.mean(errors**2))
        result.weak_error = np.abs(np.mean(paths, axis=0) - np.mean(exact_paths, axis=0))
        
        # Compute convergence rate
        if len(errors) > 1:
            log_errors = np.log(np.linalg.norm(errors, axis=1))
            log_steps = np.log(np.diff(times))
            result.convergence_rate = -np.polyfit(log_steps, log_errors[:-1], 1)[0]
    
    # Compute stability measure
    result.stability_measure = compute_stability_measure(
        np.linalg.norm(paths, axis=1),
        times
    )
    
    return result

def adaptive_rk4(
    x0: Union[float, np.ndarray],
    t0: float,
    T: float,
    drift: Callable,
    diffusion: Callable,
    rng: np.random.Generator,
    tol: float = 1e-6,
    max_steps: int = 1000,
    min_steps: int = 10,
    min_dt: float = 1e-6,
    max_dt: float = 1.0,
    exact_solution: Optional[Callable] = None
) -> RKResult:
    """
    Adaptive RK4 scheme with error control.
    
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
    min_dt : float, optional
        Minimum time step, by default 1e-6
    max_dt : float, optional
        Maximum time step, by default 1.0
    exact_solution : Optional[Callable], optional
        Exact solution for error computation, by default None
        
    Returns
    -------
    RKResult
        Container with simulation results
    """
    if isinstance(x0, np.ndarray):
        return multi_dimensional_rk4(
            x0, t0, T, max_steps,
            drift, diffusion,
            rng, exact_solution,
            adaptive=True,
            tol=tol,
            min_dt=min_dt,
            max_dt=max_dt
        )
    else:
        return rk4_scheme(
            x0, t0, T, max_steps,
            drift, diffusion,
            rng, exact_solution,
            adaptive=True,
            tol=tol,
            min_dt=min_dt,
            max_dt=max_dt
        ) 