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

import numpy as np
from numba import njit


def euler_scheme(drift_fn, diffusion_fn, dt: float):
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
    @njit(fastmath=True)
    def step(state, z):
        """
        Compute one step of the Euler-Maruyama scheme.
        
        Args:
            state (np.ndarray): Current state of the process
            z (np.ndarray): Standard normal random variables
        
        Returns:
            np.ndarray: Next state according to the Euler-Maruyama scheme
        """
        return state + drift_fn(state)*dt + diffusion_fn(state)*np.sqrt(dt)*z
    
    return step
