# File: chen3/schemes/euler.py
"""Euler-Maruyama scheme."""
import numpy as np
from numba import njit

def euler_scheme(drift_fn, diffusion_fn, dt: float):
    @njit(fastmath=True)
    def step(state, z):
        return state + drift_fn(state)*dt + diffusion_fn(state)*np.sqrt(dt)*z
    return step
