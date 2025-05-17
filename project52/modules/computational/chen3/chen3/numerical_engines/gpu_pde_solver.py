# -------------------- chen3/numerical_engines/gpu_pde_solver.py --------------------
"""
Crank-Nicolson PDE solver on GPU for Black-Scholes.
"""
import cupy as cp
from typing import Tuple

def gpu_crank_nicolson(
    S_max: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_S: int = 200,
    n_t: int = 200
) -> float:
    dS = S_max / n_S
    dt = T / n_t
    S = cp.linspace(0, S_max, n_S+1)
    V = cp.maximum(S - K, 0)
    # coefficients
    i = cp.arange(1, n_S)
    a = 0.25 * dt * (sigma**2 * i**2 - r * i)
    b = -dt * 0.5 * (sigma**2 * i**2 + r)
    c = 0.25 * dt * (sigma**2 * i**2 + r * i)
    # matrices
    A = cp.zeros((n_S-1, n_S-1))
    B = cp.zeros_like(A)
    for idx in range(n_S-1):
        A[idx, idx] = 1 - b[idx]
        B[idx, idx] = 1 + b[idx]
        if idx > 0:
            A[idx, idx-1] = -a[idx]
            B[idx, idx-1] = a[idx]
        if idx < n_S-2:
            A[idx, idx+1] = -c[idx]
            B[idx, idx+1] = c[idx]
    # time stepping
    for _ in range(n_t):
        rhs = B @ V[1:-1]
        # boundary
        rhs[0]  += a[1] * 0
        rhs[-1] += c[-2] * (S_max - K * cp.exp(-r * (_*dt)))
        V[1:-1] = cp.linalg.solve(A, rhs)
    return float(cp.interp(K, S, V))
