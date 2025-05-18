"""State evolution for a vectorised Monte‑Carlo of the surplus‑consumption ratio.

This module implements the core dynamics of the Campbell-Cochrane habit formation model,
specifically the evolution of the surplus-consumption ratio and consumption growth.
The implementation is vectorized for efficient Monte Carlo simulation of multiple paths.
"""
from __future__ import annotations

import numpy as np
from numpy.random import Generator, PCG64

from .constants import ModelParams

__all__ = ["simulate_paths"]


def simulate_paths(
    params: ModelParams,
    n_steps: int,
    n_paths: int,
    rng: Generator | None = None,
    *,
    dtype=np.float64,
) -> dict[str, np.ndarray]:
    """Simulate joint paths of surplus-consumption ratio (S_t) and log consumption growth (ΔlogC_t).
    
    This function implements the Campbell-Cochrane habit formation model dynamics,
    simulating multiple paths of the surplus-consumption ratio and consumption growth.
    The simulation is vectorized for efficiency, allowing parallel computation of
    multiple paths.
    
    The dynamics follow:
        1. Log consumption growth: ΔlogC_t = g + σ_c * ε_t
        2. Surplus-consumption ratio: ΔlogS_t = (1-φ)(log(s̄) - log(S_t)) - 0.5λ_s(σ_c²) + λ_s*σ_c*ε_t
    
    where:
        - g: mean consumption growth
        - σ_c: consumption volatility
        - ε_t: i.i.d. standard normal shock
        - φ: persistence parameter
        - s̄: steady-state surplus ratio
        - λ_s: sensitivity parameter
    
    Args:
        params (ModelParams): Model parameters container.
        n_steps (int): Number of time steps to simulate.
        n_paths (int): Number of independent paths to simulate.
        rng (Generator | None, optional): Random number generator. If None, a new
            PCG64 generator is created. Defaults to None.
        dtype (type, optional): Data type for arrays. Defaults to np.float64.
        
    Returns:
        dict[str, np.ndarray]: Dictionary containing:
            - "S": Array of surplus-consumption ratios with shape (n_paths, n_steps+1)
            - "g_c": Array of log consumption growth rates with shape (n_paths, n_steps)
            
    Note:
        The surplus-consumption ratio is initialized at its steady-state value s̄
        for all paths. The simulation uses vectorized operations for efficiency.
    """
    p = params
    rng = rng or Generator(PCG64())

    S = np.empty((n_paths, n_steps + 1), dtype=dtype)
    g_c = np.empty((n_paths, n_steps), dtype=dtype)

    # Initial surplus ratio at steady state
    S[:, 0] = p.s_bar

    for t in range(n_steps):
        # IID shock to log consumption growth
        eps = rng.standard_normal(n_paths).astype(dtype, copy=False)
        g_c[:, t] = p.g + p.sigma_c * eps

        # Surplus‑consumption dynamics (Campbell‑Cochrane Eq. (8))
        delta_s = (
            (1.0 - p.phi) * (np.log(p.s_bar) - np.log(S[:, t]))
            - 0.5 * p.lambda_s * (p.sigma_c**2)
            + p.lambda_s * p.sigma_c * eps
        )
        S[:, t + 1] = np.exp(np.log(S[:, t]) + delta_s)

    return {"S": S, "g_c": g_c}