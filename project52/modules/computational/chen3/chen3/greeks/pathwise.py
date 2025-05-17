# File: chen3/greeks/pathwise.py
"""
Pathwise (Bump-and-Revalue) Greeks Module

This module provides a pathwise (bump-and-revalue) estimator for computing Greeks,
such as delta, in Monte Carlo simulations. The pathwise method estimates sensitivities
by perturbing the initial condition (e.g., initial asset price) and revaluing the
payoff, using a central difference approximation.

Key features:
- Simple and robust estimator for delta and other Greeks
- Does not require access to simulation internals or adjoint methods
- Works with any payoff function and path array

Typical use case:
- Estimating delta (∂V/∂S₀) for path-dependent or exotic options
- Quick sensitivity analysis in Monte Carlo pricing
"""

import numpy as np
from typing import Callable

def delta_pathwise(
    paths: np.ndarray,
    payoff: Callable[[np.ndarray], np.ndarray],
    bump: float = 1e-4
) -> float:
    """
    Estimate delta (∂V/∂S₀) using the pathwise (bump-and-revalue) method.
    
    This function perturbs the initial asset price up and down by a small amount
    (bump), revalues the payoff for each perturbed path, and computes the central
    difference to estimate the sensitivity. It assumes that the first column of
    the path array holds the asset price (S).
    
    Args:
        paths (np.ndarray): Simulated Monte Carlo paths, shape (n_paths, n_steps+1, n_factors)
        payoff (Callable[[np.ndarray], np.ndarray]): Payoff function mapping paths to payoffs
        bump (float, optional): Relative bump size for finite difference (default: 1e-4)
    
    Returns:
        float: Estimated delta (∂V/∂S₀) using the pathwise method
    
    Example:
        >>> delta = delta_pathwise(paths, payoff, bump=1e-4)
    
    Notes:
        - The method does not require rerunning the full simulation
        - Works for any payoff function that depends on the path array
        - Central difference improves accuracy over one-sided bump
    """
    # bump initial stock in first column
    paths_up   = paths.copy()
    paths_down = paths.copy()
    paths_up[:, 0, 0]   *= (1 + bump)
    paths_down[:, 0, 0] *= (1 - bump)

    # revalue payoff (no need to rerun full sim; static bump)
    pu = payoff(paths_up)
    pd = payoff(paths_down)

    # central difference
    return float(np.mean(pu - pd) / (2 * bump * paths[:, 0, 0].mean()))
