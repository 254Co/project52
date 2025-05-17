# File: chen3/greeks/pathwise.py
"""Pathwise (a.k.a. ‘bump‐and‐revalue’) delta estimator."""

import numpy as np
from typing import Callable

def delta_pathwise(
    paths: np.ndarray,
    payoff: Callable[[np.ndarray], np.ndarray],
    bump: float = 1e-4
) -> float:
    """
    Estimate ∂V/∂S₀ by bumping S₀ up and down.
    Assumes paths[:,:,0] holds S.
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
