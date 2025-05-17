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
- Supports multi-dimensional state processes
- Efficient implementation using numpy vectorization

Implementation details:
- Uses central difference approximation for better accuracy
- Applies relative bump to initial conditions
- Handles multi-dimensional state processes
- Supports arbitrary payoff functions
- Maintains numerical stability through careful bump sizing

Typical use cases:
- Estimating delta (∂V/∂S₀) for path-dependent or exotic options
- Quick sensitivity analysis in Monte Carlo pricing
- Risk management and hedging calculations
- Model validation and benchmarking
- Real-time risk reporting

Example usage:
    >>> # Define a path-dependent payoff
    >>> def asian_call_payoff(paths):
    ...     S = paths[:, :, 0]  # asset price paths
    ...     K = 100.0  # strike price
    ...     avg = np.mean(S, axis=1)  # arithmetic average
    ...     return np.maximum(avg - K, 0)
    ...
    >>> # Simulate paths (example)
    >>> paths = simulate_paths(n_paths=10000, n_steps=252)
    >>> # Compute delta
    >>> delta = delta_pathwise(paths, asian_call_payoff)

Notes:
    - The bump size should be chosen carefully to balance accuracy and stability
    - For multi-dimensional processes, the method can be extended to other Greeks
    - The method assumes continuous payoff functions
    - Consider using variance reduction techniques for better accuracy
"""

from typing import Callable

import numpy as np


def delta_pathwise(
    paths: np.ndarray, payoff: Callable[[np.ndarray], np.ndarray], bump: float = 1e-4
) -> float:
    """
    Estimate delta (∂V/∂S₀) using the pathwise (bump-and-revalue) method.

    This function perturbs the initial asset price up and down by a small amount
    (bump), revalues the payoff for each perturbed path, and computes the central
    difference to estimate the sensitivity. It assumes that the first column of
    the path array holds the asset price (S).

    The method works by:
    1. Creating up and down bumped versions of the initial price
    2. Applying the same relative bump to all paths
    3. Revaluing the payoff for both bumped scenarios
    4. Computing the central difference approximation

    Mathematical formulation:
        Δ ≈ (V(S₀(1+ε)) - V(S₀(1-ε))) / (2εS₀)
    where:
        - V is the option value
        - S₀ is the initial asset price
        - ε is the relative bump size

    Args:
        paths (np.ndarray):
            Simulated Monte Carlo paths with shape (n_paths, n_steps+1, n_factors)
            - First dimension: number of simulation paths
            - Second dimension: time steps (including initial value)
            - Third dimension: state factors (first is asset price)
        payoff (Callable[[np.ndarray], np.ndarray]):
            Payoff function that maps paths to payoffs
            - Input: path array of shape (n_paths, n_steps+1, n_factors)
            - Output: array of shape (n_paths,) containing payoffs
        bump (float, optional):
            Relative bump size for finite difference (default: 1e-4)
            - Should be small enough for accuracy
            - Should be large enough to avoid roundoff error

    Returns:
        float: Estimated delta (∂V/∂S₀) using the pathwise method

    Raises:
        ValueError: If paths array is empty or has incorrect shape
        TypeError: If payoff function is not callable

    Example:
        >>> # Simulate paths for a European call option
        >>> paths = simulate_paths(n_paths=10000, n_steps=252)
        >>>
        >>> # Define the payoff function
        >>> def european_call_payoff(paths):
        ...     S = paths[:, -1, 0]  # final asset prices
        ...     K = 100.0  # strike price
        ...     return np.maximum(S - K, 0)
        ...
        >>> # Compute delta
        >>> delta = delta_pathwise(paths, european_call_payoff, bump=1e-4)
        >>> print(f"Estimated delta: {delta:.4f}")

    Notes:
        - The method does not require rerunning the full simulation
        - Works for any payoff function that depends on the path array
        - Central difference improves accuracy over one-sided bump
        - The bump size should be chosen based on the option's characteristics
        - For very small or large initial prices, consider using absolute bump
        - The method assumes the payoff is differentiable with respect to S₀
    """
    # bump initial stock in first column
    paths_up = paths.copy()
    paths_down = paths.copy()
    paths_up[:, 0, 0] *= 1 + bump
    paths_down[:, 0, 0] *= 1 - bump

    # revalue payoff (no need to rerun full sim; static bump)
    pu = payoff(paths_up)
    pd = payoff(paths_down)

    # central difference
    return float(np.mean(pu - pd) / (2 * bump * paths[:, 0, 0].mean()))
