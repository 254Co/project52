# -------------------- chen3/numerical_engines/variance_reduction.py --------------------
"""
Variance reduction techniques for Monte Carlo simulations.
"""
import numpy as np
from scipy.stats import norm
from typing import Tuple

def antithetic_normal(rng: np.random.Generator, size: Tuple[int, int]) -> np.ndarray:
    """
    Generate antithetic normal variates.

    Parameters
    ----------
    rng : np.random.Generator
        Base random number generator.
    size : Tuple[int, int]
        Desired shape: (n, d). n is half the total paths, d is dimension.

    Returns
    -------
    arr : ndarray, shape (2*n, d)
    """
    Z = rng.standard_normal(size)
    return np.vstack([Z, -Z])


def control_variate(
    payoffs: np.ndarray,
    control_vals: np.ndarray,
    control_expectation: float
) -> np.ndarray:
    """
    Adjust payoffs using a control variate.

    Returns adjusted payoffs.
    """
    cov = np.cov(payoffs, control_vals, ddof=0)[0, 1]
    var_cv = np.var(control_vals)
    beta = cov / var_cv if var_cv > 0 else 0.0
    return payoffs - beta * (control_vals - control_expectation)


def latin_hypercube_normal(
    rng: np.random.Generator,
    n_samples: int,
    dim: int
) -> np.ndarray:
    """
    Generate Gaussian variates via Latin Hypercube Sampling.
    """
    u = np.empty((n_samples, dim))
    segments = np.linspace(0, 1, n_samples + 1)
    for j in range(dim):
        low = segments[:-1]
        high = segments[1:]
        u[:, j] = rng.uniform(low, high)
        rng.shuffle(u[:, j])
    return norm.ppf(u)
