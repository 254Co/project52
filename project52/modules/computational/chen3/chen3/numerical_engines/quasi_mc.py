# -------------------- chen3/numerical_engines/quasi_mc.py --------------------
"""
Low-discrepancy (quasi-Monte Carlo) sequences.
"""
import numpy as np
from scipy.stats import qmc, norm


def sobol_normal(
    n_samples: int,
    dim: int,
    scramble: bool = True,
    seed: int = None
) -> np.ndarray:
    sampler = qmc.Sobol(d=dim, scramble=scramble, seed=seed)
    u = sampler.random(n_samples)
    return norm.ppf(u)


def halton_normal(
    n_samples: int,
    dim: int,
    scramble: bool = False,
    seed: int = None
) -> np.ndarray:
    sampler = qmc.Halton(d=dim, scramble=scramble, seed=seed)
    u = sampler.random(n_samples)
    return norm.ppf(u)