# -------------------- schemes/exact.py --------------------
"""Exact sampling for CIR via non-central chi-square."""
import numpy as np
from scipy.stats import ncx2

def exact_scheme(process, dt: float):
    def step(state, _z):
        # state: array of r values
        kappa, theta, sigma = process.kappa, process.theta, process.sigma
        c = (sigma**2 * (1 - np.exp(-kappa*dt))) / (4*kappa)
        df = 4 * kappa * theta / sigma**2
        lamb = state * np.exp(-kappa*dt) / c
        # sample non-central chi-square
        x = ncx2.rvs(df, lamb, size=state.shape)
        return c * x
    return step