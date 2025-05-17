# File: chen3/correlation.py
"""Cholesky-based correlation factory."""
import numpy as np
from numpy.linalg import cholesky

def cholesky_correlation(rho: np.ndarray) -> np.ndarray:
    """Compute lower-triangular matrix L such that L @ L.T = rho"""
    return cholesky(rho)

