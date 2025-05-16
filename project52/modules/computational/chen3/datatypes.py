# File: chen3/datatypes.py
"""Core parameter data structures."""
from typing import NamedTuple
import numpy as np

class RateParams(NamedTuple):
    kappa: float
    theta: float
    sigma: float
    r0: float

class EquityParams(NamedTuple):
    mu: float             # drift
    q: float              # dividend yield
    v0: float             # initial variance
    kappa_v: float        # variance mean-reversion
    theta_v: float        # variance long-run
    sigma_v: float        # volatility of variance

class ModelParams(NamedTuple):
    rate: RateParams
    equity: EquityParams
    corr_matrix: np.ndarray  # shape (3,3)

