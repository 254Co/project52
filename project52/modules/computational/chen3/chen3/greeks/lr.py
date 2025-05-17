# File: chen3/greeks/lr.py
"""Likelihood‐ratio (score‐function) vega estimator."""

import numpy as np
from typing import Callable

def vega_likelihood_ratio(
    paths: np.ndarray,
    payoff: Callable[[np.ndarray], np.ndarray]
) -> float:
    """
    ∂V/∂σ by using d ln f / dσ weighting.
    Here we assume a simple Black‐Scholes‐like logic:
      weight = sum_{t} (Z₃_t * sqrt(dt) - θ*S*dt)/σ
    where Z₃_t is the third Brownian increment.
    For Chen we approximate by total path noise.
    """
    n_paths, n_steps_plus1, _ = paths.shape
    # NOTE: you need to record the Z's during sim; here we approximate:
    # assume we stored all dW’s for S in an array `dw_s` of shape (n_paths, n_steps)
    # To keep interface simple, raise for now:
    raise NotImplementedError("Store and pass S-factor dW array to compute LR weights.")
