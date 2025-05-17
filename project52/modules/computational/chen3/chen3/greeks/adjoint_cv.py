# -------------------- greeks/adjoint_cv.py --------------------
"""Adjoint-SDE control variate variance reduction stub."""
import numpy as np
from typing import Callable

def adjoint_control_variate(
    paths: np.ndarray,
    payoff: Callable[[np.ndarray], np.ndarray],
    cv_term: float
) -> float:
    """Combine pathwise payoff with precomputed control variate."""
    payoffs = payoff(paths)
    # Assume cv_term = E[control_variate] known
    # and we have control variate value stored: cv_vals
    # Here cv_vals placeholder as zeros
    cv_vals = np.zeros_like(payoffs)
    beta = np.cov(payoffs, cv_vals)[0,1] / (np.var(cv_vals) + 1e-12)
    adjusted = payoffs - beta * (cv_vals - cv_term)
    return float(np.mean(adjusted))