# File: chen3/greeks/aad.py
"""Adjoint‐mode AAD stub (via pyadjoint or custom tape)."""

import numpy as np
from typing import Callable

def greeks_aad(
    paths: np.ndarray,
    payoff: Callable[[np.ndarray], np.ndarray]
) -> float:
    """
    Compute ∂V/∂S₀ using reverse‐mode AD across the entire MC engine.
    This requires a taped version of the simulation; not trivial.
    """
    # In production, hook into a C++/CUDA kernel or Python tape.
    raise NotImplementedError("AAD requires a taped simulation kernel.")
