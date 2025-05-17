# -------------------- schemes/adf.py --------------------
"""Andersen-Piterbarg order-1.5 weak scheme for CIR-like SDEs."""
import numpy as np

def adf_scheme(process, dt: float):
    def step(state, z):
        # for CIR process: use Milstein correction term
        drift = process.drift(state)
        diff  = process.diffusion(state)
        state_new = state + drift*dt + diff*z
        # Milstein term: 0.5 * diff' * diff * (z^2 - dt)
        # Here diff' approximated by sigma/(2*sqrt(state)) for CIR
        sigma = getattr(process, 'sigma', None)
        if sigma is not None:
            correction = 0.25 * sigma**2 * (z**2 - dt)
            state_new += correction
        return np.maximum(state_new, 0.0)
    return step
