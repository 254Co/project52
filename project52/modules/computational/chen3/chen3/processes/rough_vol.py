# -------------------- processes/rough_vol.py --------------------
"""Rough volatility add-on via Riemann approximation."""
import numpy as np
from .base import StateProcess

class RoughVariance(StateProcess):
    def __init__(self, hurst: float, eta: float, v0: float, steps: int):
        self.hurst = hurst  # H in (0,0.5)
        self.eta = eta      # vol of vol
        self.v0 = v0        # initial rough variance
        self.steps = steps
        # Precompute fractional kernel weights
        self.kernel = np.array([((i+1)**self.hurst - i**self.hurst) for i in range(steps)])

    def simulate(self, dt: float, rng: np.random.Generator):
        """Generate a rough variance path of length `steps+1`"""
        dw = rng.standard_normal(self.steps) * np.sqrt(dt)
        increments = self.eta * np.convolve(self.kernel, dw, mode='full')[:self.steps]
        v = np.empty(self.steps+1)
        v[0] = self.v0
        for t in range(1, self.steps+1):
            v[t] = abs(v[t-1] + increments[t-1])
        return v

    def drift(self, state):
        raise NotImplementedError("Use simulate() for rough variance evolution.")

    def diffusion(self, state):
        raise NotImplementedError("Use simulate() for rough variance evolution.")
