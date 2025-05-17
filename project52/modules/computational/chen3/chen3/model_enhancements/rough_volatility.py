# File: chen3/model_enhancements/rough_volatility.py
"""
Rough volatility model driven by fractional Brownian motion.
"""
import numpy as np
from numpy.typing import NDArray
from typing import Generator

class RoughVolatility:
    def __init__(self, hurst: float, eta: float, v0: float, n_steps: int):
        """
        hurst: Hurst exponent (0 < H < 0.5)
        eta: volatility of vol
        v0: initial variance
        n_steps: number of time steps in simulation
        """
        self.hurst = hurst
        self.eta = eta
        self.v0 = v0
        self.n_steps = n_steps
        # Precompute fractional kernel weights
        self.kernel = np.array([((i+1)**(hurst-0.5) - i**(hurst-0.5)) for i in range(n_steps)])

    def simulate(self, dt: float, rng: np.random.Generator) -> NDArray:
        """
        Simulate a rough volatility path of length n_steps+1.
        Returns an array v of shape (n_steps+1,).
        """
        # Brownian increments
        dw = rng.standard_normal(self.n_steps) * np.sqrt(dt)
        # Fractional convolution
        increments = self.eta * np.convolve(dw, self.kernel, mode='full')[:self.n_steps]
        v = np.empty(self.n_steps + 1)
        v[0] = self.v0
        for t in range(1, self.n_steps + 1):
            v[t] = max(v[t-1] + increments[t-1], 0.0)
        return v
