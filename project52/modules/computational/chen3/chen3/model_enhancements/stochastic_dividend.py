# File: chen3/model_enhancements/stochastic_dividend.py
"""
Stochastic dividend yield modeled as an Ornstein–Uhlenbeck process.
"""
import numpy as np
from numpy.typing import NDArray
from typing import Generator

class StochasticDividend:
    def __init__(self, kappa: float, theta: float, sigma: float, y0: float):
        """
        kappa: mean-reversion speed
        theta: long-term mean dividend yield
        sigma: volatility of dividend yield
        y0: initial dividend yield
        """
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.y0 = y0

    def simulate(self, dt: float, n_steps: int, n_paths: int, rng: np.random.Generator) -> NDArray:
        """
        Simulate dividend yield paths. Returns array of shape (n_paths, n_steps+1).
        """
        y = np.full(n_paths, self.y0)
        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = y
        for t in range(1, n_steps + 1):
            dy = self.kappa * (self.theta - y) * dt + self.sigma * np.sqrt(dt) * rng.standard_normal(n_paths)
            y = y + dy
            paths[:, t] = y
        return paths
