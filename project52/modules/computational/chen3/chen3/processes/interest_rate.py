# File: chen3/processes/interest_rate.py
"""CIR short-rate process."""
import numpy as np
from .base import StateProcess

class ShortRateCIR(StateProcess):
    def __init__(self, kappa: float, theta: float, sigma: float, r0: float):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.r0 = r0

    def drift(self, r: np.ndarray) -> np.ndarray:
        return self.kappa * (self.theta - r)

    def diffusion(self, r: np.ndarray) -> np.ndarray:
        return self.sigma * np.sqrt(np.maximum(r, 0.0))