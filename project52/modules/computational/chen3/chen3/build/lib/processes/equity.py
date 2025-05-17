# File: chen3/processes/equity.py
"""Heston-like equity and variance process."""
import numpy as np
from .base import StateProcess

class EquityProcess(StateProcess):
    def __init__(self, mu: float, q: float,
                 kappa_v: float, theta_v: float, sigma_v: float, v0: float):
        self.mu = mu
        self.q = q
        self.kappa_v = kappa_v
        self.theta_v = theta_v
        self.sigma_v = sigma_v
        self.v0 = v0

    def drift(self, state: np.ndarray) -> np.ndarray:
        # state[:,0]=S, state[:,1]=v
        S, v = state[:,0], state[:,1]
        return np.stack([ (self.mu - self.q)*S, self.kappa_v*(self.theta_v - v) ], axis=1)

    def diffusion(self, state: np.ndarray) -> np.ndarray:
        S, v = state[:,0], state[:,1]
        return np.stack([ np.sqrt(np.maximum(v,0.0))*S, self.sigma_v*np.sqrt(np.maximum(v,0.0)) ], axis=1)
