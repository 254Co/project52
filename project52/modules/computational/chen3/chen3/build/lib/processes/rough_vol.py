# File: chen3/processes/rough_vol.py
"""Rough volatility add-on (optional)."""
from .base import StateProcess

class RoughVariance(StateProcess):
    def __init__(self, hurst: float, eta: float):
        self.hurst = hurst  # alpha
        self.eta = eta      # volatility of vol

    def drift(self, v: any):
        raise NotImplementedError("Rough drift via memory kernel not implemented yet.")

    def diffusion(self, v: any):
        raise NotImplementedError("Rough diffusion via fractional kernel not implemented yet.")

