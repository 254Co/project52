# File: chen3/payoffs/vanilla.py
"""Vanilla European payoff."""
import numpy as np
from .base import Payoff

class Vanilla(Payoff):
    def __init__(self, strike: float, call: bool = True):
        self.strike = strike
        self.call = call

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        S_T = paths[:, -1, 0]
        if self.call:
            return np.maximum(S_T - self.strike, 0.0)
        else:
            return np.maximum(self.strike - S_T, 0.0)

