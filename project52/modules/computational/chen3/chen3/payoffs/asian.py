# File: chen3/payoffs/asian.py
"""Arithmetic-average Asian option payoff."""
import numpy as np
from .base import Payoff

class Asian(Payoff):
    def __init__(self, strike: float, call: bool = True):
        """
        strike : exercise price
        call   : True = call on average, False = put on average
        """
        self.strike = strike
        self.call = call

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        # average of the equity factor over all times
        avg_S = paths[:, :, 0].mean(axis=1)
        if self.call:
            return np.maximum(avg_S - self.strike, 0.0)
        else:
            return np.maximum(self.strike - avg_S, 0.0)
