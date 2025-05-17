# File: chen3/payoffs/convertible_bond.py
"""Convertible bond payoff stub."""
import numpy as np
from .base import Payoff

class ConvertibleBond(Payoff):
    def __init__(self, conversion_ratio: float):
        self.conversion_ratio = conversion_ratio

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        # Placeholder for complex structure
        S_T = paths[:, -1, 0]
        return self.conversion_ratio * S_T
