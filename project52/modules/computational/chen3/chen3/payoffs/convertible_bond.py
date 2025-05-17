# File: chen3/payoffs/convertible_bond.py
"""Convertible bond payoff (equity- conversion)."""
import numpy as np

from .base import Payoff


class ConvertibleBond(Payoff):
    def __init__(self, face_value: float, conversion_ratio: float):
        """
        face_value      : bond principal repaid at maturity if not converted
        conversion_ratio: number of shares per bond if converted
        """
        self.face_value = face_value
        self.conversion_ratio = conversion_ratio

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        """
        For each path, the holder chooses max(face_value, conversion_ratio * S_T).
        """
        S_T = paths[:, -1, 0]
        equity_value = self.conversion_ratio * S_T
        return np.maximum(self.face_value, equity_value)
