# -------------------- chen3/payoffs/exotic/lookback.py --------------------
"""
Lookback option payoff: based on max/min path.
"""
import numpy as np
from ..base import Payoff

class Lookback(Payoff):
    def __init__(self, call: bool = True, notional: float = 1.0):
        """
        call: True for floating-strike lookback call, False for put
        notional: scaling factor
        """
        self.call = call
        self.notional = notional

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        S = paths[:, :, 0]
        S_max = np.max(S, axis=1)
        S_min = np.min(S, axis=1)
        S_T = S[:, -1]
        if self.call:
            # floating-strike call: payoff = S_max - S_T
            return self.notional * np.maximum(S_max - S_T, 0.0)
        else:
            # floating-strike put: payoff = S_T - S_min
            return self.notional * np.maximum(S_T - S_min, 0.0)
