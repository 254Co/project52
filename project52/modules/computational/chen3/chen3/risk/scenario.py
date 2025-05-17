# -------------------- risk/scenario.py --------------------
"""Scenario shock generator and application."""
import numpy as np
from typing import Dict, Any

class Scenario:
    """
    Represents a simultaneous shock to model parameters.
    fields:
      rate_shift: float       # additive shift in short rate (bp)
      vol_shift: float        # relative shift in variance (pct)
      equity_shift: float     # relative shift in equity price (pct)
    """
    def __init__(self,
                 rate_shift: float = 0.0,
                 vol_shift: float = 0.0,
                 equity_shift: float = 0.0):
        self.rate_shift = rate_shift
        self.vol_shift = vol_shift
        self.equity_shift = equity_shift

    def apply(self, paths: np.ndarray) -> np.ndarray:
        """
        Apply scenario to simulated paths.
        paths shape: (n_paths, n_steps+1, 3) -> [S, v, r]
        Returns new shocked paths array.
        """
        shocked = paths.copy()
        # Add shift in short rate (in absolute terms)
        shocked[:,:,2] += self.rate_shift
        # Multiply variance by (1 + vol_shift)
        shocked[:,:,1] *= (1 + self.vol_shift)
        # Multiply stock price by (1 + equity_shift)
        shocked[:,:,0] *= (1 + self.equity_shift)
        return shocked