# File: chen3/payoffs/vanilla.py
"""Vanilla European option payoff."""
import numpy as np

from .base import Payoff


class Vanilla(Payoff):
    def __init__(self, strike: float, call: bool = True):
        """
        strike : exercise price
        call   : True = call, False = put
        """
        self.strike = strike
        self.call = call

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        # S_T is factor 0 at final time
        S_T = paths[:, -1, 0]
        if self.call:
            return np.maximum(S_T - self.strike, 0.0)
        else:
            return np.maximum(self.strike - S_T, 0.0)

    def payoff(self, r_paths, S_paths, v_paths):
        # Stack S_paths, r_paths, v_paths into a 3D array as expected by __call__
        # S_paths, r_paths, v_paths: (n_paths, n_steps+1)
        # paths: (n_paths, n_steps+1, 3) with [:,:,0]=S, [:,:,1]=r, [:,:,2]=v
        paths = np.stack([S_paths, r_paths, v_paths], axis=-1)
        return self.__call__(paths)
