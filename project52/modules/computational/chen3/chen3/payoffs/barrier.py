# File: chen3/payoffs/barrier.py
"""Barrier option payoffs (knock-in / knock-out)."""
import numpy as np
from .base import Payoff

class Barrier(Payoff):
    def __init__(
        self,
        strike: float,
        barrier: float,
        knock_in: bool = True,
        call: bool = True
    ):
        """
        strike    : exercise price
        barrier   : barrier level
        knock_in  : True = knock-in, False = knock-out
        call      : True = call payoff, False = put payoff
        """
        self.strike = strike
        self.barrier = barrier
        self.knock_in = knock_in
        self.call = call

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        S = paths[:, :, 0]  # shape (n_paths, n_steps+1)
        hit = (S >= self.barrier).any(axis=1)
        S_T = S[:, -1]
        if self.call:
            raw = np.maximum(S_T - self.strike, 0.0)
        else:
            raw = np.maximum(self.strike - S_T, 0.0)

        if self.knock_in:
            return raw * hit
        else:
            return raw * (~hit)
