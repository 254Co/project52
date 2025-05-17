# File: chen3/payoffs/barrier.py
"""Barrier option payoff."""
import numpy as np
from .base import Payoff

class Barrier(Payoff):
    def __init__(self, strike: float, barrier: float, knock_in: bool = True):
        self.strike = strike
        self.barrier = barrier
        self.knock_in = knock_in

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        S = paths[:, :, 0]
        hit = (S >= self.barrier).any(axis=1)
        payoff = np.maximum(paths[:, -1, 0] - self.strike, 0.0)
        if self.knock_in:
            return payoff * hit
        else:
            return payoff * ~hit

