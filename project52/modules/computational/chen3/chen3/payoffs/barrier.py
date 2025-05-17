# File: chen3/payoffs/barrier.py
"""Barrier option payoffs (knock-in / knock-out)."""
import numpy as np

from .base import Payoff


class Barrier(Payoff):
    def __init__(
        self, strike: float, barrier: float, knock_in: bool = True, call: bool = True
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

    def payoff(self, r_paths: np.ndarray, S_paths: np.ndarray, v_paths: np.ndarray) -> np.ndarray:
        """
        Compute the payoff for the barrier option.
        
        Args:
            r_paths: Interest rate paths
            S_paths: Stock price paths
            v_paths: Variance paths
            
        Returns:
            np.ndarray: Array of payoffs for each path
        """
        # Stack the paths as expected by __call__
        paths = np.stack([S_paths, r_paths, v_paths], axis=2)
        return self(paths)
