# File: chen3/payoffs/vanilla.py
"""Vanilla European option payoff."""
import numpy as np

from .base import Payoff
from ..utils.exceptions import ValidationError


class Vanilla(Payoff):
    def __init__(self, strike: float, call: bool = True):
        """
        Initialize a vanilla European option.

        Args:
            strike: Exercise price (must be positive)
            call: True for call option, False for put option

        Raises:
            ValidationError: If strike price is not positive
        """
        if strike <= 0:
            raise ValidationError("Strike price must be positive")
        self.strike = strike
        self.call = call

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute the option payoff.

        Args:
            paths: Array of shape (n_paths, n_steps+1, n_factors) containing
                  simulated paths. For vanilla options, only the final stock
                  price (factor 0) is used.

        Returns:
            Array of payoffs for each path
        """
        # S_T is factor 0 at final time
        S_T = paths[:, -1, 0]
        if self.call:
            return np.maximum(S_T - self.strike, 0.0)
        else:
            return np.maximum(self.strike - S_T, 0.0)

    def payoff(self, r_paths, S_paths, v_paths):
        """
        Compute the option payoff using separate path arrays.

        Args:
            r_paths: Interest rate paths
            S_paths: Stock price paths
            v_paths: Variance paths

        Returns:
            Array of payoffs for each path
        """
        # Stack S_paths, r_paths, v_paths into a 3D array as expected by __call__
        # S_paths, r_paths, v_paths: (n_paths, n_steps+1)
        # paths: (n_paths, n_steps+1, 3) with [:,:,0]=S, [:,:,1]=r, [:,:,2]=v
        paths = np.stack([S_paths, r_paths, v_paths], axis=-1)
        return self.__call__(paths)
