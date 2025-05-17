# File: chen3/payoffs/base.py
"""Base Payoff interface."""
from abc import ABC, abstractmethod

import numpy as np


class Payoff(ABC):
    @abstractmethod
    def __call__(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute payoff for each path.

        Parameters
        ----------
        paths : ndarray, shape (n_paths, n_steps+1, n_factors)
            Simulation output; for equity payoffs, factor 0 is S_t.

        Returns
        -------
        payoffs : ndarray, shape (n_paths,)
        """
        pass
