# File: chen3/payoffs/base.py
"""Base Payoff class."""
from abc import ABC, abstractmethod
import numpy as np

class Payoff(ABC):
    @abstractmethod
    def __call__(self, paths: np.ndarray) -> np.ndarray:
        """Compute payoff for each path."""
        pass

