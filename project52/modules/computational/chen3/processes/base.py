# File: chen3/processes/base.py
"""Abstract base for state processes."""
from abc import ABC, abstractmethod
from numpy.typing import NDArray

class StateProcess(ABC):
    @abstractmethod
    def drift(self, state: NDArray) -> NDArray:
        pass

    @abstractmethod
    def diffusion(self, state: NDArray) -> NDArray:
        pass

