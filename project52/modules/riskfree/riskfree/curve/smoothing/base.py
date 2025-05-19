"""Base classes and registry for yield curve smoothing methods.

This module provides the abstract base class for all yield curve smoothing
methods and a registry system for managing different smoothing implementations.
The base class defines the common interface that all smoothing methods must
implement.

The module implements:
1. Abstract BaseSmoother class with required methods
2. Registry system for managing smoothing implementations
3. Factory function for creating smoother instances
"""

from abc import ABC, abstractmethod
from typing import Union, Sequence
import numpy as np

# Type alias for array-like inputs (lists, tuples, or numpy arrays)
ArrayLike = Union[Sequence[float], "np.ndarray"]


class BaseSmoother(ABC):
    """Abstract base class for all yield curve smoothing methods.
    
    This class defines the interface that all smoothing methods must implement.
    It provides common functionality and enforces the contract that subclasses
    must follow.
    
    The interface requires:
    1. A fit() method to learn parameters from data
    2. A predict() method to generate smoothed rates
    3. State tracking to ensure proper usage order
    
    Attributes:
        _fitted (bool): Whether the smoother has been fitted to data
    """

    def __init__(self):
        """Initialize a new smoother instance."""
        self._fitted = False

    @abstractmethod
    def fit(self, times: ArrayLike, yields: ArrayLike):
        """Fit the smoother to observed yield data.
        
        Args:
            times: Array of tenors in years
            yields: Array of observed yields (same length as times)
        """
        ...

    @abstractmethod
    def predict(self, times: ArrayLike) -> np.ndarray:
        """Generate smoothed yields for given tenors.
        
        Args:
            times: Array of tenors to predict yields for
            
        Returns:
            Array of predicted yields
            
        Raises:
            RuntimeError: If called before fit()
        """
        ...

    # -------------------------------------------------

    def _check(self):
        """Verify that the smoother has been fitted.
        
        Raises:
            RuntimeError: If called before fit()
        """
        if not self._fitted:
            raise RuntimeError("Smoother.fit(..) must be called first")


# Registry for managing smoothing implementations
_REGISTRY = {}


def register(name: str):
    """Class decorator to register smoothing implementations.
    
    This decorator adds a smoothing class to the registry, making it
    available through the get_smoother() factory function.
    
    Args:
        name: Unique identifier for the smoothing method
        
    Returns:
        Decorated class
        
    Example:
        @register("nelson_siegel")
        class NelsonSiegelSmoother(BaseSmoother):
            ...
    """
    def _wrap(cls):
        _REGISTRY[name.lower()] = cls
        return cls
    return _wrap


def get_smoother(name: str, **kwargs) -> "BaseSmoother":
    """Create a new instance of a registered smoothing method.
    
    This factory function creates and returns an instance of the requested
    smoothing method. The method must have been previously registered using
    the @register decorator.
    
    Args:
        name: Name of the smoothing method (case-insensitive)
        **kwargs: Additional arguments to pass to the smoother's constructor
        
    Returns:
        New instance of the requested smoother
        
    Raises:
        ValueError: If the requested method is not registered
        
    Example:
        >>> smoother = get_smoother("nelson_siegel", tau=2.0)
        >>> smoother.fit(times, yields)
        >>> smoothed = smoother.predict(new_times)
    """
    try:
        cls = _REGISTRY[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown smoother '{name}'. "
                         f"Available: {list(_REGISTRY)}") from exc
    return cls(**kwargs)
