# File: chen3/processes/base.py
"""
Base Classes for Stochastic Processes

This module defines the abstract base class for stochastic processes used in the
three-factor Chen model. It provides the interface that all concrete process
implementations must follow, ensuring consistent behavior across different
stochastic components of the model.

The module focuses on:
1. Abstract base class definition for state processes
2. Interface for drift and diffusion terms
3. Type hints for numerical computations
"""

from abc import ABC, abstractmethod
from numpy.typing import NDArray


class StateProcess(ABC):
    """
    Abstract base class for stochastic state processes.
    
    This class defines the interface that all stochastic processes in the model
    must implement. It enforces the structure for drift and diffusion terms,
    which are essential components of any stochastic differential equation (SDE).
    
    The class is designed to work with numpy arrays for efficient numerical
    computation and supports vectorized operations for Monte Carlo simulations.
    
    Example:
        To implement a new stochastic process:
        >>> class MyProcess(StateProcess):
        ...     def drift(self, state: NDArray) -> NDArray:
        ...         # Implement drift term
        ...         return drift_term
        ...     
        ...     def diffusion(self, state: NDArray) -> NDArray:
        ...         # Implement diffusion term
        ...         return diffusion_term
    """
    
    @abstractmethod
    def drift(self, state: NDArray) -> NDArray:
        """
        Compute the drift term of the stochastic process.
        
        The drift term represents the deterministic component of the SDE:
            dX_t = μ(X_t)dt + σ(X_t)dW_t
        where μ(X_t) is the drift term.
        
        Args:
            state (NDArray): Current state of the process, shape (n_paths,)
                            or (n_paths, n_dims) for multi-dimensional processes
        
        Returns:
            NDArray: Drift term evaluated at the current state, same shape as input
        
        Raises:
            NotImplementedError: Must be implemented by concrete subclasses
        """
        pass

    @abstractmethod
    def diffusion(self, state: NDArray) -> NDArray:
        """
        Compute the diffusion term of the stochastic process.
        
        The diffusion term represents the stochastic component of the SDE:
            dX_t = μ(X_t)dt + σ(X_t)dW_t
        where σ(X_t) is the diffusion term.
        
        Args:
            state (NDArray): Current state of the process, shape (n_paths,)
                            or (n_paths, n_dims) for multi-dimensional processes
        
        Returns:
            NDArray: Diffusion term evaluated at the current state, same shape as input
        
        Raises:
            NotImplementedError: Must be implemented by concrete subclasses
        """
        pass

