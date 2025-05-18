# File: lsm_engine/payoffs/vanilla.py
"""
Vanilla Option Payoff Generators

This module provides payoff functions for plain-vanilla American and European options.
The payoff functions are designed to work with the Longstaff-Schwartz Method (LSM)
engine for option pricing.

Available payoff functions:
- American put: max(K - S, 0) at each exercise date
- American call: max(S - K, 0) at each exercise date
- European put: max(K - S, 0) at maturity only
- European call: max(S - K, 0) at maturity only

Each function returns a callable that takes a price path array and returns
the corresponding payoff matrix.
"""
import numpy as np
from typing import Callable


def _validate_paths(paths: np.ndarray) -> None:
    """
    Validate the input price paths array.
    
    This helper function ensures that the input array:
    1. Is a numpy array
    2. Has exactly 2 dimensions (paths × time steps)
    3. Contains numeric values
    
    Args:
        paths: Array of price paths of shape (n_paths, n_steps)
        
    Raises:
        TypeError: If paths is not a numpy array
        ValueError: If paths does not have 2 dimensions
    """
    if not isinstance(paths, np.ndarray):
        raise TypeError(f"paths must be a numpy array, got {type(paths)}")
    if paths.ndim != 2:
        raise ValueError(f"paths must be 2D array (n_paths, n_steps), got shape {paths.shape}")


def american_put(strike: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create an American put option payoff function.
    
    The American put option gives the holder the right to sell the underlying
    asset at the strike price K at any time before or at maturity. The payoff
    at each time step is max(K - S, 0), where S is the asset price.
    
    Args:
        strike: Strike price K of the option
        
    Returns:
        A function that takes price paths and returns the payoff matrix:
        - Input: paths array of shape (n_paths, n_steps)
        - Output: payoff array of shape (n_paths, n_steps)
        
    Example:
        >>> payoff_fn = american_put(100.0)
        >>> payoffs = payoff_fn(paths)  # shape (n_paths, n_steps)
    """
    def payoff(paths: np.ndarray) -> np.ndarray:
        _validate_paths(paths)
        return np.maximum(strike - paths, 0.0)
    return payoff


def american_call(strike: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create an American call option payoff function.
    
    The American call option gives the holder the right to buy the underlying
    asset at the strike price K at any time before or at maturity. The payoff
    at each time step is max(S - K, 0), where S is the asset price.
    
    Args:
        strike: Strike price K of the option
        
    Returns:
        A function that takes price paths and returns the payoff matrix:
        - Input: paths array of shape (n_paths, n_steps)
        - Output: payoff array of shape (n_paths, n_steps)
    """
    def payoff(paths: np.ndarray) -> np.ndarray:
        _validate_paths(paths)
        return np.maximum(paths - strike, 0.0)
    return payoff


def european_put(strike: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a European put option payoff function.
    
    The European put option gives the holder the right to sell the underlying
    asset at the strike price K only at maturity. The payoff is max(K - S, 0)
    at maturity and zero at all other times.
    
    Args:
        strike: Strike price K of the option
        
    Returns:
        A function that takes price paths and returns the payoff matrix:
        - Input: paths array of shape (n_paths, n_steps)
        - Output: payoff array of shape (n_paths, n_steps)
        - All entries are zero except at maturity
    """
    def payoff(paths: np.ndarray) -> np.ndarray:
        _validate_paths(paths)
        pay = np.zeros_like(paths)
        pay[:, -1] = np.maximum(strike - paths[:, -1], 0.0)
        return pay
    return payoff


def european_call(strike: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a European call option payoff function.
    
    The European call option gives the holder the right to buy the underlying
    asset at the strike price K only at maturity. The payoff is max(S - K, 0)
    at maturity and zero at all other times.
    
    Args:
        strike: Strike price K of the option
        
    Returns:
        A function that takes price paths and returns the payoff matrix:
        - Input: paths array of shape (n_paths, n_steps)
        - Output: payoff array of shape (n_paths, n_steps)
        - All entries are zero except at maturity
    """
    def payoff(paths: np.ndarray) -> np.ndarray:
        _validate_paths(paths)
        pay = np.zeros_like(paths)
        pay[:, -1] = np.maximum(paths[:, -1] - strike, 0.0)
        return pay
    return payoff


__all__ = [
    "american_put",
    "american_call",
    "european_put",
    "european_call",
]