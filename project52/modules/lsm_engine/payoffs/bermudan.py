# File: lsm_engine/payoffs/bermudan.py
"""
Bermudan Option Payoff Generators

This module provides payoff functions for Bermudan options, which can be exercised
at a discrete set of dates before maturity. The payoff functions are designed to
work with the Longstaff-Schwartz Method (LSM) engine for option pricing.

Available payoff functions:
- Bermudan put: max(K - S, 0) at each exercise date
- Bermudan call: max(S - K, 0) at each exercise date

Note that while the payoff functions are similar to American options, the exercise
scheduling (which dates exercise is allowed) is handled by the LSM engine through
the exercise_idx parameter. This allows for flexible exercise schedules while
keeping the payoff functions simple.
"""
import numpy as np
from typing import Callable
from .vanilla import _validate_paths


def bermudan_put(strike: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a Bermudan put option payoff function.
    
    The Bermudan put option gives the holder the right to sell the underlying
    asset at the strike price K at specific exercise dates before or at maturity.
    The payoff at each exercise date is max(K - S, 0), where S is the asset price.
    
    Args:
        strike: Strike price K of the option
        
    Returns:
        A function that takes price paths and returns the payoff matrix:
        - Input: paths array of shape (n_paths, n_steps)
        - Output: payoff array of shape (n_paths, n_steps)
        - Payoffs are computed at all time steps, but exercise is only allowed
          at dates specified by exercise_idx in the LSM engine
        
    Example:
        >>> payoff_fn = bermudan_put(100.0)
        >>> payoffs = payoff_fn(paths)  # shape (n_paths, n_steps)
    """
    def payoff(paths: np.ndarray) -> np.ndarray:
        _validate_paths(paths)
        return np.maximum(strike - paths, 0.0)
    return payoff


def bermudan_call(strike: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a Bermudan call option payoff function.
    
    The Bermudan call option gives the holder the right to buy the underlying
    asset at the strike price K at specific exercise dates before or at maturity.
    The payoff at each exercise date is max(S - K, 0), where S is the asset price.
    
    Args:
        strike: Strike price K of the option
        
    Returns:
        A function that takes price paths and returns the payoff matrix:
        - Input: paths array of shape (n_paths, n_steps)
        - Output: payoff array of shape (n_paths, n_steps)
        - Payoffs are computed at all time steps, but exercise is only allowed
          at dates specified by exercise_idx in the LSM engine
    """
    def payoff(paths: np.ndarray) -> np.ndarray:
        _validate_paths(paths)
        return np.maximum(paths - strike, 0.0)
    return payoff


__all__ = [
    "bermudan_put",
    "bermudan_call",
]