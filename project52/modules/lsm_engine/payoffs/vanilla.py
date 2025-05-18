"""
Vanilla option payoff functions for the LSM Engine.

This module provides payoff functions for vanilla options (e.g., calls, puts)
that can be used with the LSM pricing engine.
"""

# File: lsm_engine/payoffs/vanilla.py
import numpy as np

def american_put(strike: float):
    """
    Create a payoff function for an American put option.
    
    This function returns a payoff function that computes the intrinsic value
    of an American put option at each time step. The payoff is:
    max(K - S, 0) where K is the strike price and S is the asset price.
    
    Args:
        strike: Strike price of the put option
        
    Returns:
        Callable: A function that computes put option payoffs for given paths
        
    Example:
        >>> payoff_fn = american_put(strike=100.0)
        >>> payoffs = payoff_fn(paths)  # paths shape: (n_paths, n_steps)
    """
    def payoff(paths: np.ndarray) -> np.ndarray:
        """
        Compute put option payoffs for given paths.
        
        Args:
            paths: Array of shape (n_paths, n_steps) containing asset price paths
            
        Returns:
            np.ndarray: Array of shape (n_paths, n_steps) containing option payoffs
        """
        return np.maximum(strike - paths, 0.0)
    return payoff