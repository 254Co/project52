"""Small helpers independent of NumPy versions.

This module provides mathematical utility functions that are version-independent
and commonly used throughout the codebase. These functions include basic
statistical operations and mathematical transformations.
"""
from __future__ import annotations

import numpy as np

__all__ = ["moving_average", "logistic"]


def moving_average(x: np.ndarray, n: int = 5) -> np.ndarray:
    """Compute a simple moving average using convolution.
    
    This function calculates the n-period moving average of a time series
    using a convolution operation. The moving average is computed by taking
    the mean of n consecutive values, sliding the window one step at a time.
    
    Args:
        x (np.ndarray): Input time series array.
        n (int, optional): Window size for the moving average.
            Must be greater than 1. Defaults to 5.
            
    Returns:
        np.ndarray: Array of moving averages with length len(x) - n + 1.
        
    Note:
        - If n <= 1, returns a copy of the input array.
        - The output length is shorter than the input by n-1 elements
          due to the 'valid' convolution mode.
        - Uses uniform weights (1/n) for each element in the window.
    """
    if n <= 1:
        return x.copy()
    return np.convolve(x, np.ones(n) / n, mode="valid")


def logistic(z: np.ndarray | float) -> np.ndarray | float:
    """Compute the logistic (sigmoid) function.
    
    The logistic function is defined as:
        f(z) = 1 / (1 + exp(-z))
    
    This function maps any real number to the interval (0,1), making it
    useful for probability calculations and smooth transitions.
    
    Args:
        z (np.ndarray | float): Input value(s).
            
    Returns:
        np.ndarray | float: Output value(s) in the range (0,1).
            Returns the same type as the input.
            
    Note:
        The function is vectorized and works with both scalar and array inputs.
        For large negative inputs, the output approaches 0.
        For large positive inputs, the output approaches 1.
    """
    return 1.0 / (1.0 + np.exp(-z))