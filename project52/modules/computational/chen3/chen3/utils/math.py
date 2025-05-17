# File: chen3/utils/math.py
"""
Mathematical utilities for the Chen3 package.

This module provides mathematical helper functions commonly used in
financial calculations and numerical computations. It includes functions
for discounting and safe mathematical operations.
"""

from typing import Union

import numpy as np


def incurred_discount(rate: float, t: float) -> float:
    """
    Calculate the discount factor for a given rate and time.

    This function computes the present value of a unit payment at time t,
    using the formula: (1 + rate)^(-t)

    Args:
        rate (float): The discount rate (e.g., 0.05 for 5%)
        t (float): The time in years

    Returns:
        float: The discount factor

    Example:
        >>> incurred_discount(0.05, 1.0)
        0.9523809523809523
    """
    return (1 + rate) ** (-t)


def safe_sqrt(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the square root of a number or array safely.

    This function handles negative inputs by returning 0.0 instead of
    raising an error. It works with both scalar values and numpy arrays.

    Args:
        x (Union[float, np.ndarray]): Input value or array

    Returns:
        Union[float, np.ndarray]: Square root of x if x >= 0, 0.0 otherwise

    Example:
        >>> safe_sqrt(4.0)
        2.0
        >>> safe_sqrt(-1.0)
        0.0
    """
    return x**0.5 if x >= 0 else 0.0
