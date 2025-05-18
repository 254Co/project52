"""Quick runtime checks – raise early if NaNs or Infs.

This module provides utility functions for validating numerical arrays and
ensuring they contain only finite values. These checks help catch numerical
issues early in the computation process.
"""

from __future__ import annotations

import numpy as np


def assert_finite(arr: np.ndarray, name: str = "array") -> None:
    """Assert that an array contains only finite values.
    
    This function checks if all elements in the input array are finite
    (not NaN, not Inf, not -Inf). If any non-finite values are found,
    it raises a ValueError with a descriptive message.
    
    Args:
        arr (np.ndarray): The array to check.
        name (str, optional): Name of the array for error messages.
            Defaults to "array".
            
    Raises:
        ValueError: If the array contains any non-finite values.
        
    Note:
        This function is used throughout the codebase to validate
        numerical computations and catch potential issues early.
    """
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non‑finite values!")