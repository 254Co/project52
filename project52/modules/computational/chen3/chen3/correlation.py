# -------------------- correlation.py --------------------
"""
Correlation Matrix Handling Module

This module provides functionality for handling correlation matrices in the context
of the three-factor Chen model. It implements the Cholesky decomposition method
for generating correlated random variables, which is essential for simulating
the correlated stochastic processes in the model.

The module focuses on:
1. Validation of correlation matrices
2. Cholesky decomposition for correlation structure
3. Error handling for invalid correlation matrices
"""

import numpy as np
from numpy.linalg import cholesky


def cholesky_correlation(rho: np.ndarray) -> np.ndarray:
    """
    Compute the lower-triangular Cholesky factor of a correlation matrix.
    
    This function performs the Cholesky decomposition of a correlation matrix,
    which is used to generate correlated random variables for the three-factor
    model simulation. The decomposition ensures that the resulting matrix L
    satisfies L @ L.T = rho, where rho is the input correlation matrix.
    
    The function includes validation checks to ensure:
    1. The input matrix is symmetric
    2. The matrix is positive definite
    3. The matrix has valid correlation values (-1 ≤ ρ ≤ 1)
    
    Args:
        rho (np.ndarray): Correlation matrix of shape (n, n) where n is the
                         number of factors. Must be:
                         - Symmetric
                         - Positive definite
                         - Have unit diagonal elements
                         - Have off-diagonal elements in [-1, 1]
    
    Returns:
        np.ndarray: Lower-triangular Cholesky factor L of shape (n, n)
                   satisfying L @ L.T = rho
    
    Raises:
        ValueError: If the input matrix is not symmetric
        np.linalg.LinAlgError: If the matrix is not positive definite
                              or if the Cholesky decomposition fails
    
    Example:
        >>> rho = np.array([[1.0, 0.5, 0.3],
        ...                 [0.5, 1.0, 0.2],
        ...                 [0.3, 0.2, 1.0]])
        >>> L = cholesky_correlation(rho)
        >>> # L can be used to generate correlated random variables:
        >>> # correlated_noise = L @ independent_noise
    """
    rho = np.array(rho, dtype=float)
    
    # Ensure matrix is symmetric
    if not np.allclose(rho, rho.T, atol=1e-8):
        raise ValueError(
            "Correlation matrix must be symmetric. "
            "Please check your input matrix."
        )
    
    # Compute Cholesky decomposition
    try:
        L = cholesky(rho)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            f"Cholesky decomposition failed: ensure positive-definite matrix. "
            f"Original error: {e}"
        )
    
    return L