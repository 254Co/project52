# -------------------- correlation.py --------------------
"""
Correlation Handling Module for the Chen Model

This module provides the core functionality for handling correlations between
the three factors in the Chen model: interest rates, equity prices, and volatility.
It implements the Cholesky decomposition method for generating correlated random
variables, which is essential for simulating correlated stochastic processes.

The module supports:
1. Correlation Matrix Validation:
   - Symmetry check
   - Positive definiteness
   - Valid correlation values [-1, 1]

2. Cholesky Decomposition:
   - Efficient computation of correlation factors
   - Numerical stability
   - Error handling

3. Correlated Random Variables:
   - Generation of correlated Brownian motions
   - Support for multiple dimensions
   - Efficient matrix operations

Mathematical Formulation:
------------------------
1. Correlation Matrix:
   ρ = [ρ_ij] where:
   - ρ_ij: correlation between factors i and j
   - ρ_ii = 1 (diagonal elements)
   - ρ_ij = ρ_ji (symmetry)
   - |ρ_ij| ≤ 1 (valid correlation)

2. Cholesky Decomposition:
   ρ = LL^T where:
   - L: lower triangular matrix
   - L^T: transpose of L
   - L_ii > 0 (positive diagonal)

3. Correlated Random Variables:
   X = LZ where:
   - X: correlated random variables
   - Z: independent random variables
   - L: Cholesky factor

Example Usage:
-------------
    >>> from chen3.correlation import cholesky_correlation
    >>> import numpy as np
    >>>
    >>> # Define correlation matrix
    >>> rho = np.array([
    ...     [1.0, 0.5, 0.3],
    ...     [0.5, 1.0, 0.2],
    ...     [0.3, 0.2, 1.0]
    ... ])
    >>>
    >>> # Get Cholesky factor
    >>> L = cholesky_correlation(rho)
    >>>
    >>> # Generate correlated random variables
    >>> n_paths = 1000
    >>> Z = np.random.normal(0, 1, size=(n_paths, 3))
    >>> X = Z @ L.T
    >>>
    >>> # Verify correlation
    >>> corr = np.corrcoef(X.T)
    >>> print(np.allclose(corr, rho, atol=1e-2))
"""

import numpy as np
from typing import Tuple

def cholesky_correlation(rho: np.ndarray) -> np.ndarray:
    """
    Compute the Cholesky decomposition of a correlation matrix.
    
    This function performs the following steps:
    1. Validates the input correlation matrix
    2. Computes the Cholesky decomposition
    3. Returns the lower triangular factor
    
    Mathematical Properties:
    ----------------------
    1. Input Matrix:
       - Must be symmetric
       - Must be positive definite
       - Diagonal elements must be 1
       - Off-diagonal elements in [-1, 1]
    
    2. Cholesky Decomposition:
       ρ = LL^T where:
       - L: lower triangular matrix
       - L_ii > 0 (positive diagonal)
       - L_ij = 0 for i < j
    
    3. Numerical Stability:
       - Uses numpy's cholesky implementation
       - Handles numerical errors
       - Ensures positive definiteness
    
    Args:
        rho (np.ndarray): Correlation matrix
            Shape: (n, n) where n is the number of factors
            Must be symmetric and positive definite
    
    Returns:
        np.ndarray: Cholesky factor L
            Shape: (n, n)
            Lower triangular matrix
    
    Raises:
        ValueError: If matrix is not symmetric
        ValueError: If matrix is not positive definite
        ValueError: If matrix has invalid correlation values
    
    Example:
        >>> rho = np.array([
        ...     [1.0, 0.5, 0.3],
        ...     [0.5, 1.0, 0.2],
        ...     [0.3, 0.2, 1.0]
        ... ])
        >>> L = cholesky_correlation(rho)
        >>> # Verify decomposition
        >>> np.allclose(rho, L @ L.T)
        True
    """
    # Validate input
    if not np.allclose(rho, rho.T):
        raise ValueError("Correlation matrix must be symmetric")
    
    if not np.all(np.diag(rho) == 1.0):
        raise ValueError("Correlation matrix diagonal must be 1.0")
    
    if not np.all(np.abs(rho) <= 1.0):
        raise ValueError("Correlation values must be in [-1, 1]")
    
    try:
        # Compute Cholesky decomposition
        L = np.linalg.cholesky(rho)
        return L
    except np.linalg.LinAlgError:
        raise ValueError("Correlation matrix must be positive definite")