"""
Numerical validation and stability checks for the Chen3 package.

This module provides functions for validating numerical stability and
parameter consistency in the Chen3 model.
"""

import numpy as np
from typing import Optional, Tuple
from .exceptions import ValidationError, NumericalError
from .logging import logger

def check_feller_condition(
    kappa: float,
    theta: float,
    sigma: float,
    process_name: str = "process"
) -> bool:
    """
    Check the Feller condition: 2κθ > σ²
    
    Args:
        kappa: Mean reversion speed
        theta: Long-term mean
        sigma: Volatility
        process_name: Name of the process for logging
    
    Returns:
        bool: True if condition is satisfied, False otherwise
    """
    condition = 2 * kappa * theta > sigma * sigma
    if not condition:
        logger.warning(
            f"Feller condition violated for {process_name}: "
            f"2κθ = {2 * kappa * theta:.4f} ≤ σ² = {sigma * sigma:.4f}"
        )
    return condition

def validate_correlation_matrix(
    corr_matrix: np.ndarray,
    tolerance: float = 1e-10
) -> Tuple[bool, Optional[str]]:
    """
    Validate a correlation matrix.
    
    Args:
        corr_matrix: The correlation matrix to validate
        tolerance: Numerical tolerance for checks
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    # Check shape
    if corr_matrix.shape != (3, 3):
        return False, f"Correlation matrix must be 3x3, got shape {corr_matrix.shape}"
    
    # Check symmetry
    if not np.allclose(corr_matrix, corr_matrix.T, rtol=tolerance):
        return False, "Correlation matrix must be symmetric"
    
    # Check diagonal elements
    if not np.allclose(np.diag(corr_matrix), 1.0, rtol=tolerance):
        return False, "Diagonal elements must be 1.0"
    
    # Check positive definiteness
    try:
        np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        return False, "Correlation matrix must be positive definite"
    
    # Check bounds
    if not np.all(np.abs(corr_matrix) <= 1.0 + tolerance):
        return False, "Correlation values must be between -1 and 1"
    
    return True, None

def check_numerical_stability(
    values: np.ndarray,
    min_value: float = 1e-10,
    max_value: float = 1e10,
    name: str = "values"
) -> bool:
    """
    Check numerical stability of an array of values.
    
    Args:
        values: Array of values to check
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        name: Name of the values for logging
    
    Returns:
        bool: True if values are stable, False otherwise
    """
    if np.any(np.isnan(values)):
        logger.error(f"NaN values detected in {name}")
        return False
    
    if np.any(np.isinf(values)):
        logger.error(f"Infinite values detected in {name}")
        return False
    
    if np.any(values < min_value):
        logger.warning(
            f"Values below minimum threshold detected in {name}: "
            f"min = {np.min(values):.4e} < {min_value:.4e}"
        )
        return False
    
    if np.any(values > max_value):
        logger.warning(
            f"Values above maximum threshold detected in {name}: "
            f"max = {np.max(values):.4e} > {max_value:.4e}"
        )
        return False
    
    return True

def validate_time_grid(
    time_points: np.ndarray,
    min_dt: float = 1e-6,
    max_dt: float = 1.0
) -> Tuple[bool, Optional[str]]:
    """
    Validate a time grid for simulation.
    
    Args:
        time_points: Array of time points
        min_dt: Minimum allowed time step
        max_dt: Maximum allowed time step
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if not np.all(np.diff(time_points) > 0):
        return False, "Time points must be strictly increasing"
    
    dt = np.diff(time_points)
    if np.any(dt < min_dt):
        return False, f"Time steps must be at least {min_dt}"
    
    if np.any(dt > max_dt):
        return False, f"Time steps must be at most {max_dt}"
    
    return True, None 