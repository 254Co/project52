"""
Numerical validation and stability checks for the Chen3 package.

This module provides functions for validating numerical stability and
parameter consistency in the Chen3 model. It includes checks for:
- Feller condition for mean-reverting processes
- Correlation matrix validity and positive definiteness
- Numerical stability of arrays and time series
- Time grid consistency for simulations

The module uses a combination of mathematical conditions and numerical
tolerance checks to ensure the stability and validity of calculations.
"""

from typing import List, Optional, Tuple, Union

import numpy as np

from .exceptions import NumericalError, ValidationError
from .logging import logger


def check_feller_condition(
    kappa: float, theta: float, sigma: float, process_name: str = "process"
) -> bool:
    """
    Check the Feller condition: 2κθ > σ² for mean-reverting processes.

    The Feller condition ensures that the variance process remains strictly
    positive in a mean-reverting process. This is a critical condition for
    the stability of the Chen3 model.

    Args:
        kappa (float): Mean reversion speed parameter
        theta (float): Long-term mean level parameter
        sigma (float): Volatility parameter
        process_name (str): Name of the process for logging purposes

    Returns:
        bool: True if the Feller condition is satisfied, False otherwise

    Note:
        If the condition is violated, a warning is logged with the specific
        values that caused the violation.
    """
    condition = 2 * kappa * theta > sigma * sigma
    if not condition:
        logger.warning(
            f"Feller condition violated for {process_name}: "
            f"2κθ = {2 * kappa * theta:.4f} ≤ σ² = {sigma * sigma:.4f}"
        )
    return condition


def validate_correlation_matrix(
    corr_matrix: np.ndarray, tolerance: float = 1e-10
) -> Tuple[bool, Optional[str]]:
    """
    Validate a correlation matrix for the Chen3 model.

    Performs comprehensive checks on the correlation matrix:
    1. Shape validation (must be 3x3)
    2. Symmetry check
    3. Diagonal elements check (must be 1.0)
    4. Positive definiteness check using Cholesky decomposition
    5. Correlation bounds check (-1 ≤ ρ ≤ 1)

    Args:
        corr_matrix (np.ndarray): The correlation matrix to validate
        tolerance (float): Numerical tolerance for floating-point comparisons

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing:
            - bool: True if matrix is valid, False otherwise
            - Optional[str]: Error message if invalid, None if valid

    Raises:
        ValidationError: If the input is not a numpy array
    """
    if not isinstance(corr_matrix, np.ndarray):
        raise ValidationError("Input must be a numpy array")

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
    name: str = "values",
    max_growth_rate: Optional[float] = None,
    max_volatility: Optional[float] = None,
) -> bool:
    """
    Check numerical stability of an array of values.

    For payoffs, allow zero values (valid for options). Only flag negative, NaN, or infinite values as unstable.
    """
    if np.any(np.isnan(values)):
        logger.error(f"NaN values detected in {name}")
        return False

    if np.any(np.isinf(values)):
        logger.error(f"Infinite values detected in {name}")
        return False

    if name == "payoffs":
        if np.any(values < 0):
            logger.warning(
                f"Negative values detected in {name}: min = {np.min(values):.4e}"
            )
            return False
    else:
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

    # Check growth rate if specified
    if max_growth_rate is not None:
        log_returns = np.log(np.abs(values[1:] / values[:-1]))
        growth_rate = np.mean(log_returns)
        if abs(growth_rate) > max_growth_rate:
            logger.warning(
                f"Growth rate exceeds maximum threshold in {name}: "
                f"rate = {abs(growth_rate):.4f} > {max_growth_rate:.4f}"
            )
            return False

    # Check volatility if specified
    if max_volatility is not None:
        log_returns = np.log(np.abs(values[1:] / values[:-1]))
        volatility = np.std(log_returns)
        if volatility > max_volatility:
            logger.warning(
                f"Volatility exceeds maximum threshold in {name}: "
                f"vol = {volatility:.4f} > {max_volatility:.4f}"
            )
            return False

    return True


def validate_time_grid(
    time_points: np.ndarray, min_dt: float = 1e-6, max_dt: float = 1.0
) -> Tuple[bool, Optional[str]]:
    """
    Validate a time grid for simulation purposes.

    Performs comprehensive checks on the time grid:
    1. Strictly increasing time points
    2. Minimum time step check
    3. Maximum time step check

    Args:
        time_points (np.ndarray): Array of time points
        min_dt (float): Minimum allowed time step (default: 1e-6)
        max_dt (float): Maximum allowed time step (default: 1.0)

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing:
            - bool: True if time grid is valid, False otherwise
            - Optional[str]: Error message if invalid, None if valid

    Note:
        The time grid must be strictly increasing to ensure proper
        simulation of the stochastic processes.
    """
    if not np.all(np.diff(time_points) > 0):
        return False, "Time points must be strictly increasing"

    dt = np.diff(time_points)
    if np.any(dt < min_dt):
        return False, f"Time steps must be at least {min_dt}"

    if np.any(dt > max_dt):
        return False, f"Time steps must be at most {max_dt}"

    return True, None
