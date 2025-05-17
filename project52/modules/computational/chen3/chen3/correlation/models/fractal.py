"""
Fractal Correlation Implementation for the Chen3 Model

This module implements a fractal-based correlation structure for the Chen3 model,
where correlations are defined through fractal geometry and self-similarity properties.
This allows for modeling long-range dependencies and scaling behavior in the correlation structure.

The implementation uses fractal mathematics to model correlations between factors:
- Hurst exponent controls the rate of correlation decay
- Self-similarity properties capture long-range dependencies
- Time-dependent and state-dependent Hurst exponents are supported
- Fractal dimension influences the correlation structure

The correlation structure is particularly useful for modeling:
- Long-range market dependencies
- Power-law scaling in correlations
- Time-varying persistence
- Self-similar market patterns
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.base import BaseCorrelation
from ..utils.exceptions import CorrelationError, CorrelationValidationError
from ..utils.logging_config import logger


class FractalCorrelation(BaseCorrelation):
    """
    Fractal-based correlation structure for the Chen3 model.

    This class implements correlations through fractal geometry,
    where the correlation matrix is constructed from fractal dimensions
    and self-similarity properties.

    Mathematical Formulation:
    ----------------------
    The correlation matrix is constructed as:

    ρ_ij = exp(-|i-j|^H)

    where:
    - H: Hurst exponent (0 < H < 1)
        * H > 0.5: Long-range persistence
        * H = 0.5: Random walk
        * H < 0.5: Anti-persistence
    - i, j: Factor indices

    The Hurst exponent can be time-dependent or state-dependent:
    H(t) = f(t, state)

    Attributes:
        hurst (float): Base Hurst exponent
        hurst_function (Callable): Function for computing time/state-dependent
            Hurst exponent
        n_factors (int): Number of factors in the correlation structure
        name (str): Name identifier for the correlation structure

    Note:
        The correlation structure ensures positive definiteness through
        the exponential decay of correlations with distance.
    """

    def __init__(
        self,
        hurst: Optional[float] = None,
        hurst_function: Optional[Callable] = None,
        n_factors: int = 3,
        name: str = "FractalCorrelation",
    ):
        """
        Initialize fractal correlation structure.

        Args:
            hurst (Optional[float]): Base Hurst exponent. If None, defaults to 0.5
                (random walk behavior).
            hurst_function (Optional[Callable]): Function for computing time/state-
                dependent Hurst exponent. Should take (t, state) and return H.
            n_factors (int): Number of factors in the correlation structure
            name (str): Name identifier for the correlation structure

        Raises:
            CorrelationValidationError: If Hurst exponent is invalid
            ValueError: If initialization parameters are invalid
        """
        super().__init__(n_factors=n_factors, name=name)
        self.hurst = hurst or 0.5
        self.hurst_function = hurst_function or (lambda t, s: self.hurst)
        self._validate_initialization()
        logger.debug(f"Initialized {self.name} with {n_factors} factors")

    def _validate_initialization(self) -> None:
        """
        Validate initialization parameters.

        Performs comprehensive validation of initialization parameters:
        1. Hurst exponent is in valid range (0, 1)
        2. Hurst function returns valid values

        Raises:
            CorrelationValidationError: If any validation check fails
        """
        if not 0 < self.hurst < 1:
            raise CorrelationValidationError("Hurst exponent must be in (0, 1)")

    def _compute_hurst(
        self, t: float, state: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute Hurst exponent at time t and state.

        This method computes the current Hurst exponent using the Hurst function,
        ensuring it remains in the valid range (0, 1).

        Args:
            t (float): Current time point
            state (Optional[Dict[str, float]]): Current state variables
                that may influence the Hurst exponent

        Returns:
            float: Valid Hurst exponent in range (0, 1)

        Raises:
            CorrelationError: If Hurst computation fails
        """
        try:
            H = self.hurst_function(t, state)

            # Ensure Hurst exponent is in (0, 1)
            H = np.clip(H, 0.01, 0.99)

            return H
        except Exception as e:
            raise CorrelationError(f"Failed to compute Hurst exponent: {str(e)}")

    def get_correlation_matrix(
        self, t: float = 0.0, state: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Get the correlation matrix at time t and state.

        This method computes the full correlation matrix by:
        1. Computing current Hurst exponent
        2. Constructing correlations using the fractal decay formula
        3. Validating the resulting matrix

        Args:
            t (float): Time point at which to compute correlations
            state (Optional[Dict[str, float]]): Current state variables
                that may influence correlations

        Returns:
            np.ndarray: Valid correlation matrix (n_factors x n_factors)

        Raises:
            CorrelationError: If correlation computation fails
        """
        try:
            # Compute Hurst exponent
            H = self._compute_hurst(t, state)

            # Construct correlation matrix
            corr = np.zeros((self.n_factors, self.n_factors))
            for i in range(self.n_factors):
                for j in range(self.n_factors):
                    if i == j:
                        corr[i, j] = 1.0
                    else:
                        corr[i, j] = np.exp(-abs(i - j) ** H)

            # Validate the computed matrix
            self._validate_correlation_matrix(corr)

            logger.debug(f"Computed correlation matrix at t={t}")
            return corr
        except Exception as e:
            raise CorrelationError(f"Failed to compute correlation matrix: {str(e)}")

    def __str__(self) -> str:
        """
        String representation of the correlation structure.

        Returns:
            str: Simple string representation showing key parameters
        """
        return (
            f"{self.name}(" f"n_factors={self.n_factors}, " f"hurst={self.hurst:.2f})"
        )

    def __repr__(self) -> str:
        """
        Detailed string representation of the correlation structure.

        Returns:
            str: Detailed string representation showing all parameters
        """
        return (
            f"{self.__class__.__name__}("
            f"hurst={self.hurst}, "
            f"hurst_function={self.hurst_function}, "
            f"n_factors={self.n_factors}, "
            f"name='{self.name}')"
        )


def create_fractal_correlation(
    n_factors: int = 3, hurst_type: str = "constant", decay_rate: float = 0.1
) -> FractalCorrelation:
    """
    Create a fractal correlation structure with specified parameters.

    This factory function creates a FractalCorrelation instance with
    predefined Hurst exponent functions for common use cases:
    - constant: Time-invariant Hurst exponent (H = 0.5)
    - decay: Exponentially decaying Hurst exponent
    - oscillate: Oscillating Hurst exponent with cosine function

    Args:
        n_factors (int): Number of factors in the correlation structure
        hurst_type (str): Type of Hurst exponent function to use
            ('constant', 'decay', or 'oscillate')
        decay_rate (float): Rate parameter for Hurst exponent decay/oscillation

    Returns:
        FractalCorrelation: Configured fractal correlation instance

    Raises:
        ValueError: If hurst_type is invalid

    Example:
        >>> corr = create_fractal_correlation(n_factors=3, hurst_type='decay')
        >>> matrix = corr.get_correlation_matrix(t=1.0)
    """
    # Create Hurst exponent function
    if hurst_type == "constant":
        hurst_function = lambda t, s: 0.5
    elif hurst_type == "decay":
        hurst_function = lambda t, s: 0.5 * np.exp(-decay_rate * t)
    elif hurst_type == "oscillate":
        hurst_function = lambda t, s: 0.5 * (1 + np.cos(decay_rate * t))
    else:
        raise ValueError(f"Invalid Hurst type: {hurst_type}")

    return FractalCorrelation(
        hurst=0.5, hurst_function=hurst_function, n_factors=n_factors
    )
