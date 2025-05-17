"""
State-Dependent Correlation Implementation

This module implements state-dependent correlation structures for the Chen3 model,
allowing correlations to vary based on the current state of the model.
"""

from typing import Any, Callable, Dict, Optional

import numpy as np

from ..core.base import BaseCorrelation
from ..utils.exceptions import CorrelationError, CorrelationValidationError
from ..utils.logging_config import logger


class StateDependentCorrelation(BaseCorrelation):
    """
    State-dependent correlation structure.

    This class implements correlations that depend on the current state
    of the model, such as interest rates, stock prices, or volatility.

    Mathematical Formulation:
    ----------------------
    The correlation matrix at time t is determined by the current state:

    ρ(t) = f(state(t))

    where:
    - state(t): Current state variables
    - f: Correlation function mapping state to correlation matrix

    Attributes:
        correlation_function (Callable): Function that computes correlation matrix
        default_state (Dict[str, float]): Default state values
        n_factors (int): Number of factors
        name (str): Name of the correlation structure
    """

    def __init__(
        self,
        correlation_function: Callable[[Dict[str, float]], np.ndarray],
        default_state: Dict[str, float],
        n_factors: int = 3,
        name: str = "StateDependentCorrelation",
    ):
        """
        Initialize state-dependent correlation.

        Args:
            correlation_function: Function that computes correlation matrix
            default_state: Default state values
            n_factors: Number of factors
            name: Name of the correlation structure

        Raises:
            CorrelationValidationError: If initialization fails
        """
        self.correlation_function = correlation_function
        self.default_state = default_state
        super().__init__(n_factors=n_factors, name=name)
        self._validate_initialization()
        logger.debug(f"Initialized {self.name} with {n_factors} factors")

    def _validate_initialization(self):
        """
        Validate initialization parameters.

        Raises:
            CorrelationValidationError: If validation fails
        """
        if not callable(self.correlation_function):
            raise CorrelationValidationError("Correlation function must be callable")

        try:
            corr = self.correlation_function(self.default_state)
            self._validate_correlation_matrix(corr)
        except Exception as e:
            raise CorrelationValidationError(
                f"Invalid correlation function or default state: {str(e)}"
            )

    def get_correlation_matrix(
        self, t: float = 0.0, state: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Get the correlation matrix at time t and state.

        Args:
            t: Time point (not used in this implementation)
            state: Current state variables

        Returns:
            Correlation matrix (n_factors x n_factors)

        Raises:
            CorrelationError: If computation fails
        """
        try:
            # Use provided state or default state
            current_state = state or self.default_state

            # Compute correlation matrix
            corr = self.correlation_function(current_state)

            # Validate the computed matrix
            self._validate_correlation_matrix(corr)

            logger.debug(f"Computed correlation matrix for state {current_state}")
            return corr
        except Exception as e:
            raise CorrelationError(f"Failed to compute correlation matrix: {str(e)}")

    def __str__(self) -> str:
        """String representation of the correlation structure."""
        return f"{self.name}(n_factors={self.n_factors})"

    def __repr__(self) -> str:
        """Detailed string representation of the correlation structure."""
        return (
            f"{self.__class__.__name__}("
            f"correlation_function={self.correlation_function}, "
            f"default_state={self.default_state}, "
            f"n_factors={self.n_factors}, "
            f"name='{self.name}')"
        )


def linear_state_correlation(state: Dict[str, float]) -> np.ndarray:
    """
    Example: Linear state-dependent correlation.

    This function computes correlations that vary linearly with the state variables.

    Args:
        state: Current state variables

    Returns:
        Correlation matrix (3x3)
    """
    r, S, v = state["r"], state["S"], state["v"]

    # Base correlation matrix
    corr = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])

    # Adjust correlations based on state
    corr[0, 1] = corr[1, 0] = np.clip(0.5 + 0.05 * r, -0.8, 0.8)  # rate-equity correlation
    corr[0, 2] = corr[2, 0] = np.clip(0.3 + 0.02 * v, -0.8, 0.8)  # rate-variance correlation
    corr[1, 2] = corr[2, 1] = np.clip(0.2 + 0.07 * S, -0.8, 0.8)  # equity-variance correlation

    return corr


def exponential_state_correlation(state: Dict[str, float]) -> np.ndarray:
    """
    Example: Exponential state-dependent correlation.

    This function computes correlations that vary exponentially with the state variables.

    Args:
        state: Current state variables

    Returns:
        Correlation matrix (3x3)
    """
    r, S, v = state["r"], state["S"], state["v"]

    # Base correlation matrix
    corr = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])

    # Adjust correlations based on state
    corr[0, 1] = corr[1, 0] = np.clip(0.5 * np.exp(0.05 * r), -0.8, 0.8)  # rate-equity correlation
    corr[0, 2] = corr[2, 0] = np.clip(0.3 * np.exp(0.02 * v), -0.8, 0.8)  # rate-variance correlation
    corr[1, 2] = corr[2, 1] = np.clip(0.2 * np.exp(0.07 * S), -0.8, 0.8)  # equity-variance correlation

    return corr
