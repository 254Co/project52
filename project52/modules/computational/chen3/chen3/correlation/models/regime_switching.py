"""
Regime-Switching Correlation Implementation

This module implements regime-switching correlation structures for the Chen3 model,
allowing for different correlation regimes based on market conditions.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import expm

from ..core.base import BaseCorrelation
from ..utils.exceptions import CorrelationError, CorrelationValidationError
from ..utils.logging_config import logger


class RegimeSwitchingCorrelation(BaseCorrelation):
    """
    Regime-switching correlation structure.

    This class implements correlations that switch between different regimes
    based on a continuous-time Markov chain.

    Mathematical Formulation:
    ----------------------
    The correlation matrix at time t is given by:
    ρ(t) = ∑ᵢ pᵢ(t)ρᵢ
    where pᵢ(t) is the probability of being in regime i at time t,
    and ρᵢ is the correlation matrix for regime i.

    The regime probabilities evolve according to:
    P(t) = exp(Qt)P(0)
    where Q is the transition rate matrix and P(0) is the initial regime distribution.

    Attributes:
        correlation_matrices (List[np.ndarray]): List of correlation matrices for each regime
        transition_rates (np.ndarray): Transition rate matrix between regimes
        n_regimes (int): Number of regimes
        n_factors (int): Number of factors
        name (str): Name of the correlation structure
    """

    def __init__(
        self,
        correlation_matrices: List[np.ndarray],
        transition_rates: np.ndarray,
        n_factors: int = 3,
        name: str = "RegimeSwitchingCorrelation",
    ):
        """
        Initialize regime-switching correlation.

        Args:
            correlation_matrices: List of correlation matrices for each regime
            transition_rates: Transition rate matrix between regimes
            n_factors: Number of factors
            name: Name of the correlation structure

        Raises:
            CorrelationValidationError: If initialization fails
        """
        self._correlation_matrices = correlation_matrices
        self.transition_rates = transition_rates
        self.n_regimes = len(correlation_matrices)
        super().__init__(n_factors=n_factors, name=name)
        self._validate_initialization()
        logger.debug(f"Initialized {self.name} with {n_factors} factors")

    @property
    def correlation_matrices(self) -> List[np.ndarray]:
        """Get the correlation matrices."""
        return self._correlation_matrices

    def _validate_initialization(self):
        """
        Validate initialization parameters.

        Raises:
            CorrelationValidationError: If validation fails
        """
        # Validate number of regimes
        if self.n_regimes < 2:
            raise CorrelationValidationError(
                "At least two regimes are required for regime switching"
            )

        # Validate correlation matrices
        for i, corr_matrix in enumerate(self._correlation_matrices):
            self._validate_correlation_matrix(corr_matrix)
            if corr_matrix.shape != (self.n_factors, self.n_factors):
                raise CorrelationValidationError(
                    f"Correlation matrix {i} has incorrect shape"
                )

        # Validate transition rates
        if self.transition_rates.shape != (self.n_regimes, self.n_regimes):
            raise CorrelationValidationError(
                "Transition rate matrix has incorrect shape"
            )
        if not np.allclose(self.transition_rates.sum(axis=1), 0):
            raise CorrelationValidationError(
                "Transition rate matrix rows must sum to zero"
            )

    def get_correlation_matrix(
        self, t: float = 0.0, state: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Get the correlation matrix at time t.

        Args:
            t: Time point
            state: Not used in this implementation

        Returns:
            Correlation matrix (n_factors x n_factors)

        Raises:
            CorrelationError: If computation fails
        """
        try:
            # Compute regime probabilities at time t
            P = expm(self.transition_rates * t)
            p = P[0]  # Initial regime is 0

            # Compute weighted average of correlation matrices
            corr_matrix = np.zeros((self.n_factors, self.n_factors))
            for i, p_i in enumerate(p):
                corr_matrix += p_i * self._correlation_matrices[i]

            return corr_matrix
        except Exception as e:
            raise CorrelationError(f"Failed to compute correlation matrix: {str(e)}")

    def __str__(self) -> str:
        """String representation of the correlation structure."""
        return f"{self.name}(n_regimes={self.n_regimes}, n_factors={self.n_factors})"

    def __repr__(self) -> str:
        """Detailed string representation of the correlation structure."""
        return (
            f"{self.__class__.__name__}("
            f"correlation_matrices={self._correlation_matrices}, "
            f"transition_rates={self.transition_rates}, "
            f"n_factors={self.n_factors}, "
            f"name='{self.name}')"
        )


def create_two_regime_correlation(
    low_corr: float = 0.2,
    high_corr: float = 0.8,
    transition_rate: float = 0.1,
    n_factors: int = 3,
) -> RegimeSwitchingCorrelation:
    """
    Create a two-regime correlation structure.

    This function creates a simple two-regime correlation structure where
    correlations switch between low and high values.

    Args:
        low_corr: Low correlation value
        high_corr: High correlation value
        transition_rate: Rate of regime transitions
        n_factors: Number of factors

    Returns:
        RegimeSwitchingCorrelation instance
    """
    # Define correlation matrices for each regime
    low_corr_matrix = np.ones((n_factors, n_factors)) * low_corr
    np.fill_diagonal(low_corr_matrix, 1.0)

    high_corr_matrix = np.ones((n_factors, n_factors)) * high_corr
    np.fill_diagonal(high_corr_matrix, 1.0)

    # Define transition rate matrix
    transition_rates = np.array(
        [[-transition_rate, transition_rate], [transition_rate, -transition_rate]]
    )

    return RegimeSwitchingCorrelation(
        correlation_matrices=[low_corr_matrix, high_corr_matrix],
        transition_rates=transition_rates,
        n_factors=n_factors,
    )
