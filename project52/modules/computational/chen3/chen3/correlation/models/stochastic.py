"""
Stochastic Correlation Implementation

This module implements stochastic correlation structures for the Chen3 model,
allowing correlations to evolve according to stochastic processes.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from ..core.base import BaseCorrelation
from ..utils.exceptions import CorrelationError, CorrelationValidationError
from ..utils.logging_config import logger


class StochasticCorrelation(BaseCorrelation):
    """
    Stochastic correlation structure.

    This class implements correlations that evolve according to stochastic
    processes, similar to the Heston model for volatility.

    Mathematical Formulation:
    ----------------------
    The correlation matrix elements evolve according to:

    dρ_ij = κ_ij(θ_ij - ρ_ij)dt + σ_ij√(1 - ρ_ij²)dW_ij

    where:
    - ρ_ij: Correlation between factors i and j
    - κ_ij: Mean reversion speed
    - θ_ij: Long-term mean
    - σ_ij: Volatility of correlation
    - W_ij: Brownian motion

    The process ensures that correlations stay in [-1, 1] and exhibit
    mean reversion.

    Attributes:
        mean_reversion (np.ndarray): Mean reversion speeds
        long_term_mean (np.ndarray): Long-term mean correlations
        volatility (np.ndarray): Correlation volatilities
        initial_corr (np.ndarray): Initial correlation matrix
        n_factors (int): Number of factors
        name (str): Name of the correlation structure
    """

    def __init__(
        self,
        mean_reversion: np.ndarray,
        long_term_mean: np.ndarray,
        volatility: np.ndarray,
        initial_corr: np.ndarray,
        n_factors: int = 3,
        name: str = "StochasticCorrelation",
    ):
        """
        Initialize stochastic correlation.

        Args:
            mean_reversion: Mean reversion speeds
            long_term_mean: Long-term mean correlations
            volatility: Correlation volatilities
            initial_corr: Initial correlation matrix
            n_factors: Number of factors
            name: Name of the correlation structure

        Raises:
            CorrelationValidationError: If initialization fails
        """
        super().__init__(n_factors=n_factors, name=name)
        self.mean_reversion = np.asarray(mean_reversion)
        self.long_term_mean = np.asarray(long_term_mean)
        self.volatility = np.asarray(volatility)
        self.initial_corr = np.asarray(initial_corr)
        self._validate_initialization()
        logger.debug(f"Initialized {self.name} with {n_factors} factors")

    def _validate_initialization(self):
        """
        Validate initialization parameters.

        Raises:
            CorrelationValidationError: If validation fails
        """
        # Validate shapes
        if not all(
            x.shape == (self.n_factors, self.n_factors)
            for x in [
                self.mean_reversion,
                self.long_term_mean,
                self.volatility,
                self.initial_corr,
            ]
        ):
            raise CorrelationValidationError(
                f"All parameter matrices must be {self.n_factors}x{self.n_factors}"
            )

        # Validate mean reversion
        if not np.all(self.mean_reversion > 0):
            raise CorrelationValidationError("Mean reversion speeds must be positive")

        # Validate long-term mean
        if not np.all(np.abs(self.long_term_mean) <= 1):
            raise CorrelationValidationError(
                "Long-term mean correlations must be in [-1, 1]"
            )

        # Validate volatility
        if not np.all(self.volatility >= 0):
            raise CorrelationValidationError(
                "Correlation volatilities must be non-negative"
            )

        # Validate initial correlation
        self._validate_correlation_matrix(self.initial_corr)

    def _simulate_correlation(
        self, t: float, dt: float = 0.01, n_steps: int = 100
    ) -> np.ndarray:
        """
        Simulate correlation evolution.

        Args:
            t: Time point
            dt: Time step
            n_steps: Number of time steps

        Returns:
            Correlation matrix at time t

        Raises:
            CorrelationError: If simulation fails
        """
        try:
            # Initialize correlation matrix
            corr = self.initial_corr.copy()

            # Number of steps to simulate
            n = min(int(t / dt), n_steps)
            if n <= 0:
                return corr

            # Simulate evolution
            for _ in range(n):
                # Generate random increments
                dW = np.random.normal(
                    0, np.sqrt(dt), size=(self.n_factors, self.n_factors)
                )
                dW = (dW + dW.T) / 2  # Ensure symmetry

                # Update correlations
                drift = self.mean_reversion * (self.long_term_mean - corr)
                diffusion = self.volatility * np.sqrt(1 - corr**2) * dW
                corr += drift * dt + diffusion

                # Ensure symmetry
                corr = (corr + corr.T) / 2

                # Ensure valid correlation
                corr = np.clip(corr, -1, 1)
                np.fill_diagonal(corr, 1.0)

            return corr
        except Exception as e:
            raise CorrelationError(f"Failed to simulate correlation: {str(e)}")

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
            # Simulate correlation evolution
            corr = self._simulate_correlation(t)

            # Validate the computed matrix
            self._validate_correlation_matrix(corr)

            logger.debug(f"Computed correlation matrix at t={t}")
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
            f"mean_reversion={self.mean_reversion}, "
            f"long_term_mean={self.long_term_mean}, "
            f"volatility={self.volatility}, "
            f"initial_corr={self.initial_corr}, "
            f"n_factors={self.n_factors}, "
            f"name='{self.name}')"
        )


def create_stochastic_correlation(
    mean_reversion: float = 1.0,
    long_term_mean: float = 0.5,
    volatility: float = 0.2,
    initial_corr: float = 0.3,
    n_factors: int = 3,
) -> StochasticCorrelation:
    """
    Create a stochastic correlation structure.

    This function creates a simple stochastic correlation structure where
    all correlations evolve with the same parameters.

    Args:
        mean_reversion: Mean reversion speed
        long_term_mean: Long-term mean correlation
        volatility: Correlation volatility
        initial_corr: Initial correlation value
        n_factors: Number of factors

    Returns:
        StochasticCorrelation instance
    """
    # Create parameter matrices
    mean_reversion_matrix = np.full((n_factors, n_factors), mean_reversion)
    long_term_mean_matrix = np.full((n_factors, n_factors), long_term_mean)
    volatility_matrix = np.full((n_factors, n_factors), volatility)
    initial_corr_matrix = np.full((n_factors, n_factors), initial_corr)

    # Set diagonal elements
    np.fill_diagonal(mean_reversion_matrix, 0)
    np.fill_diagonal(long_term_mean_matrix, 1)
    np.fill_diagonal(volatility_matrix, 0)
    np.fill_diagonal(initial_corr_matrix, 1)

    return StochasticCorrelation(
        mean_reversion=mean_reversion_matrix,
        long_term_mean=long_term_mean_matrix,
        volatility=volatility_matrix,
        initial_corr=initial_corr_matrix,
        n_factors=n_factors,
    )
