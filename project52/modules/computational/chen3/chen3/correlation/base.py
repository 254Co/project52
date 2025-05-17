"""
Base Correlation Classes for the Chen3 Model

This module defines the base classes and interfaces for correlation structures
in the Chen3 model, including the abstract base class and common utilities.
It provides a framework for implementing different correlation structures
with validation, Cholesky decomposition, and correlated random number generation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

import numpy as np

from ..utils.exceptions import CorrelationError
from ..utils.logging_config import logger


class BaseCorrelation(ABC):
    """
    Abstract base class for correlation structures.

    This class defines the interface that all correlation structures must implement.
    It provides common validation and utility methods for correlation matrices,
    including:
    - Matrix validation (symmetry, positive definiteness, etc.)
    - Cholesky decomposition for correlated random number generation
    - Generation of correlated random variables

    All correlation structures in the Chen3 model must inherit from this class
    and implement the get_correlation_matrix method.
    """

    def __init__(self):
        """
        Initialize the base correlation structure.

        This method calls _validate_initialization to ensure the correlation
        structure is properly initialized. Subclasses can override this method
        to add their own initialization logic.
        """
        self._validate_initialization()

    @abstractmethod
    def get_correlation_matrix(
        self, t: float = 0.0, state: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Get the correlation matrix at time t and state.

        This is an abstract method that must be implemented by all subclasses.
        It should return a valid 3x3 correlation matrix for the given time
        and state variables.

        Args:
            t (float): Time point at which to compute correlations
            state (Optional[Dict[str, float]]): Current state variables
                that may affect correlations

        Returns:
            np.ndarray: A 3x3 correlation matrix

        Raises:
            CorrelationError: If correlation computation fails or
                returns an invalid matrix
        """
        pass

    def _validate_initialization(self) -> None:
        """
        Validate the initialization of the correlation structure.

        This method is called during initialization and can be overridden
        by subclasses to add their own validation logic.

        Raises:
            CorrelationError: If initialization is invalid
        """
        pass

    def _validate_correlation_matrix(self, corr: np.ndarray) -> None:
        """
        Validate a correlation matrix.

        Performs comprehensive validation of a correlation matrix:
        1. Type check (must be numpy array)
        2. Shape check (must be 3x3)
        3. Symmetry check
        4. Diagonal elements check (must be 1.0)
        5. Correlation bounds check (-1 ≤ ρ ≤ 1)
        6. Positive definiteness check using Cholesky decomposition

        Args:
            corr (np.ndarray): Correlation matrix to validate

        Raises:
            CorrelationError: If any validation check fails
        """
        if not isinstance(corr, np.ndarray):
            raise CorrelationError("Correlation must be a numpy array")

        if corr.shape != (3, 3):
            raise CorrelationError("Correlation matrix must be 3x3")

        if not np.allclose(corr, corr.T):
            raise CorrelationError("Correlation matrix must be symmetric")

        if not np.all(np.diag(corr) == 1.0):
            raise CorrelationError("Correlation matrix diagonal must be 1.0")

        if not np.all(np.abs(corr) <= 1.0):
            raise CorrelationError("Correlation values must be in [-1, 1]")

        try:
            np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            raise CorrelationError("Correlation matrix must be positive definite")

    def _compute_cholesky(self, corr: np.ndarray) -> np.ndarray:
        """
        Compute the Cholesky decomposition of a correlation matrix.

        The Cholesky decomposition is used to generate correlated random
        variables. It decomposes the correlation matrix C into L * L^T,
        where L is a lower triangular matrix.

        Args:
            corr (np.ndarray): Correlation matrix to decompose

        Returns:
            np.ndarray: Lower triangular Cholesky factor

        Raises:
            CorrelationError: If the matrix is not positive definite
                or decomposition fails
        """
        try:
            return np.linalg.cholesky(corr)
        except np.linalg.LinAlgError as e:
            raise CorrelationError(f"Cholesky decomposition failed: {str(e)}")

    def _generate_correlated_random(self, n_paths: int, corr: np.ndarray) -> np.ndarray:
        """
        Generate correlated random variables using Cholesky decomposition.

        This method generates n_paths sets of correlated random variables
        using the Cholesky decomposition of the correlation matrix. The
        resulting variables will have the specified correlation structure.

        Args:
            n_paths (int): Number of paths to generate
            corr (np.ndarray): Correlation matrix

        Returns:
            np.ndarray: Array of shape (n_paths, 3) containing correlated
                random variables

        Raises:
            CorrelationError: If generation fails due to invalid
                correlation matrix or other errors
        """
        try:
            L = self._compute_cholesky(corr)
            Z = np.random.normal(0, 1, size=(n_paths, 3))
            return Z @ L.T
        except Exception as e:
            raise CorrelationError(
                f"Failed to generate correlated random variables: {str(e)}"
            )

    def __str__(self) -> str:
        """
        String representation of the correlation structure.

        Returns:
            str: Simple string representation of the class
        """
        return f"{self.__class__.__name__}()"

    def __repr__(self) -> str:
        """
        Detailed string representation of the correlation structure.

        Returns:
            str: Detailed string representation of the class
        """
        return f"{self.__class__.__name__}()"
