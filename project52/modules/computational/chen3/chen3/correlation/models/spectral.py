"""
Spectral Correlation Implementation for the Chen3 Model

This module implements a spectral correlation structure for the Chen3 model,
where correlations are defined through spectral decomposition (eigenvalues and eigenvectors).
This allows for modeling correlations based on principal components or eigenmodes.

The implementation uses spectral analysis to model correlations between factors:
- Eigenvalue decomposition captures the principal modes of correlation
- Time-dependent and state-dependent eigenvalues are supported
- Orthonormal eigenvectors ensure valid correlation matrices
- Different eigenvalue patterns (constant, decay, oscillate) can be used

The correlation structure is particularly useful for modeling:
- Principal component-based correlations
- Time-varying correlation patterns
- Market regime changes
- Factor-based correlation structures

Mathematical Formulation:
---------------------
The correlation matrix is constructed through spectral decomposition:

ρ = QΛQ^T

where:
- Q: Matrix of orthonormal eigenvectors
- Λ: Diagonal matrix of eigenvalues
- Q^T: Transpose of Q

The eigenvalues can be time-dependent or state-dependent:
λ_i(t) = f_i(t, state)

Properties:
---------
1. Positive Definiteness: Ensured by non-negative eigenvalues
2. Symmetry: Guaranteed by the spectral decomposition
3. Unit Diagonal: Maintained through normalization
4. Time Evolution: Controlled by eigenvalue functions
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.base import BaseCorrelation
from ..utils.exceptions import CorrelationError, CorrelationValidationError
from ..utils.logging_config import logger


class SpectralCorrelation(BaseCorrelation):
    """
    Spectral-based correlation structure for the Chen3 model.

    This class implements correlations through spectral decomposition,
    where the correlation matrix is constructed from eigenvalues and eigenvectors.

    Mathematical Formulation:
    ----------------------
    The correlation matrix is constructed as:

    ρ = QΛQ^T

    where:
    - Q: Matrix of eigenvectors (orthonormal)
    - Λ: Diagonal matrix of eigenvalues
    - Q^T: Transpose of Q

    The eigenvalues can be time-dependent or state-dependent:
    λ_i(t) = f_i(t, state)

    Properties:
    ---------
    1. Positive Definiteness: Ensured by non-negative eigenvalues
    2. Symmetry: Guaranteed by the spectral decomposition
    3. Unit Diagonal: Maintained through normalization
    4. Time Evolution: Controlled by eigenvalue functions

    Attributes:
        eigenvalues (List[float]): List of base eigenvalues
        eigenvectors (np.ndarray): Matrix of orthonormal eigenvectors
        eigenvalue_functions (List[Callable]): List of functions for computing
            time/state-dependent eigenvalues
        n_factors (int): Number of factors in the correlation structure
        name (str): Name identifier for the correlation structure

    Note:
        The correlation structure ensures positive definiteness through
        the orthonormal eigenvector matrix and non-negative eigenvalues.
    """

    def __init__(
        self,
        eigenvalues: Optional[List[float]] = None,
        eigenvectors: Optional[np.ndarray] = None,
        eigenvalue_functions: Optional[List[Callable]] = None,
        n_factors: int = 3,
        name: str = "SpectralCorrelation",
    ):
        """
        Initialize spectral correlation structure.

        Args:
            eigenvalues (Optional[List[float]]): List of base eigenvalues.
                If None, defaults to ones.
            eigenvectors (Optional[np.ndarray]): Matrix of orthonormal eigenvectors.
                If None, defaults to identity matrix.
            eigenvalue_functions (Optional[List[Callable]]): List of functions
                for computing time/state-dependent eigenvalues. Each function
                should take (t, state) and return an eigenvalue.
            n_factors (int): Number of factors in the correlation structure
            name (str): Name identifier for the correlation structure

        Raises:
            CorrelationValidationError: If initialization parameters are invalid
            ValueError: If eigenvalue functions are invalid

        Example:
            >>> # Create a spectral correlation with decaying eigenvalues
            >>> n_factors = 3
            >>> eigenvalues = [1.0, 0.8, 0.6]
            >>> eigenvectors = np.eye(n_factors)
            >>> corr = SpectralCorrelation(eigenvalues, eigenvectors)
        """
        super().__init__(n_factors=n_factors, name=name)
        self.eigenvalues = eigenvalues or [1.0] * n_factors
        self.eigenvectors = eigenvectors or np.eye(n_factors)
        self.eigenvalue_functions = (
            eigenvalue_functions or [lambda t, s: 1.0] * n_factors
        )
        self._validate_initialization()
        logger.debug(f"Initialized {self.name} with {n_factors} factors")

    def _validate_initialization(self) -> None:
        """
        Validate initialization parameters.

        Performs comprehensive validation of initialization parameters:
        1. Number of eigenvalues matches number of factors
        2. Eigenvector matrix is square and matches number of factors
        3. Number of eigenvalue functions matches number of factors
        4. Eigenvalues are in [0, 1]
        5. Eigenvectors are orthonormal

        Raises:
            CorrelationValidationError: If any validation check fails

        Note:
            The validation ensures that the correlation matrix will be
            positive definite and have unit diagonal elements.
        """
        if len(self.eigenvalues) != self.n_factors:
            raise CorrelationValidationError(
                "Number of eigenvalues must match number of factors"
            )

        if self.eigenvectors.shape != (self.n_factors, self.n_factors):
            raise CorrelationValidationError(
                "Eigenvector matrix must be square and match number of factors"
            )

        if len(self.eigenvalue_functions) != self.n_factors:
            raise CorrelationValidationError(
                "Number of eigenvalue functions must match number of factors"
            )

        # Check eigenvalues
        if not all(0 <= λ <= 1 for λ in self.eigenvalues):
            raise CorrelationValidationError("Eigenvalues must be in [0, 1]")

        # Check eigenvectors
        if not np.allclose(
            self.eigenvectors @ self.eigenvectors.T, np.eye(self.n_factors)
        ):
            raise CorrelationValidationError("Eigenvectors must be orthonormal")

    def _compute_eigenvalues(
        self, t: float, state: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Compute eigenvalues at time t and state.

        This method computes the current eigenvalues using the
        eigenvalue functions, ensuring they remain in the valid range [0, 1].

        Args:
            t (float): Current time point
            state (Optional[Dict[str, float]]): Current state variables
                that may influence the eigenvalues

        Returns:
            np.ndarray: Array of valid eigenvalues in range [0, 1]

        Raises:
            CorrelationError: If eigenvalue computation fails

        Note:
            The eigenvalues are clipped to [0, 1] to ensure positive
            definiteness of the correlation matrix.
        """
        try:
            eigenvalues = np.array([f(t, state) for f in self.eigenvalue_functions])

            # Ensure eigenvalues are in [0, 1]
            eigenvalues = np.clip(eigenvalues, 0, 1)

            return eigenvalues
        except Exception as e:
            raise CorrelationError(f"Failed to compute eigenvalues: {str(e)}")

    def get_correlation_matrix(
        self, t: float = 0.0, state: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Get the correlation matrix at time t and state.

        This method computes the full correlation matrix by:
        1. Computing current eigenvalues
        2. Constructing correlations using the spectral formula
        3. Ensuring valid correlation values
        4. Validating the resulting matrix

        Args:
            t (float): Time point at which to compute correlations
            state (Optional[Dict[str, float]]): Current state variables
                that may influence correlations

        Returns:
            np.ndarray: Valid correlation matrix (n_factors x n_factors)

        Raises:
            CorrelationError: If correlation computation fails

        Example:
            >>> corr = SpectralCorrelation(n_factors=3)
            >>> matrix = corr.get_correlation_matrix(t=1.0)
            >>> print(matrix)
            [[1.0  0.0  0.0]
             [0.0  1.0  0.0]
             [0.0  0.0  1.0]]
        """
        try:
            # Compute eigenvalues
            eigenvalues = self._compute_eigenvalues(t, state)

            # Construct correlation matrix
            Λ = np.diag(eigenvalues)
            corr = self.eigenvectors @ Λ @ self.eigenvectors.T

            # Ensure valid correlation matrix
            corr = np.clip(corr, -1, 1)
            np.fill_diagonal(corr, 1.0)

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
        return f"{self.name}(" f"n_factors={self.n_factors})"

    def __repr__(self) -> str:
        """
        Detailed string representation of the correlation structure.

        Returns:
            str: Detailed string representation showing all parameters
        """
        return (
            f"{self.__class__.__name__}("
            f"eigenvalues={self.eigenvalues}, "
            f"eigenvectors={self.eigenvectors}, "
            f"eigenvalue_functions={self.eigenvalue_functions}, "
            f"n_factors={self.n_factors}, "
            f"name='{self.name}')"
        )


def create_spectral_correlation(
    n_factors: int = 3, eigenvalue_type: str = "constant", decay_rate: float = 0.1
) -> SpectralCorrelation:
    """
    Create a spectral correlation structure with specified parameters.

    This factory function creates a SpectralCorrelation instance with
    predefined eigenvalue functions for common use cases:
    - constant: Time-invariant eigenvalues
    - decay: Exponentially decaying eigenvalues
    - oscillate: Oscillating eigenvalues with cosine function

    Args:
        n_factors (int): Number of factors in the correlation structure
        eigenvalue_type (str): Type of eigenvalue function to use
            ('constant', 'decay', or 'oscillate')
        decay_rate (float): Rate parameter for eigenvalue decay/oscillation

    Returns:
        SpectralCorrelation: Configured spectral correlation instance

    Raises:
        ValueError: If eigenvalue_type is invalid

    Example:
        >>> # Create a spectral correlation with decaying eigenvalues
        >>> corr = create_spectral_correlation(
        ...     n_factors=3,
        ...     eigenvalue_type='decay',
        ...     decay_rate=0.1
        ... )
        >>> matrix = corr.get_correlation_matrix(t=1.0)
    """
    # Create default eigenvalues
    eigenvalues = [1.0] * n_factors

    # Create default eigenvectors (identity matrix)
    eigenvectors = np.eye(n_factors)

    # Create eigenvalue functions based on type
    if eigenvalue_type == "constant":
        eigenvalue_functions = [lambda t, s: 1.0] * n_factors
    elif eigenvalue_type == "decay":
        eigenvalue_functions = [
            lambda t, s, i=i: np.exp(-decay_rate * t * (i + 1))
            for i in range(n_factors)
        ]
    elif eigenvalue_type == "oscillate":
        eigenvalue_functions = [
            lambda t, s, i=i: 0.5 * (1 + np.cos(decay_rate * t * (i + 1)))
            for i in range(n_factors)
        ]
    else:
        raise ValueError(f"Invalid eigenvalue type: {eigenvalue_type}")

    return SpectralCorrelation(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        eigenvalue_functions=eigenvalue_functions,
        n_factors=n_factors,
    )
