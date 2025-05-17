"""
Spectral Correlation Implementation

This module implements a spectral correlation structure for the Chen3 model,
where correlations are defined through spectral decomposition (eigenvalues and eigenvectors).
This allows for modeling correlations based on principal components or eigenmodes.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from ..core.base import BaseCorrelation
from ..utils.exceptions import CorrelationError, CorrelationValidationError
from ..utils.logging_config import logger

class SpectralCorrelation(BaseCorrelation):
    """
    Spectral correlation structure.
    
    This class implements correlations through spectral decomposition,
    where the correlation matrix is constructed from eigenvalues and eigenvectors.
    
    Mathematical Formulation:
    ----------------------
    The correlation matrix is constructed as:
    
    ρ = QΛQ^T
    
    where:
    - Q: Matrix of eigenvectors
    - Λ: Diagonal matrix of eigenvalues
    - Q^T: Transpose of Q
    
    The eigenvalues can be time-dependent or state-dependent:
    λ_i(t) = f_i(t, state)
    
    Attributes:
        eigenvalues (List[float]): List of eigenvalues
        eigenvectors (np.ndarray): Matrix of eigenvectors
        eigenvalue_functions (List[callable]): List of eigenvalue functions
        n_factors (int): Number of factors
        name (str): Name of the correlation structure
    """
    
    def __init__(
        self,
        eigenvalues: Optional[List[float]] = None,
        eigenvectors: Optional[np.ndarray] = None,
        eigenvalue_functions: Optional[List[callable]] = None,
        n_factors: int = 3,
        name: str = "SpectralCorrelation"
    ):
        """
        Initialize spectral correlation.
        
        Args:
            eigenvalues: List of eigenvalues
            eigenvectors: Matrix of eigenvectors
            eigenvalue_functions: List of eigenvalue functions
            n_factors: Number of factors
            name: Name of the correlation structure
            
        Raises:
            CorrelationValidationError: If initialization fails
        """
        super().__init__(n_factors=n_factors, name=name)
        self.eigenvalues = eigenvalues or [1.0] * n_factors
        self.eigenvectors = eigenvectors or np.eye(n_factors)
        self.eigenvalue_functions = eigenvalue_functions or [lambda t, s: 1.0] * n_factors
        self._validate_initialization()
        logger.debug(f"Initialized {self.name} with {n_factors} factors")
    
    def _validate_initialization(self):
        """
        Validate initialization parameters.
        
        Raises:
            CorrelationValidationError: If validation fails
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
        if not np.allclose(self.eigenvectors @ self.eigenvectors.T, np.eye(self.n_factors)):
            raise CorrelationValidationError("Eigenvectors must be orthonormal")
    
    def _compute_eigenvalues(
        self,
        t: float,
        state: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Compute eigenvalues at time t.
        
        Args:
            t: Time point
            state: Current state variables
            
        Returns:
            Array of eigenvalues
            
        Raises:
            CorrelationError: If computation fails
        """
        try:
            eigenvalues = np.array([
                f(t, state) for f in self.eigenvalue_functions
            ])
            
            # Ensure eigenvalues are in [0, 1]
            eigenvalues = np.clip(eigenvalues, 0, 1)
            
            return eigenvalues
        except Exception as e:
            raise CorrelationError(f"Failed to compute eigenvalues: {str(e)}")
    
    def get_correlation_matrix(
        self,
        t: float = 0.0,
        state: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Get the correlation matrix at time t.
        
        Args:
            t: Time point
            state: Current state variables
            
        Returns:
            Correlation matrix (n_factors x n_factors)
            
        Raises:
            CorrelationError: If computation fails
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
        """String representation of the correlation structure."""
        return (
            f"{self.name}("
            f"n_factors={self.n_factors})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation of the correlation structure."""
        return (
            f"{self.__class__.__name__}("
            f"eigenvalues={self.eigenvalues}, "
            f"eigenvectors={self.eigenvectors}, "
            f"eigenvalue_functions={self.eigenvalue_functions}, "
            f"n_factors={self.n_factors}, "
            f"name='{self.name}')"
        )

def create_spectral_correlation(
    n_factors: int = 3,
    eigenvalue_type: str = 'constant',
    decay_rate: float = 0.1
) -> SpectralCorrelation:
    """
    Create a spectral correlation structure.
    
    This function creates a spectral correlation structure with specified
    number of factors and eigenvalue type.
    
    Args:
        n_factors: Number of factors
        eigenvalue_type: Type of eigenvalue function ('constant', 'decay', 'oscillate')
        decay_rate: Rate of eigenvalue decay
        
    Returns:
        SpectralCorrelation instance
    """
    # Create default eigenvalues
    eigenvalues = [1.0] * n_factors
    
    # Create default eigenvectors (identity matrix)
    eigenvectors = np.eye(n_factors)
    
    # Create eigenvalue functions
    if eigenvalue_type == 'constant':
        eigenvalue_functions = [lambda t, s: 1.0] * n_factors
    elif eigenvalue_type == 'decay':
        eigenvalue_functions = [
            lambda t, s, i=i: np.exp(-decay_rate * t * (i + 1))
            for i in range(n_factors)
        ]
    elif eigenvalue_type == 'oscillate':
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
        n_factors=n_factors
    ) 