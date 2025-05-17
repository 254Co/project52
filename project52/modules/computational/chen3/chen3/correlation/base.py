"""
Base Correlation Classes for the Chen3 Model

This module defines the base classes and interfaces for correlation structures
in the Chen3 model, including the abstract base class and common utilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union
import numpy as np
from ..utils.exceptions import CorrelationError
from ..utils.logging_config import logger

class BaseCorrelation(ABC):
    """
    Abstract base class for correlation structures.
    
    This class defines the interface that all correlation structures must implement.
    It provides common validation and utility methods for correlation matrices.
    """
    
    def __init__(self):
        """Initialize the base correlation structure."""
        self._validate_initialization()
    
    @abstractmethod
    def get_correlation_matrix(
        self,
        t: float = 0.0,
        state: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Get the correlation matrix at time t and state.
        
        Args:
            t: Time point
            state: Current state variables
            
        Returns:
            Correlation matrix (3x3)
            
        Raises:
            CorrelationError: If correlation computation fails
        """
        pass
    
    def _validate_initialization(self):
        """Validate the initialization of the correlation structure."""
        pass
    
    def _validate_correlation_matrix(self, corr: np.ndarray) -> None:
        """
        Validate a correlation matrix.
        
        Args:
            corr: Correlation matrix to validate
            
        Raises:
            CorrelationError: If matrix is invalid
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
        
        Args:
            corr: Correlation matrix
            
        Returns:
            Cholesky factor (lower triangular)
            
        Raises:
            CorrelationError: If decomposition fails
        """
        try:
            return np.linalg.cholesky(corr)
        except np.linalg.LinAlgError as e:
            raise CorrelationError(f"Cholesky decomposition failed: {str(e)}")
    
    def _generate_correlated_random(
        self,
        n_paths: int,
        corr: np.ndarray
    ) -> np.ndarray:
        """
        Generate correlated random variables.
        
        Args:
            n_paths: Number of paths
            corr: Correlation matrix
            
        Returns:
            Correlated random variables
            
        Raises:
            CorrelationError: If generation fails
        """
        try:
            L = self._compute_cholesky(corr)
            Z = np.random.normal(0, 1, size=(n_paths, 3))
            return Z @ L.T
        except Exception as e:
            raise CorrelationError(f"Failed to generate correlated random variables: {str(e)}")
    
    def __str__(self) -> str:
        """String representation of the correlation structure."""
        return f"{self.__class__.__name__}()"
    
    def __repr__(self) -> str:
        """Detailed string representation of the correlation structure."""
        return f"{self.__class__.__name__}()" 