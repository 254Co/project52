"""
Time-Dependent Correlation Implementation

This module implements time-dependent correlation structures for the Chen3 model,
allowing correlations to vary smoothly over time.
"""

from typing import List, Optional, Dict, Tuple
import numpy as np
from scipy.interpolate import interp1d
from ..core.base import BaseCorrelation
from ..utils.exceptions import CorrelationError, CorrelationValidationError
from ..utils.logging_config import logger

class TimeDependentCorrelation(BaseCorrelation):
    """
    Time-dependent correlation structure.
    
    This class implements correlations that vary smoothly over time using
    interpolation between specified time points and correlation matrices.
    
    Mathematical Formulation:
    ----------------------
    The correlation matrix at time t is computed by interpolating between
    the nearest time points:
    
    ρ(t) = w₁ρ(t₁) + w₂ρ(t₂)
    
    where:
    - t₁, t₂: Nearest time points
    - w₁, w₂: Interpolation weights
    - ρ(t): Correlation matrix at time t
    
    Attributes:
        time_points (np.ndarray): Array of time points
        correlation_matrices (List[np.ndarray]): List of correlation matrices
        interpolators (List[Tuple[int, int, interp1d]]): List of interpolators
    """
    
    def __init__(
        self,
        time_points: np.ndarray,
        correlation_matrices: List[np.ndarray],
        n_factors: int = 3,
        name: Optional[str] = None
    ):
        """
        Initialize time-dependent correlation.
        
        Args:
            time_points: Array of time points
            correlation_matrices: List of correlation matrices
            n_factors: Number of factors (default: 3)
            name: Optional name for the correlation structure
            
        Raises:
            CorrelationValidationError: If initialization fails
        """
        self.time_points = np.asarray(time_points)
        self.correlation_matrices = correlation_matrices
        super().__init__(n_factors=n_factors, name=name)
        self._validate_initialization()
        self._setup_interpolation()
    
    def _validate_initialization(self) -> None:
        """
        Validate initialization parameters.
        
        Raises:
            CorrelationValidationError: If validation fails
        """
        if len(self.time_points) != len(self.correlation_matrices):
            raise CorrelationValidationError(
                "Number of time points must match number of correlation matrices"
            )
        
        if not np.all(np.diff(self.time_points) > 0):
            raise CorrelationValidationError("Time points must be strictly increasing")
        
        for i, corr_matrix in enumerate(self.correlation_matrices):
            if not np.allclose(corr_matrix, corr_matrix.T):
                raise CorrelationValidationError(
                    f"Correlation matrix at time {self.time_points[i]} is not symmetric"
                )
            if not np.all(np.diag(corr_matrix) == 1.0):
                raise CorrelationValidationError(
                    f"Correlation matrix at time {self.time_points[i]} must have ones on diagonal"
                )
            try:
                np.linalg.cholesky(corr_matrix)
            except np.linalg.LinAlgError:
                raise CorrelationValidationError(
                    f"Correlation matrix at time {self.time_points[i]} is not positive definite"
                )
    
    def _setup_interpolation(self) -> None:
        """
        Set up interpolation for correlation matrices.
        
        Raises:
            CorrelationError: If interpolation setup fails
        """
        try:
            # Create interpolation for each correlation element
            self.interpolators: List[Tuple[int, int, interp1d]] = []
            for i in range(self.n_factors):
                for j in range(i + 1):
                    values = [corr[i, j] for corr in self.correlation_matrices]
                    interpolator = interp1d(
                        self.time_points,
                        values,
                        kind='linear',
                        bounds_error=False,
                        fill_value=(values[0], values[-1])
                    )
                    self.interpolators.append((i, j, interpolator))
            
            logger.debug(
                f"Set up {len(self.interpolators)} interpolators for "
                f"{self.n_factors}x{self.n_factors} correlation matrix"
            )
        except Exception as e:
            raise CorrelationError(f"Failed to set up interpolation: {str(e)}")
    
    def get_correlation_matrix(
        self,
        t: float = 0.0,
        state: Optional[Dict[str, float]] = None
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
            # Initialize correlation matrix
            corr = np.eye(self.n_factors)
            
            # Fill in interpolated values
            for i, j, interpolator in self.interpolators:
                value = float(interpolator(t))
                corr[i, j] = value
                corr[j, i] = value
            
            # Validate the interpolated matrix
            self._validate_correlation_matrix(corr)
            
            logger.debug(f"Computed correlation matrix at t={t}")
            return corr
        except Exception as e:
            raise CorrelationError(f"Failed to compute correlation matrix: {str(e)}")
    
    def __str__(self) -> str:
        """String representation of the correlation structure."""
        return (
            f"{self.name}("
            f"time_points={self.time_points}, "
            f"n_matrices={len(self.correlation_matrices)})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation of the correlation structure."""
        return (
            f"{self.__class__.__name__}("
            f"time_points={self.time_points}, "
            f"correlation_matrices={self.correlation_matrices}, "
            f"n_factors={self.n_factors}, "
            f"name='{self.name}')"
        ) 