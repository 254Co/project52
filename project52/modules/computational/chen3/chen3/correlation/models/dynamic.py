"""
Dynamic Correlation Implementation

This module implements a dynamic correlation structure for the Chen3 model,
combining multiple correlation structures with time-varying weights.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from ..core.base import BaseCorrelation
from ..utils.exceptions import CorrelationError, CorrelationValidationError
from ..utils.logging_config import logger

class DynamicCorrelation(BaseCorrelation):
    """
    Dynamic correlation structure.
    
    This class implements correlations that combine multiple correlation structures
    with time-varying weights, allowing for flexible and adaptive correlation modeling.
    
    Mathematical Formulation:
    ----------------------
    The correlation matrix at time t is computed as a weighted sum:
    
    ρ(t) = Σ w_i(t) * ρ_i(t)
    
    where:
    - w_i(t): Weight of correlation structure i at time t
    - ρ_i(t): Correlation matrix from structure i at time t
    
    The weights can vary over time based on:
    1. Time-dependent functions
    2. State-dependent functions
    3. Regime indicators
    4. Statistical measures
    
    Attributes:
        correlation_structures (List[BaseCorrelation]): List of correlation structures
        weight_functions (List[Callable]): List of weight functions
        n_factors (int): Number of factors
        name (str): Name of the correlation structure
    """
    
    def __init__(
        self,
        correlation_structures: List[BaseCorrelation],
        weight_functions: Optional[List[callable]] = None,
        n_factors: int = 3,
        name: str = "DynamicCorrelation"
    ):
        """
        Initialize dynamic correlation.
        
        Args:
            correlation_structures: List of correlation structures
            weight_functions: List of weight functions
            n_factors: Number of factors
            name: Name of the correlation structure
            
        Raises:
            CorrelationValidationError: If initialization fails
        """
        super().__init__(n_factors=n_factors, name=name)
        self.correlation_structures = correlation_structures
        self.weight_functions = weight_functions or [lambda t, s: 1.0/len(correlation_structures)] * len(correlation_structures)
        self._validate_initialization()
        logger.debug(f"Initialized {self.name} with {len(correlation_structures)} structures")
    
    def _validate_initialization(self):
        """
        Validate initialization parameters.
        
        Raises:
            CorrelationValidationError: If validation fails
        """
        if not self.correlation_structures:
            raise CorrelationValidationError("At least one correlation structure is required")
        
        if len(self.weight_functions) != len(self.correlation_structures):
            raise CorrelationValidationError(
                "Number of weight functions must match number of correlation structures"
            )
        
        for corr in self.correlation_structures:
            if not isinstance(corr, BaseCorrelation):
                raise CorrelationValidationError(
                    "All correlation structures must inherit from BaseCorrelation"
                )
            if corr.n_factors != self.n_factors:
                raise CorrelationValidationError(
                    "All correlation structures must have the same number of factors"
                )
    
    def _compute_weights(
        self,
        t: float,
        state: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Compute weights for each correlation structure.
        
        Args:
            t: Time point
            state: Current state variables
            
        Returns:
            Array of weights
            
        Raises:
            CorrelationError: If computation fails
        """
        try:
            weights = np.array([
                w(t, state) for w in self.weight_functions
            ])
            
            # Normalize weights to sum to 1
            weights = weights / weights.sum()
            
            return weights
        except Exception as e:
            raise CorrelationError(f"Failed to compute weights: {str(e)}")
    
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
            # Compute weights
            weights = self._compute_weights(t, state)
            
            # Get correlation matrices from each structure
            corr_matrices = [
                corr.get_correlation_matrix(t, state)
                for corr in self.correlation_structures
            ]
            
            # Compute weighted average
            corr = np.zeros((self.n_factors, self.n_factors))
            for w, c in zip(weights, corr_matrices):
                corr += w * c
            
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
            f"n_structures={len(self.correlation_structures)})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation of the correlation structure."""
        return (
            f"{self.__class__.__name__}("
            f"correlation_structures={self.correlation_structures}, "
            f"weight_functions={self.weight_functions}, "
            f"n_factors={self.n_factors}, "
            f"name='{self.name}')"
        )

def create_dynamic_correlation(
    correlation_structures: List[BaseCorrelation],
    weight_type: str = 'equal',
    n_factors: int = 3
) -> DynamicCorrelation:
    """
    Create a dynamic correlation structure.
    
    This function creates a dynamic correlation structure with specified
    correlation structures and weight type.
    
    Args:
        correlation_structures: List of correlation structures
        weight_type: Type of weight function ('equal', 'time', 'state')
        n_factors: Number of factors
        
    Returns:
        DynamicCorrelation instance
    """
    if weight_type == 'equal':
        weight_functions = [
            lambda t, s: 1.0/len(correlation_structures)
        ] * len(correlation_structures)
    elif weight_type == 'time':
        weight_functions = [
            lambda t, s: np.exp(-0.1 * t) / len(correlation_structures)
        ] * len(correlation_structures)
    elif weight_type == 'state':
        weight_functions = [
            lambda t, s: 1.0/len(correlation_structures)
            if s is None else
            np.exp(-0.1 * abs(s.get('rate', 0.0))) / len(correlation_structures)
        ] * len(correlation_structures)
    else:
        raise ValueError(f"Invalid weight type: {weight_type}")
    
    return DynamicCorrelation(
        correlation_structures=correlation_structures,
        weight_functions=weight_functions,
        n_factors=n_factors
    ) 