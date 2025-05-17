"""
State-Dependent Correlation Implementation

This module implements state-dependent correlation structures for the Chen3 model,
allowing correlations to vary based on the current state of the model.
"""

from typing import Callable, Dict, Optional, Any
import numpy as np
from ..core.base import BaseCorrelation
from ..utils.exceptions import CorrelationError, CorrelationValidationError
from ..utils.logging_config import logger

class StateDependentCorrelation(BaseCorrelation):
    """
    State-dependent correlation structure.
    
    This class implements correlations that vary based on the current state
    of the model, such as interest rates, equity prices, or volatility levels.
    
    Mathematical Formulation:
    ----------------------
    The correlation matrix at state s is computed using a user-defined function:
    
    ρ(s) = f(s)
    
    where:
    - s: Current state variables
    - f: State-dependent correlation function
    - ρ(s): Correlation matrix at state s
    
    The state-dependent function must return a valid correlation matrix
    for any valid state input.
    
    Attributes:
        correlation_function (Callable): Function that computes correlation matrix
                                       based on state variables
        default_state (Dict[str, float]): Default state values
    """
    
    def __init__(
        self,
        correlation_function: Callable[[Dict[str, float]], np.ndarray],
        default_state: Optional[Dict[str, float]] = None,
        n_factors: int = 3,
        name: Optional[str] = None
    ):
        """
        Initialize state-dependent correlation.
        
        Args:
            correlation_function: Function that computes correlation matrix
                                based on state variables
            default_state: Default state values
            n_factors: Number of factors (default: 3)
            name: Optional name for the correlation structure
            
        Raises:
            CorrelationValidationError: If initialization fails
        """
        super().__init__(n_factors=n_factors, name=name)
        self.correlation_function = correlation_function
        self.default_state = default_state or {
            'r': 0.0,  # interest rate
            'S': 0.0,  # equity price
            'v': 0.0   # variance
        }
        self._validate_initialization()
    
    def _validate_initialization(self) -> None:
        """
        Validate initialization parameters.
        
        Raises:
            CorrelationValidationError: If validation fails
        """
        try:
            # Test correlation function with default state
            corr = self.correlation_function(self.default_state)
            self._validate_correlation_matrix(corr)
            
            logger.debug(
                f"Validated state-dependent correlation function with "
                f"default state: {self.default_state}"
            )
        except Exception as e:
            raise CorrelationValidationError(
                f"Invalid correlation function or default state: {str(e)}"
            )
    
    def get_correlation_matrix(
        self,
        t: float = 0.0,
        state: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Get the correlation matrix at the given state.
        
        Args:
            t: Not used in this implementation
            state: Current state variables
            
        Returns:
            Correlation matrix (n_factors x n_factors)
            
        Raises:
            CorrelationError: If computation fails
        """
        try:
            # Use provided state or default
            current_state = state or self.default_state
            
            # Compute correlation matrix
            corr = self.correlation_function(current_state)
            
            # Validate the computed matrix
            self._validate_correlation_matrix(corr)
            
            logger.debug(
                f"Computed correlation matrix for state: {current_state}"
            )
            return corr
        except Exception as e:
            raise CorrelationError(f"Failed to compute correlation matrix: {str(e)}")
    
    def __str__(self) -> str:
        """String representation of the correlation structure."""
        return (
            f"{self.name}("
            f"default_state={self.default_state})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation of the correlation structure."""
        return (
            f"{self.__class__.__name__}("
            f"correlation_function={self.correlation_function.__name__}, "
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
    r, S, v = state['r'], state['S'], state['v']
    
    # Base correlation matrix
    corr = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.2],
        [0.3, 0.2, 1.0]
    ])
    
    # Adjust correlations based on state
    corr[0, 1] = corr[1, 0] = 0.5 + 0.1 * r  # rate-equity correlation
    corr[0, 2] = corr[2, 0] = 0.3 + 0.05 * v  # rate-variance correlation
    corr[1, 2] = corr[2, 1] = 0.2 + 0.15 * S  # equity-variance correlation
    
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
    r, S, v = state['r'], state['S'], state['v']
    
    # Base correlation matrix
    corr = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.2],
        [0.3, 0.2, 1.0]
    ])
    
    # Adjust correlations based on state
    corr[0, 1] = corr[1, 0] = 0.5 * np.exp(0.1 * r)  # rate-equity correlation
    corr[0, 2] = corr[2, 0] = 0.3 * np.exp(0.05 * v)  # rate-variance correlation
    corr[1, 2] = corr[2, 1] = 0.2 * np.exp(0.15 * S)  # equity-variance correlation
    
    return corr 