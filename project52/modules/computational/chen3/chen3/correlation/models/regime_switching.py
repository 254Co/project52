"""
Regime-Switching Correlation Implementation

This module implements regime-switching correlation structures for the Chen3 model,
allowing correlations to switch between different regimes based on a Markov chain.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
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
    The correlation matrix at time t is determined by the current regime:
    
    ρ(t) = ρ_i if regime(t) = i
    
    where:
    - regime(t): Current regime at time t
    - ρ_i: Correlation matrix for regime i
    
    The regime transitions follow a continuous-time Markov chain with
    transition rate matrix Q.
    
    Attributes:
        correlation_matrices (List[np.ndarray]): List of correlation matrices
        transition_rates (np.ndarray): Transition rate matrix
        initial_regime (int): Initial regime
        n_factors (int): Number of factors
        name (str): Name of the correlation structure
    """
    
    def __init__(
        self,
        correlation_matrices: List[np.ndarray],
        transition_rates: np.ndarray,
        initial_regime: int = 0,
        n_factors: int = 3,
        name: str = "RegimeSwitchingCorrelation"
    ):
        """
        Initialize regime-switching correlation.
        
        Args:
            correlation_matrices: List of correlation matrices
            transition_rates: Transition rate matrix
            initial_regime: Initial regime
            n_factors: Number of factors
            name: Name of the correlation structure
            
        Raises:
            CorrelationValidationError: If initialization fails
        """
        super().__init__(n_factors=n_factors, name=name)
        self.correlation_matrices = correlation_matrices
        self.transition_rates = np.asarray(transition_rates)
        self.initial_regime = initial_regime
        self._validate_initialization()
        logger.debug(f"Initialized {self.name} with {len(correlation_matrices)} regimes")
    
    def _validate_initialization(self):
        """
        Validate initialization parameters.
        
        Raises:
            CorrelationValidationError: If validation fails
        """
        n_regimes = len(self.correlation_matrices)
        
        if n_regimes != self.transition_rates.shape[0]:
            raise CorrelationValidationError(
                "Number of regimes must match transition rate matrix size"
            )
        
        if not np.allclose(self.transition_rates.sum(axis=1), 0):
            raise CorrelationValidationError(
                "Transition rate matrix rows must sum to zero"
            )
        
        if not np.all(self.transition_rates.diagonal() <= 0):
            raise CorrelationValidationError(
                "Transition rate matrix diagonal must be non-positive"
            )
        
        if not np.all(self.transition_rates >= 0):
            raise CorrelationValidationError(
                "Transition rate matrix must be non-negative"
            )
        
        if not 0 <= self.initial_regime < n_regimes:
            raise CorrelationValidationError("Invalid initial regime")
        
        for corr in self.correlation_matrices:
            self._validate_correlation_matrix(corr)
    
    def _compute_regime_probabilities(self, t: float) -> np.ndarray:
        """
        Compute regime probabilities at time t.
        
        Args:
            t: Time point
            
        Returns:
            Array of regime probabilities
            
        Raises:
            CorrelationError: If computation fails
        """
        try:
            # Compute transition probability matrix
            P = np.expm(self.transition_rates * t)
            
            # Get probabilities for initial regime
            return P[self.initial_regime]
        except Exception as e:
            raise CorrelationError(f"Failed to compute regime probabilities: {str(e)}")
    
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
            # Compute regime probabilities
            probs = self._compute_regime_probabilities(t)
            
            # Compute weighted average of correlation matrices
            corr = np.zeros((self.n_factors, self.n_factors))
            for i, p in enumerate(probs):
                corr += p * self.correlation_matrices[i]
            
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
            f"n_regimes={len(self.correlation_matrices)}, "
            f"initial_regime={self.initial_regime})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation of the correlation structure."""
        return (
            f"{self.__class__.__name__}("
            f"correlation_matrices={self.correlation_matrices}, "
            f"transition_rates={self.transition_rates}, "
            f"initial_regime={self.initial_regime}, "
            f"n_factors={self.n_factors}, "
            f"name='{self.name}')"
        )

def create_two_regime_correlation(
    low_corr: float = 0.2,
    high_corr: float = 0.8,
    transition_rate: float = 0.1,
    n_factors: int = 3
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
    transition_rates = np.array([
        [-transition_rate, transition_rate],
        [transition_rate, -transition_rate]
    ])
    
    return RegimeSwitchingCorrelation(
        correlation_matrices=[low_corr_matrix, high_corr_matrix],
        transition_rates=transition_rates,
        initial_regime=0,
        n_factors=n_factors
    ) 