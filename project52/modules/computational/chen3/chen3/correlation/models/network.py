"""
Network Correlation Implementation

This module implements a network-based correlation structure for the Chen3 model,
where correlations are defined through graph theory and network properties.
This allows for modeling complex dependencies and connectivity patterns between factors.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from ..core.base import BaseCorrelation
from ..utils.exceptions import CorrelationError, CorrelationValidationError
from ..utils.logging_config import logger

class NetworkCorrelation(BaseCorrelation):
    """
    Network-based correlation structure.
    
    This class implements correlations through graph theory,
    where the correlation matrix is constructed from network properties
    such as edge weights, node centrality, and path lengths.
    
    Mathematical Formulation:
    ----------------------
    The correlation matrix is constructed as:
    
    ρ_ij = w_ij * exp(-d_ij/λ)
    
    where:
    - w_ij: Edge weight between nodes i and j
    - d_ij: Shortest path length between nodes i and j
    - λ: Decay parameter
    
    The edge weights can be time-dependent or state-dependent:
    w_ij(t) = f(t, state)
    
    Attributes:
        edge_weights (np.ndarray): Matrix of edge weights
        weight_function (callable): Function for computing edge weights
        decay_param (float): Decay parameter for path length
        n_factors (int): Number of factors
        name (str): Name of the correlation structure
    """
    
    def __init__(
        self,
        edge_weights: Optional[np.ndarray] = None,
        weight_function: Optional[callable] = None,
        decay_param: float = 1.0,
        n_factors: int = 3,
        name: str = "NetworkCorrelation"
    ):
        """
        Initialize network correlation.
        
        Args:
            edge_weights: Matrix of edge weights
            weight_function: Function for computing edge weights
            decay_param: Decay parameter for path length
            n_factors: Number of factors
            name: Name of the correlation structure
            
        Raises:
            CorrelationValidationError: If initialization fails
        """
        super().__init__(n_factors=n_factors, name=name)
        self.decay_param = decay_param
        
        # Initialize edge weights
        if edge_weights is not None:
            self.edge_weights = edge_weights
        else:
            self.edge_weights = np.ones((n_factors, n_factors))
            np.fill_diagonal(self.edge_weights, 0.0)
        
        self.weight_function = weight_function or (lambda t, s: self.edge_weights)
        self._validate_initialization()
        logger.debug(f"Initialized {self.name} with {n_factors} factors")
    
    def _validate_initialization(self):
        """
        Validate initialization parameters.
        
        Raises:
            CorrelationValidationError: If validation fails
        """
        if self.edge_weights.shape != (self.n_factors, self.n_factors):
            raise CorrelationValidationError(
                f"Edge weights must have shape ({self.n_factors}, {self.n_factors})"
            )
        
        if not np.all(self.edge_weights >= 0):
            raise CorrelationValidationError("Edge weights must be non-negative")
        
        if not np.all(np.diag(self.edge_weights) == 0):
            raise CorrelationValidationError("Diagonal edge weights must be zero")
    
    def _compute_weights(
        self,
        t: float,
        state: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Compute edge weights at time t.
        
        Args:
            t: Time point
            state: Current state variables
            
        Returns:
            Matrix of edge weights
            
        Raises:
            CorrelationError: If computation fails
        """
        try:
            weights = self.weight_function(t, state)
            
            # Ensure weights are non-negative
            weights = np.maximum(weights, 0.0)
            
            # Ensure diagonal is zero
            np.fill_diagonal(weights, 0.0)
            
            return weights
        except Exception as e:
            raise CorrelationError(f"Failed to compute edge weights: {str(e)}")
    
    def _compute_shortest_paths(self, weights: np.ndarray) -> np.ndarray:
        """
        Compute shortest path lengths between all nodes.
        
        Args:
            weights: Matrix of edge weights
            
        Returns:
            Matrix of shortest path lengths
        """
        # Initialize distance matrix
        dist = np.full_like(weights, np.inf)
        np.fill_diagonal(dist, 0.0)
        
        # Set direct edge distances
        mask = weights > 0
        dist[mask] = 1.0 / weights[mask]
        
        # Floyd-Warshall algorithm
        for k in range(self.n_factors):
            for i in range(self.n_factors):
                for j in range(self.n_factors):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        
        return dist
    
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
            # Compute edge weights
            weights = self._compute_weights(t, state)
            
            # Compute shortest paths
            paths = self._compute_shortest_paths(weights)
            
            # Construct correlation matrix
            corr = np.zeros((self.n_factors, self.n_factors))
            for i in range(self.n_factors):
                for j in range(self.n_factors):
                    if i == j:
                        corr[i, j] = 1.0
                    else:
                        corr[i, j] = weights[i, j] * np.exp(-paths[i, j] / self.decay_param)
            
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
            f"n_factors={self.n_factors}, "
            f"decay_param={self.decay_param:.2f})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation of the correlation structure."""
        return (
            f"{self.__class__.__name__}("
            f"edge_weights={self.edge_weights}, "
            f"weight_function={self.weight_function}, "
            f"decay_param={self.decay_param}, "
            f"n_factors={self.n_factors}, "
            f"name='{self.name}')"
        )

def create_network_correlation(
    n_factors: int = 3,
    weight_type: str = 'constant',
    decay_rate: float = 0.1
) -> NetworkCorrelation:
    """
    Create a network correlation structure.
    
    This function creates a network correlation structure with specified
    number of factors and weight type.
    
    Args:
        n_factors: Number of factors
        weight_type: Type of weight function ('constant', 'decay', 'oscillate')
        decay_rate: Rate of weight decay
        
    Returns:
        NetworkCorrelation instance
    """
    # Create weight function
    if weight_type == 'constant':
        weight_function = lambda t, s: np.ones((n_factors, n_factors))
    elif weight_type == 'decay':
        weight_function = lambda t, s: np.exp(-decay_rate * t) * np.ones((n_factors, n_factors))
    elif weight_type == 'oscillate':
        weight_function = lambda t, s: (1 + np.cos(decay_rate * t)) * np.ones((n_factors, n_factors))
    else:
        raise ValueError(f"Invalid weight type: {weight_type}")
    
    # Set diagonal to zero
    weights = weight_function(0.0, None)
    np.fill_diagonal(weights, 0.0)
    
    return NetworkCorrelation(
        edge_weights=weights,
        weight_function=weight_function,
        decay_param=1.0,
        n_factors=n_factors
    ) 