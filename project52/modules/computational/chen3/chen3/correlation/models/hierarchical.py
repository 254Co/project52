"""
Hierarchical Correlation Implementation

This module implements a hierarchical correlation structure for the Chen3 model,
where correlations are organized in a hierarchical manner with different levels
of correlation between factors.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from ..core.base import BaseCorrelation
from ..utils.exceptions import CorrelationError, CorrelationValidationError
from ..utils.logging_config import logger

class HierarchicalCorrelation(BaseCorrelation):
    """
    Hierarchical correlation structure.
    
    This class implements correlations that are organized in a hierarchical manner,
    where factors are grouped into clusters and correlations are defined at different
    levels (within cluster, between clusters, etc.).
    
    Mathematical Formulation:
    ----------------------
    The correlation matrix is constructed as:
    
    ρ_ij = {
        ρ_within    if i,j in same cluster
        ρ_between   if i,j in different clusters
        ρ_global    if i,j in different hierarchies
    }
    
    where:
    - ρ_within: Correlation within a cluster
    - ρ_between: Correlation between clusters
    - ρ_global: Global correlation between hierarchies
    
    Attributes:
        clusters (List[List[int]]): List of factor clusters
        within_corr (float): Correlation within clusters
        between_corr (float): Correlation between clusters
        global_corr (float): Global correlation
        n_factors (int): Number of factors
        name (str): Name of the correlation structure
    """
    
    def __init__(
        self,
        clusters: List[List[int]],
        within_corr: float = 0.8,
        between_corr: float = 0.4,
        global_corr: float = 0.2,
        n_factors: int = 3,
        name: str = "HierarchicalCorrelation"
    ):
        """
        Initialize hierarchical correlation.
        
        Args:
            clusters: List of factor clusters
            within_corr: Correlation within clusters
            between_corr: Correlation between clusters
            global_corr: Global correlation
            n_factors: Number of factors
            name: Name of the correlation structure
            
        Raises:
            CorrelationValidationError: If initialization fails
        """
        super().__init__(n_factors=n_factors, name=name)
        self.clusters = clusters
        self.within_corr = within_corr
        self.between_corr = between_corr
        self.global_corr = global_corr
        self._validate_initialization()
        logger.debug(f"Initialized {self.name} with {len(clusters)} clusters")
    
    def _validate_initialization(self):
        """
        Validate initialization parameters.
        
        Raises:
            CorrelationValidationError: If validation fails
        """
        if not self.clusters:
            raise CorrelationValidationError("At least one cluster is required")
        
        # Check cluster validity
        all_factors = []
        for cluster in self.clusters:
            if not cluster:
                raise CorrelationValidationError("Empty clusters are not allowed")
            if not all(isinstance(f, int) for f in cluster):
                raise CorrelationValidationError("Cluster factors must be integers")
            if not all(0 <= f < self.n_factors for f in cluster):
                raise CorrelationValidationError("Invalid factor indices in clusters")
            all_factors.extend(cluster)
        
        # Check for duplicate factors
        if len(set(all_factors)) != len(all_factors):
            raise CorrelationValidationError("Duplicate factors in clusters")
        
        # Check correlation values
        if not -1 <= self.within_corr <= 1:
            raise CorrelationValidationError("Invalid within-cluster correlation")
        if not -1 <= self.between_corr <= 1:
            raise CorrelationValidationError("Invalid between-cluster correlation")
        if not -1 <= self.global_corr <= 1:
            raise CorrelationValidationError("Invalid global correlation")
    
    def _get_cluster_index(self, factor: int) -> int:
        """
        Get the cluster index for a factor.
        
        Args:
            factor: Factor index
            
        Returns:
            Cluster index
            
        Raises:
            CorrelationError: If factor not found
        """
        for i, cluster in enumerate(self.clusters):
            if factor in cluster:
                return i
        raise CorrelationError(f"Factor {factor} not found in any cluster")
    
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
            corr = np.zeros((self.n_factors, self.n_factors))
            
            # Fill correlation matrix
            for i in range(self.n_factors):
                for j in range(i + 1):
                    if i == j:
                        corr[i, j] = 1.0
                    else:
                        cluster_i = self._get_cluster_index(i)
                        cluster_j = self._get_cluster_index(j)
                        
                        if cluster_i == cluster_j:
                            corr[i, j] = self.within_corr
                        elif abs(cluster_i - cluster_j) == 1:
                            corr[i, j] = self.between_corr
                        else:
                            corr[i, j] = self.global_corr
                        
                        # Ensure symmetry
                        corr[j, i] = corr[i, j]
            
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
            f"n_clusters={len(self.clusters)}, "
            f"within={self.within_corr:.2f}, "
            f"between={self.between_corr:.2f}, "
            f"global={self.global_corr:.2f})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation of the correlation structure."""
        return (
            f"{self.__class__.__name__}("
            f"clusters={self.clusters}, "
            f"within_corr={self.within_corr}, "
            f"between_corr={self.between_corr}, "
            f"global_corr={self.global_corr}, "
            f"n_factors={self.n_factors}, "
            f"name='{self.name}')"
        )

def create_hierarchical_correlation(
    n_factors: int = 3,
    n_clusters: int = 2,
    within_corr: float = 0.8,
    between_corr: float = 0.4,
    global_corr: float = 0.2
) -> HierarchicalCorrelation:
    """
    Create a hierarchical correlation structure.
    
    This function creates a hierarchical correlation structure with specified
    number of factors and clusters.
    
    Args:
        n_factors: Number of factors
        n_clusters: Number of clusters
        within_corr: Correlation within clusters
        between_corr: Correlation between clusters
        global_corr: Global correlation
        
    Returns:
        HierarchicalCorrelation instance
    """
    if n_clusters > n_factors:
        raise ValueError("Number of clusters cannot exceed number of factors")
    
    # Create clusters
    factors_per_cluster = n_factors // n_clusters
    remainder = n_factors % n_clusters
    
    clusters = []
    start_idx = 0
    for i in range(n_clusters):
        cluster_size = factors_per_cluster + (1 if i < remainder else 0)
        clusters.append(list(range(start_idx, start_idx + cluster_size)))
        start_idx += cluster_size
    
    return HierarchicalCorrelation(
        clusters=clusters,
        within_corr=within_corr,
        between_corr=between_corr,
        global_corr=global_corr,
        n_factors=n_factors
    ) 