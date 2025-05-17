"""
Tests for the hierarchical correlation model.
"""

import numpy as np
import pytest

from ..models.hierarchical import (
    HierarchicalCorrelation,
    create_hierarchical_correlation,
)
from ..utils.exceptions import CorrelationError, CorrelationValidationError


def test_initialization():
    """Test initialization of hierarchical correlation."""
    # Test default initialization
    clusters = [[0, 1], [2]]
    corr = HierarchicalCorrelation(clusters)
    assert corr.n_factors == 3
    assert len(corr.clusters) == 2
    assert corr.within_corr == 0.8
    assert corr.between_corr == 0.4
    assert corr.global_corr == 0.2

    # Test custom initialization
    corr = HierarchicalCorrelation(
        clusters=[[0], [1], [2]],
        within_corr=0.9,
        between_corr=0.3,
        global_corr=0.1,
        n_factors=3,
        name="CustomHierarchical",
    )
    assert corr.name == "CustomHierarchical"
    assert corr.within_corr == 0.9
    assert corr.between_corr == 0.3
    assert corr.global_corr == 0.1

    # Test invalid initialization
    with pytest.raises(CorrelationValidationError):
        HierarchicalCorrelation([])  # Empty clusters

    with pytest.raises(CorrelationValidationError):
        HierarchicalCorrelation([[]])  # Empty cluster

    with pytest.raises(CorrelationValidationError):
        HierarchicalCorrelation([[0, 1], [1, 2]])  # Duplicate factors

    with pytest.raises(CorrelationValidationError):
        HierarchicalCorrelation([[0, 1], [2, 3]])  # Invalid factor index

    with pytest.raises(CorrelationValidationError):
        HierarchicalCorrelation([[0, 1], [2]], within_corr=1.5)  # Invalid correlation

    with pytest.raises(CorrelationValidationError):
        HierarchicalCorrelation([[0, 1], [2]], between_corr=-1.5)  # Invalid correlation

    with pytest.raises(CorrelationValidationError):
        HierarchicalCorrelation([[0, 1], [2]], global_corr=2.0)  # Invalid correlation


def test_correlation_matrix():
    """Test correlation matrix computation."""
    # Test two clusters
    clusters = [[0, 1], [2]]
    corr = HierarchicalCorrelation(clusters)
    matrix = corr.get_correlation_matrix()

    # Check shape and properties
    assert matrix.shape == (3, 3)
    assert np.all(np.diag(matrix) == 1.0)
    assert np.all(np.abs(matrix) <= 1.0)

    # Check within-cluster correlation
    assert np.allclose(matrix[0, 1], 0.8)
    assert np.allclose(matrix[1, 0], 0.8)

    # Check between-cluster correlation
    assert np.allclose(matrix[0, 2], 0.4)
    assert np.allclose(matrix[1, 2], 0.4)
    assert np.allclose(matrix[2, 0], 0.4)
    assert np.allclose(matrix[2, 1], 0.4)

    # Test three clusters
    clusters = [[0], [1], [2]]
    corr = HierarchicalCorrelation(
        clusters, within_corr=0.9, between_corr=0.3, global_corr=0.1
    )
    matrix = corr.get_correlation_matrix()

    # Check within-cluster correlation (diagonal)
    assert np.all(np.diag(matrix) == 1.0)

    # Check between-cluster correlation (adjacent)
    assert np.allclose(matrix[0, 1], 0.3)
    assert np.allclose(matrix[1, 2], 0.3)

    # Check global correlation (non-adjacent)
    assert np.allclose(matrix[0, 2], 0.1)


def test_cluster_index():
    """Test cluster index lookup."""
    clusters = [[0, 1], [2]]
    corr = HierarchicalCorrelation(clusters)

    # Test valid factors
    assert corr._get_cluster_index(0) == 0
    assert corr._get_cluster_index(1) == 0
    assert corr._get_cluster_index(2) == 1

    # Test invalid factor
    with pytest.raises(CorrelationError):
        corr._get_cluster_index(3)


def test_create_hierarchical_correlation():
    """Test factory function for creating hierarchical correlations."""
    # Test default creation
    corr = create_hierarchical_correlation()
    assert isinstance(corr, HierarchicalCorrelation)
    assert corr.n_factors == 3
    assert len(corr.clusters) == 2

    # Test custom creation
    corr = create_hierarchical_correlation(
        n_factors=4, n_clusters=2, within_corr=0.9, between_corr=0.3, global_corr=0.1
    )
    assert corr.n_factors == 4
    assert len(corr.clusters) == 2
    assert corr.within_corr == 0.9
    assert corr.between_corr == 0.3
    assert corr.global_corr == 0.1

    # Test invalid creation
    with pytest.raises(ValueError):
        create_hierarchical_correlation(n_factors=3, n_clusters=4)


def test_string_representations():
    """Test string representations of hierarchical correlation."""
    clusters = [[0, 1], [2]]
    corr = HierarchicalCorrelation(clusters)

    # Test string representation
    assert (
        str(corr)
        == "HierarchicalCorrelation(n_clusters=2, within=0.80, between=0.40, global=0.20)"
    )

    # Test detailed representation
    repr_str = repr(corr)
    assert "HierarchicalCorrelation" in repr_str
    assert "clusters=" in repr_str
    assert "within_corr=0.8" in repr_str
    assert "between_corr=0.4" in repr_str
    assert "global_corr=0.2" in repr_str
    assert "n_factors=3" in repr_str
