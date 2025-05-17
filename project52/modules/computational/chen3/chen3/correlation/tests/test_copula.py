"""
Unit tests for the copula-based correlation module.
"""

import numpy as np
import pytest

from ..models.copula import (
    CopulaCorrelation,
    create_clayton_copula_correlation,
    create_gaussian_copula_correlation,
    create_gumbel_copula_correlation,
    create_student_copula_correlation,
)
from ..utils.exceptions import CorrelationError, CorrelationValidationError


def test_initialization():
    """Test copula correlation initialization."""
    # Test default initialization
    corr = create_gaussian_copula_correlation()
    assert corr.n_factors == 3
    assert corr.name == "CopulaCorrelation"
    assert corr.copula_type == "gaussian"
    assert corr.correlation_matrix.shape == (3, 3)

    # Test custom initialization
    n_factors = 4
    corr_matrix = np.ones((n_factors, n_factors)) * 0.5
    np.fill_diagonal(corr_matrix, 1.0)

    corr = CopulaCorrelation(
        copula_type="student",
        correlation_matrix=corr_matrix,
        copula_params={"df": 5.0},
        n_factors=n_factors,
        name="CustomCorr",
    )
    assert corr.n_factors == n_factors
    assert corr.name == "CustomCorr"
    assert corr.copula_type == "student"
    assert corr.correlation_matrix.shape == (n_factors, n_factors)

    # Test invalid initialization
    with pytest.raises(CorrelationValidationError):
        CopulaCorrelation(copula_type="invalid", correlation_matrix=corr_matrix)

    with pytest.raises(CorrelationValidationError):
        CopulaCorrelation(
            copula_type="student",
            correlation_matrix=corr_matrix,
            copula_params={},  # Missing df parameter
        )

    with pytest.raises(CorrelationValidationError):
        CopulaCorrelation(
            copula_type="clayton",
            correlation_matrix=corr_matrix,
            copula_params={},  # Missing theta parameter
        )


def test_correlation_matrix():
    """Test correlation matrix computation."""
    corr = create_gaussian_copula_correlation()

    # Test basic properties
    result = corr.get_correlation_matrix()
    assert result.shape == (3, 3)
    assert np.allclose(np.diag(result), 1.0)
    assert np.allclose(result, result.T)
    assert np.all(np.abs(result) <= 1)

    # Test different copula types
    corr = create_student_copula_correlation(df=5.0)
    result = corr.get_correlation_matrix()
    assert result.shape == (3, 3)
    assert np.allclose(np.diag(result), 1.0)
    assert np.allclose(result, result.T)
    assert np.all(np.abs(result) <= 1)

    corr = create_clayton_copula_correlation(theta=2.0)
    result = corr.get_correlation_matrix()
    assert result.shape == (3, 3)
    assert np.allclose(np.diag(result), 1.0)
    assert np.allclose(result, result.T)
    assert np.all(np.abs(result) <= 1)

    corr = create_gumbel_copula_correlation(theta=2.0)
    result = corr.get_correlation_matrix()
    assert result.shape == (3, 3)
    assert np.allclose(np.diag(result), 1.0)
    assert np.allclose(result, result.T)
    assert np.all(np.abs(result) <= 1)


def test_copula_functions():
    """Test individual copula functions."""
    corr = create_gaussian_copula_correlation()

    # Test Gaussian copula
    u = np.array([0.5, 0.5, 0.5])
    c = corr._gaussian_copula(u)
    assert 0 <= c <= 1

    # Test Student's t copula
    corr = create_student_copula_correlation(df=5.0)
    c = corr._student_copula(u)
    assert 0 <= c <= 1

    # Test Clayton copula
    corr = create_clayton_copula_correlation(theta=2.0)
    c = corr._clayton_copula(u)
    assert 0 <= c <= 1

    # Test Gumbel copula
    corr = create_gumbel_copula_correlation(theta=2.0)
    c = corr._gumbel_copula(u)
    assert 0 <= c <= 1


def test_string_representations():
    """Test string representations."""
    corr = create_gaussian_copula_correlation()

    # Test string representation
    expected_str = "CopulaCorrelation(type=gaussian)"
    assert str(corr) == expected_str

    # Test detailed representation
    expected_repr = (
        f"CopulaCorrelation("
        f"copula_type=gaussian, "
        f"correlation_matrix={corr.correlation_matrix}, "
        f"copula_params={{}}, "
        f"n_factors=3, "
        f"name='CopulaCorrelation')"
    )
    assert repr(corr) == expected_repr


def test_factory_functions():
    """Test the copula correlation factory functions."""
    # Test Gaussian copula
    corr = create_gaussian_copula_correlation(base_corr=0.6, n_factors=4)
    assert corr.n_factors == 4
    assert corr.copula_type == "gaussian"
    assert np.allclose(
        corr.correlation_matrix,
        np.array(
            [
                [1.0, 0.6, 0.6, 0.6],
                [0.6, 1.0, 0.6, 0.6],
                [0.6, 0.6, 1.0, 0.6],
                [0.6, 0.6, 0.6, 1.0],
            ]
        ),
    )

    # Test Student's t copula
    corr = create_student_copula_correlation(base_corr=0.6, df=5.0, n_factors=4)
    assert corr.n_factors == 4
    assert corr.copula_type == "student"
    assert corr.copula_params["df"] == 5.0
    assert np.allclose(
        corr.correlation_matrix,
        np.array(
            [
                [1.0, 0.6, 0.6, 0.6],
                [0.6, 1.0, 0.6, 0.6],
                [0.6, 0.6, 1.0, 0.6],
                [0.6, 0.6, 0.6, 1.0],
            ]
        ),
    )

    # Test Clayton copula
    corr = create_clayton_copula_correlation(base_corr=0.6, theta=2.0, n_factors=4)
    assert corr.n_factors == 4
    assert corr.copula_type == "clayton"
    assert corr.copula_params["theta"] == 2.0
    assert np.allclose(
        corr.correlation_matrix,
        np.array(
            [
                [1.0, 0.6, 0.6, 0.6],
                [0.6, 1.0, 0.6, 0.6],
                [0.6, 0.6, 1.0, 0.6],
                [0.6, 0.6, 0.6, 1.0],
            ]
        ),
    )

    # Test Gumbel copula
    corr = create_gumbel_copula_correlation(base_corr=0.6, theta=2.0, n_factors=4)
    assert corr.n_factors == 4
    assert corr.copula_type == "gumbel"
    assert corr.copula_params["theta"] == 2.0
    assert np.allclose(
        corr.correlation_matrix,
        np.array(
            [
                [1.0, 0.6, 0.6, 0.6],
                [0.6, 1.0, 0.6, 0.6],
                [0.6, 0.6, 1.0, 0.6],
                [0.6, 0.6, 0.6, 1.0],
            ]
        ),
    )
