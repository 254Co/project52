"""
Tests for correlation structures and their properties.
"""

import numpy as np
import pytest

from chen3.correlation import (
    CopulaCorrelation,
    RegimeSwitchingCorrelation,
    StateDependentCorrelation,
    StochasticCorrelation,
    TimeDependentCorrelation,
)
from chen3.correlation.utils.exceptions import CorrelationValidationError
from chen3.utils.exceptions import ValidationError


def test_time_dependent_correlation():
    """Test time-dependent correlation structure."""
    # Test valid correlation matrix
    valid_matrix = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])

    # Test with valid time points
    time_corr = TimeDependentCorrelation(
        time_points=np.array([0.0, 1.0]),
        correlation_matrices=[valid_matrix, valid_matrix],
    )
    assert time_corr.n_factors == 3
    assert time_corr.name == "TimeDependentCorrelation"

    # Test with non-increasing time points
    with pytest.raises(CorrelationValidationError):
        TimeDependentCorrelation(
            time_points=np.array([1.0, 0.0]),
            correlation_matrices=[valid_matrix, valid_matrix],
        )

    # Test with invalid correlation matrix
    invalid_matrix = np.array([[1.0, 2.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 1.0]])
    with pytest.raises(CorrelationValidationError):
        TimeDependentCorrelation(
            time_points=np.array([0.0, 1.0]),
            correlation_matrices=[valid_matrix, invalid_matrix],
        )


def test_state_dependent_correlation():
    """Test state-dependent correlation structure."""
    # Test valid correlation matrix
    valid_matrix = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])

    # Define correlation function
    def corr_func(state):
        r, S, v = state["r"], state["S"], state["v"]
        return valid_matrix

    # Test with valid state
    state_corr = StateDependentCorrelation(
        correlation_function=corr_func,
        default_state={"r": 0.05, "S": 100.0, "v": 0.2},
    )
    assert state_corr.n_factors == 3
    assert state_corr.name == "StateDependentCorrelation"

    # Test with invalid correlation function
    with pytest.raises(CorrelationValidationError):
        StateDependentCorrelation(
            correlation_function=lambda x: np.array([[1.0, 0.9], [0.9, 1.0]]),
            default_state={"r": 0.05, "S": 100.0, "v": 0.2},
        )


def test_regime_switching_correlation():
    """Test regime-switching correlation structure."""
    # Test valid correlation matrices
    low_corr = np.array([[1.0, 0.2, 0.1], [0.2, 1.0, 0.1], [0.1, 0.1, 1.0]])
    high_corr = np.array([[1.0, 0.8, 0.7], [0.8, 1.0, 0.6], [0.7, 0.6, 1.0]])

    # Test with valid transition rates
    regime_corr = RegimeSwitchingCorrelation(
        correlation_matrices=[low_corr, high_corr],
        transition_rates=np.array([[-0.1, 0.1], [0.2, -0.2]]),
    )
    assert regime_corr.n_factors == 3
    assert regime_corr.name == "RegimeSwitchingCorrelation"

    # Test with invalid transition rates
    with pytest.raises(CorrelationValidationError):
        RegimeSwitchingCorrelation(
            correlation_matrices=[low_corr, high_corr],
            transition_rates=np.array([[-0.1, 0.2], [0.1, -0.2]]),  # Row sums not zero
        )


def test_stochastic_correlation():
    """Test stochastic correlation structure."""
    # Test valid parameter matrices
    mean_reversion = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])
    long_term_mean = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.1], [0.2, 0.1, 1.0]])
    volatility = np.array([[0.1, 0.05, 0.03], [0.05, 0.1, 0.02], [0.03, 0.02, 0.1]])
    initial_corr = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.1], [0.2, 0.1, 1.0]])

    # Test with valid parameters
    stoch_corr = StochasticCorrelation(
        mean_reversion=mean_reversion,
        long_term_mean=long_term_mean,
        volatility=volatility,
        initial_corr=initial_corr,
    )
    assert stoch_corr.n_factors == 3
    assert stoch_corr.name == "StochasticCorrelation"

    # Test with invalid mean reversion
    with pytest.raises(CorrelationValidationError):
        StochasticCorrelation(
            mean_reversion=-mean_reversion,  # Negative mean reversion
            long_term_mean=long_term_mean,
            volatility=volatility,
            initial_corr=initial_corr,
        )


def test_copula_correlation():
    """Test copula-based correlation structure."""
    # Test valid correlation matrix
    base_corr = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])

    # Test Gaussian copula
    gaussian_corr = CopulaCorrelation(
        copula_type="gaussian",
        correlation_matrix=base_corr,
        copula_params={},
    )
    assert gaussian_corr.n_factors == 3
    assert gaussian_corr.name == "CopulaCorrelation"

    # Test Student's t copula
    student_corr = CopulaCorrelation(
        copula_type="student",
        correlation_matrix=base_corr,
        copula_params={"df": 5.0},
    )
    assert student_corr.n_factors == 3

    # Test Clayton copula
    clayton_corr = CopulaCorrelation(
        copula_type="clayton",
        correlation_matrix=base_corr,
        copula_params={"theta": 2.0},
    )
    assert clayton_corr.n_factors == 3

    # Test Gumbel copula
    gumbel_corr = CopulaCorrelation(
        copula_type="gumbel",
        correlation_matrix=base_corr,
        copula_params={"theta": 2.0},
    )
    assert gumbel_corr.n_factors == 3

    # Test invalid copula type
    with pytest.raises(CorrelationValidationError):
        CopulaCorrelation(
            copula_type="invalid",
            correlation_matrix=base_corr,
        )

    # Test missing parameters
    with pytest.raises(CorrelationValidationError):
        CopulaCorrelation(
            copula_type="student",
            correlation_matrix=base_corr,
        )


def test_correlation_validation():
    """Test validation of correlation parameters."""
    # Test valid correlation matrix
    valid_matrix = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])

    time_corr = TimeDependentCorrelation(
        time_points=np.array([0.0, 1.0]),
        correlation_matrices=[valid_matrix, valid_matrix],
    )

    # Test invalid correlation matrix (not positive definite)
    invalid_matrix = np.array([[1.0, 2.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 1.0]])

    with pytest.raises(CorrelationValidationError):
        TimeDependentCorrelation(
            time_points=np.array([0.0, 1.0]),
            correlation_matrices=[valid_matrix, invalid_matrix],
        )
