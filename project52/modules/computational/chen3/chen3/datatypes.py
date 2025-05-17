# File: chen3/datatypes.py
"""
Core Parameter Data Structures for the Chen3 Model

This module defines the fundamental data structures used to parameterize the three-factor
Chen model using Pydantic for robust validation and serialization.

The module provides three main parameter classes:
1. RateParams: Parameters for the interest rate process
2. EquityParams: Parameters for the equity and variance processes
3. ModelParams: Container for all model parameters including correlations

Mathematical Formulation:
------------------------
The three-factor Chen model consists of the following stochastic differential equations:

1. Interest Rate Process (CIR):
   dr_t = κ(θ - r_t)dt + σ√r_t dW^r_t

2. Equity Price Process:
   dS_t = (r_t - q)S_t dt + √v_t S_t dW^S_t

3. Variance Process (Heston):
   dv_t = κ_v(θ_v - v_t)dt + σ_v√v_t dW^v_t

where:
- r_t: Interest rate at time t
- S_t: Equity price at time t
- v_t: Variance at time t
- W^r_t, W^S_t, W^v_t: Correlated Brownian motions
- κ, θ, σ: CIR process parameters
- κ_v, θ_v, σ_v: Heston process parameters
- q: Continuous dividend yield

The correlation structure between the Brownian motions can be:
- Constant (3x3 correlation matrix)
- Time-dependent
- State-dependent
- Regime-switching
- Stochastic
- Copula-based

Example Usage:
-------------
    >>> from chen3 import RateParams, EquityParams, ModelParams
    >>> from chen3.correlation import TimeDependentCorrelation
    >>> import numpy as np
    >>>
    >>> # Define interest rate parameters
    >>> rate_params = RateParams(
    ...     kappa=0.1,    # Mean reversion speed
    ...     theta=0.05,   # Long-term mean
    ...     sigma=0.1,    # Volatility
    ...     r0=0.03       # Initial rate
    ... )
    >>>
    >>> # Define equity parameters
    >>> equity_params = EquityParams(
    ...     mu=0.05,      # Drift
    ...     q=0.02,       # Dividend yield
    ...     S0=100.0,     # Initial stock price
    ...     v0=0.04,      # Initial variance
    ...     kappa_v=2.0,  # Variance mean reversion
    ...     theta_v=0.04, # Long-term variance
    ...     sigma_v=0.3   # Volatility of variance
    ... )
    >>>
    >>> # Define correlation structure
    >>> corr_matrix = np.array([
    ...     [1.0, 0.5, 0.3],
    ...     [0.5, 1.0, 0.2],
    ...     [0.3, 0.2, 1.0]
    ... ])
    >>>
    >>> # Create model parameters
    >>> model_params = ModelParams(
    ...     rate=rate_params,
    ...     equity=equity_params,
    ...     correlation=corr_matrix
    ... )
"""

from typing import Optional, Union, Dict, Any

import numpy as np
from pydantic import BaseModel, Field, field_validator, ValidationError

from chen3.correlation import (
    CopulaCorrelation,
    RegimeSwitchingCorrelation,
    StateDependentCorrelation,
    StochasticCorrelation,
    TimeDependentCorrelation,
)

from .utils.logging import logger


class RateParams(BaseModel):
    """
    Parameters for the Cox-Ingersoll-Ross (CIR) interest rate process.

    The CIR process follows the SDE:
        dr_t = κ(θ - r_t)dt + σ√r_t dW_t

    This process ensures that interest rates remain positive and exhibit
    mean reversion, making it suitable for modeling short-term interest rates.

    Mathematical Properties:
    ----------------------
    1. Mean Reversion:
       - The process reverts to θ at speed κ
       - Higher κ means faster mean reversion

    2. Volatility:
       - The volatility term σ√r_t ensures that:
         * Volatility increases with the level of rates
         * Rates cannot become negative

    3. Stationary Distribution:
       - The process has a gamma stationary distribution
       - Mean: θ
       - Variance: θσ²/(2κ)

    Attributes:
        kappa (float): Mean reversion speed of the interest rate
                      Higher values indicate faster mean reversion
                      Typical range: 0.05 to 0.5
        theta (float): Long-term mean level of the interest rate
                      The rate reverts to this level in the long run
                      Typical range: 0.02 to 0.08
        sigma (float): Volatility of the interest rate process
                      Controls the magnitude of random fluctuations
                      Typical range: 0.05 to 0.2
        r0 (float): Initial interest rate level
                   Starting point for the simulation
                   Should be positive

    Example:
        >>> rate_params = RateParams(
        ...     kappa=0.1,    # Moderate mean reversion
        ...     theta=0.05,   # 5% long-term rate
        ...     sigma=0.1,    # 10% volatility
        ...     r0=0.03       # Start at 3%
        ... )
    """

    kappa: float = Field(gt=0, description="Mean reversion speed of the interest rate")
    theta: float = Field(gt=0, description="Long-term mean level of the interest rate")
    sigma: float = Field(gt=0, description="Volatility of the interest rate process")
    r0: float = Field(gt=0, description="Initial interest rate level")

    @property
    def feller_condition_satisfied(self) -> bool:
        """Check if the Feller condition is satisfied: 2κθ > σ²."""
        return 2 * self.kappa * self.theta > self.sigma**2

    @property
    def mean_reversion_time(self) -> float:
        """Time scale for mean reversion: 1/κ."""
        return 1.0 / self.kappa

    @property
    def volatility_scale(self) -> float:
        """Scale of volatility in the stationary distribution: σ/√(2κ)."""
        return self.sigma / np.sqrt(2 * self.kappa)

    @field_validator("kappa")
    def validate_kappa(cls, v):
        if v <= 0:
            raise ValueError("Mean reversion speed must be positive")
        if v > 10.0:  # Reasonable upper bound
            raise ValueError("Mean reversion speed seems too high")
        return v

    @field_validator("theta")
    def validate_theta(cls, v):
        if v <= 0:
            raise ValueError("Long-term mean must be positive")
        if v > 0.5:  # Reasonable upper bound for interest rates
            raise ValueError("Long-term mean seems too high")
        return v

    @field_validator("sigma")
    def validate_sigma(cls, v, values):
        if v <= 0:
            raise ValueError("Volatility must be positive")
        if v > 1.0:  # Reasonable upper bound for volatility
            raise ValueError("Volatility seems too high")
        
        kappa = values.data.get("kappa")
        theta = values.data.get("theta")
        if kappa is not None and theta is not None:
            if 2 * kappa * theta <= v**2:
                raise ValueError("Feller condition violated: 2κθ ≤ σ²")
        return v

    @field_validator("r0")
    def validate_r0(cls, v):
        if v < 0:
            raise ValueError("Initial rate must be non-negative")
        if v > 0.5:  # Reasonable upper bound for initial rate
            raise ValueError("Initial rate seems too high")
        return v

    def validate_parameters(self) -> None:
        """Validate all parameters together."""
        if not self.feller_condition_satisfied:
            raise ValueError(
                "Feller condition violated: 2κθ ≤ σ². "
                "This may lead to numerical instability."
            )
        
        # Check if mean reversion time is reasonable
        if self.mean_reversion_time < 0.1:  # Less than 1.2 months
            raise ValueError(
                "Mean reversion time too short. "
                "This may lead to numerical instability."
            )
        
        # Check if volatility scale is reasonable
        if self.volatility_scale > 0.5:  # More than 50% annualized
            raise ValueError(
                "Volatility scale too high. "
                "This may lead to numerical instability."
            )


class EquityParams(BaseModel):
    """
    Parameters for the equity and variance processes.

    The equity process follows:
        dS_t = (r_t - q)S_t dt + √v_t S_t dW^S_t
    The variance process follows:
        dv_t = κ_v(θ_v - v_t)dt + σ_v√v_t dW^v_t

    This specification combines:
    1. A geometric Brownian motion for the equity price
    2. A Heston-type stochastic volatility process
    3. Correlation with the interest rate process

    Mathematical Properties:
    ----------------------
    1. Equity Process:
       - Log-normal distribution in the short term
       - Drift adjusted for risk-free rate and dividends
       - Volatility driven by the variance process

    2. Variance Process:
       - Mean-reverting process
       - Ensures positive variance
       - Allows for volatility clustering

    Attributes:
        mu (float): Drift rate of the equity process
                   Typically set to risk-free rate in risk-neutral measure
                   Typical range: 0.02 to 0.08
        q (float): Continuous dividend yield
                  Reduces the growth rate of the stock price
                  Typical range: 0.0 to 0.05
        S0 (float): Initial stock price
                   Starting point for the simulation
                   Must be positive
        v0 (float): Initial variance level
                   Starting point for the volatility process
                   Must be positive
        kappa_v (float): Mean reversion speed of the variance process
                        Higher values indicate faster mean reversion
                        Typical range: 1.0 to 5.0
        theta_v (float): Long-term mean level of the variance
                        The variance reverts to this level in the long run
                        Typical range: 0.01 to 0.09
        sigma_v (float): Volatility of the variance process
                        Controls the magnitude of volatility fluctuations
                        Typical range: 0.1 to 0.5

    Example:
        >>> equity_params = EquityParams(
        ...     mu=0.05,      # 5% drift
        ...     q=0.02,       # 2% dividend yield
        ...     S0=100.0,     # Start at $100
        ...     v0=0.04,      # 20% initial volatility
        ...     kappa_v=2.0,  # Moderate mean reversion
        ...     theta_v=0.04, # 20% long-term volatility
        ...     sigma_v=0.3   # 30% vol of vol
        ... )
    """

    mu: float = Field(description="Drift rate of the equity process")
    q: float = Field(ge=0, description="Continuous dividend yield")
    S0: float = Field(gt=0, description="Initial stock price")
    v0: float = Field(gt=0, description="Initial variance level")
    kappa_v: float = Field(
        gt=0, description="Mean reversion speed of the variance process"
    )
    theta_v: float = Field(gt=0, description="Long-term mean level of the variance")
    sigma_v: float = Field(gt=0, description="Volatility of the variance process")

    @property
    def feller_condition_satisfied(self) -> bool:
        """Check if the Feller condition is satisfied: 2κ_vθ_v > σ_v²."""
        return 2 * self.kappa_v * self.theta_v > self.sigma_v**2

    @property
    def mean_reversion_time(self) -> float:
        """Time scale for variance mean reversion: 1/κ_v."""
        return 1.0 / self.kappa_v

    @property
    def volatility_scale(self) -> float:
        """Scale of variance volatility in the stationary distribution: σ_v/√(2κ_v)."""
        return self.sigma_v / np.sqrt(2 * self.kappa_v)

    @property
    def total_return(self) -> float:
        """Total return rate: μ - q."""
        return self.mu - self.q

    @field_validator("mu")
    def validate_mu(cls, v):
        if abs(v) > 1.0:  # Reasonable bound for drift
            raise ValueError("Drift rate seems too high")
        return v

    @field_validator("q")
    def validate_q(cls, v):
        if v < 0:
            raise ValueError("Dividend yield must be non-negative")
        if v > 0.2:  # Reasonable upper bound for dividend yield
            raise ValueError("Dividend yield seems too high")
        return v

    @field_validator("S0")
    def validate_S0(cls, v):
        if v <= 0:
            raise ValueError("Initial stock price must be positive")
        if v > 10000:  # Reasonable upper bound for stock price
            raise ValueError("Initial stock price seems too high")
        return v

    @field_validator("v0")
    def validate_v0(cls, v):
        if v <= 0:
            raise ValueError("Initial variance must be positive")
        if v > 1.0:  # Reasonable upper bound for variance
            raise ValueError("Initial variance seems too high")
        return v

    @field_validator("kappa_v")
    def validate_kappa_v(cls, v):
        if v <= 0:
            raise ValueError("Variance mean reversion must be positive")
        if v > 10.0:  # Reasonable upper bound
            raise ValueError("Variance mean reversion seems too high")
        return v

    @field_validator("theta_v")
    def validate_theta_v(cls, v):
        if v <= 0:
            raise ValueError("Long-term variance must be positive")
        if v > 1.0:  # Reasonable upper bound for variance
            raise ValueError("Long-term variance seems too high")
        return v

    @field_validator("sigma_v")
    def validate_sigma_v(cls, v, values):
        if v <= 0:
            raise ValueError("Volatility of variance must be positive")
        if v > 2.0:  # Reasonable upper bound for vol of vol
            raise ValueError("Volatility of variance seems too high")
        
        kappa_v = values.data.get("kappa_v")
        theta_v = values.data.get("theta_v")
        if kappa_v is not None and theta_v is not None:
            if 2 * kappa_v * theta_v <= v**2:
                raise ValueError("Feller condition violated for variance process")
        return v

    def validate_parameters(self) -> None:
        """Validate all parameters together."""
        # Check Feller condition for variance process
        if 2 * self.kappa_v * self.theta_v <= self.sigma_v**2:
            raise ValueError(
                "Feller condition violated for variance process: 2κ_vθ_v ≤ σ_v². "
                "This may lead to numerical instability."
            )
        
        # Check if mean reversion time is reasonable
        mean_rev_time = 1.0 / self.kappa_v
        if mean_rev_time < 0.1:  # Less than 1.2 months
            raise ValueError(
                "Variance mean reversion time too short. "
                "This may lead to numerical instability."
            )
        
        # Check if volatility scale is reasonable
        vol_scale = self.sigma_v / np.sqrt(2 * self.kappa_v)
        if vol_scale > 0.5:  # More than 50% annualized
            raise ValueError(
                "Variance volatility scale too high. "
                "This may lead to numerical instability."
            )
        
        # Check if initial variance is reasonable compared to long-term variance
        if self.v0 > 2 * self.theta_v:
            raise ValueError(
                "Initial variance too high compared to long-term variance. "
                "This may lead to numerical instability."
            )


class ModelParams(BaseModel):
    """
    Container for all model parameters including correlations.

    This class combines the interest rate and equity parameters with their
    correlation structure to form a complete model specification.

    The correlation structure can be one of several types:
    1. Constant Correlation:
       - Simple 3x3 correlation matrix
       - Fixed throughout the simulation
       - Must be positive definite

    2. Time-Dependent Correlation:
       - Correlations vary with time
       - Interpolated between specified time points
       - Useful for term structure effects

    3. State-Dependent Correlation:
       - Correlations depend on current state
       - Can capture regime effects
       - Flexible functional form

    4. Regime-Switching Correlation:
       - Discrete correlation regimes
       - Markov chain transitions
       - Captures market regime changes

    5. Stochastic Correlation:
       - Continuous correlation process
       - Mean-reverting dynamics
       - Similar to Heston model

    6. Copula-Based Correlation:
       - Flexible dependency structure
       - Can capture tail dependencies
       - Various copula families available

    Attributes:
        rate (RateParams): Parameters for the interest rate process
        equity (EquityParams): Parameters for the equity and variance processes
        correlation (Union[np.ndarray, TimeDependentCorrelation, StateDependentCorrelation,
                          RegimeSwitchingCorrelation, StochasticCorrelation, CopulaCorrelation]):
            Correlation structure between the three factors. Can be one of:
            - np.ndarray: Constant correlation matrix (shape 3x3)
            - TimeDependentCorrelation: Time-varying correlations
            - StateDependentCorrelation: State-dependent correlations
            - RegimeSwitchingCorrelation: Regime-switching correlations
            - StochasticCorrelation: Stochastic correlation process
            - CopulaCorrelation: Copula-based correlation structure

    Example:
        >>> from chen3.correlation import TimeDependentCorrelation
        >>> import numpy as np
        >>>
        >>> # Create time-dependent correlation
        >>> time_points = np.array([0.0, 1.0, 2.0])
        >>> corr_matrices = [
        ...     np.array([[1.0, 0.5, 0.3],
        ...              [0.5, 1.0, 0.2],
        ...              [0.3, 0.2, 1.0]]),
        ...     np.array([[1.0, 0.6, 0.4],
        ...              [0.6, 1.0, 0.3],
        ...              [0.4, 0.3, 1.0]]),
        ...     np.array([[1.0, 0.7, 0.5],
        ...              [0.7, 1.0, 0.4],
        ...              [0.5, 0.4, 1.0]])
        ... ]
        >>> time_corr = TimeDependentCorrelation(
        ...     time_points=time_points,
        ...     correlation_matrices=corr_matrices
        ... )
        >>>
        >>> # Create model parameters
        >>> model_params = ModelParams(
        ...     rate=rate_params,
        ...     equity=equity_params,
        ...     correlation=time_corr
        ... )
    """

    rate: RateParams
    equity: EquityParams
    correlation: Union[
        np.ndarray,
        TimeDependentCorrelation,
        StateDependentCorrelation,
        RegimeSwitchingCorrelation,
        StochasticCorrelation,
        CopulaCorrelation,
    ]

    @field_validator("correlation")
    def validate_correlation(cls, v):
        """Validate the correlation structure."""
        if isinstance(v, np.ndarray):
            if v.shape != (3, 3):
                raise ValueError("Correlation matrix must be 3x3")
            if not np.allclose(v, v.T):
                raise ValueError("Correlation matrix must be symmetric")
            if not np.all(np.linalg.eigvals(v) > 0):
                raise ValueError("Correlation matrix must be positive definite")
        return v

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True
        validate_assignment = True


class CorrelationParams(BaseModel):
    """Parameters for correlations between rate, stock, and variance processes."""
    rho_rs: float = Field(default=0.1, description="Rate-stock correlation")
    rho_rv: float = Field(default=-0.1, description="Rate-variance correlation")
    rho_sv: float = Field(default=-0.5, description="Stock-variance correlation")

    @field_validator('rho_rs', 'rho_rv', 'rho_sv')
    def validate_correlation(cls, v: float) -> float:
        """Validate that correlation is between -1 and 1."""
        if not -1 <= v <= 1:
            raise ValueError(f"Correlation must be between -1 and 1, got {v}")
        return v

    def to_matrix(self) -> np.ndarray:
        """Convert correlation parameters to a correlation matrix."""
        return np.array([
            [1.0, self.rho_rs, self.rho_rv],
            [self.rho_rs, 1.0, self.rho_sv],
            [self.rho_rv, self.rho_sv, 1.0]
        ])
