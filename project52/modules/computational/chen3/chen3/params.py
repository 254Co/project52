"""
Parameter classes for the Chen3 model.
"""

from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator

from chen3.correlation.models.time_dependent import TimeDependentCorrelation


class RateParams(BaseModel):
    """Parameters for the interest rate process."""

    kappa: float = Field(
        ...,
        gt=0,
        lt=10.0,  # Reasonable upper bound for mean reversion
        description="Mean reversion speed",
    )
    theta: float = Field(
        ...,
        gt=0,
        lt=0.5,  # Reasonable upper bound for long-term rate
        description="Long-term mean level",
    )
    sigma: float = Field(
        ...,
        gt=0,
        lt=1.0,  # Reasonable upper bound for volatility
        description="Volatility",
    )
    r0: float = Field(
        ...,
        gt=0,
        lt=0.5,  # Reasonable upper bound for initial rate
        description="Initial rate",
    )

    @field_validator("sigma")
    def validate_feller_condition(cls, v, values):
        """Validate the Feller condition: 2κθ > σ²."""
        kappa = values.data.get("kappa")
        theta = values.data.get("theta")
        if kappa is not None and theta is not None:
            if 2 * kappa * theta <= v * v:
                raise ValueError(
                    f"Feller condition violated: 2κθ ({2 * kappa * theta:.4f}) "
                    f"must be greater than σ² ({v * v:.4f})"
                )
        return v

    @field_validator("r0")
    def validate_initial_rate(cls, v, values):
        """Validate initial rate is reasonable relative to long-term mean."""
        theta = values.data.get("theta")
        if theta is not None:
            if v > 2 * theta:
                raise ValueError(
                    f"Initial rate ({v:.4f}) should not be more than twice "
                    f"the long-term mean ({theta:.4f})"
                )
        return v


class EquityParams(BaseModel):
    """Parameters for the equity process."""

    mu: float = Field(
        ..., gt=-0.5, lt=0.5, description="Drift rate"  # Reasonable bounds for drift
    )
    q: float = Field(
        ...,
        ge=0,
        lt=0.2,  # Reasonable upper bound for dividend yield
        description="Dividend yield",
    )
    S0: float = Field(
        ...,
        gt=0,
        lt=1e6,  # Reasonable upper bound for stock price
        description="Initial stock price",
    )
    v0: float = Field(
        ...,
        gt=0,
        lt=1.0,  # Reasonable upper bound for initial variance
        description="Initial variance",
    )
    kappa_v: float = Field(
        ...,
        gt=0,
        lt=10.0,  # Reasonable upper bound for mean reversion
        description="Variance mean reversion speed",
    )
    theta_v: float = Field(
        ...,
        gt=0,
        lt=1.0,  # Reasonable upper bound for long-term variance
        description="Variance long-term mean",
    )
    sigma_v: float = Field(
        ...,
        gt=0,
        lt=2.0,  # Reasonable upper bound for vol of vol
        description="Variance volatility",
    )

    @field_validator("sigma_v")
    def validate_feller_condition(cls, v, values):
        """Validate the Feller condition: 2κθ > σ²."""
        kappa_v = values.data.get("kappa_v")
        theta_v = values.data.get("theta_v")
        if kappa_v is not None and theta_v is not None:
            if 2 * kappa_v * theta_v <= v * v:
                raise ValueError(
                    f"Feller condition violated: 2κ_vθ_v ({2 * kappa_v * theta_v:.4f}) "
                    f"must be greater than σ_v² ({v * v:.4f})"
                )
        return v

    @field_validator("v0")
    def validate_initial_variance(cls, v, values):
        """Validate initial variance is reasonable relative to long-term mean."""
        theta_v = values.data.get("theta_v")
        if theta_v is not None:
            if v > 2 * theta_v:
                raise ValueError(
                    f"Initial variance ({v:.4f}) should not be more than twice "
                    f"the long-term mean ({theta_v:.4f})"
                )
        return v

    @field_validator("q")
    def validate_dividend_yield(cls, v, values):
        """Validate dividend yield is reasonable relative to drift."""
        mu = values.data.get("mu")
        if mu is not None:
            if v > abs(mu):
                raise ValueError(
                    f"Dividend yield ({v:.4f}) should not exceed "
                    f"absolute drift rate ({abs(mu):.4f})"
                )
        return v


class ModelParams(BaseModel):
    """Parameters for the Chen3 model."""

    rate: RateParams = Field(..., description="Interest rate parameters")
    equity: EquityParams = Field(..., description="Equity parameters")
    correlation: TimeDependentCorrelation = Field(
        ..., description="Correlation structure"
    )

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
            # Check correlation bounds
            if np.any(np.abs(v) > 1.0):
                raise ValueError("Correlation values must be between -1 and 1")
        return v

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
