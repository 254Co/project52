"""Global numerical constants & default parameters for the Campbell‑Cochrane habit formation model.

This module defines the core parameters used in the Campbell-Cochrane habit formation model,
which is a prominent asset pricing model that incorporates habit formation in consumption.
The parameters are organized in an immutable dataclass for type safety and ease of use.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ModelParams:
    """Container for Campbell-Cochrane model parameters.
    
    This immutable dataclass holds all the key parameters needed to specify
    the Campbell-Cochrane habit formation model. The parameters are based on
    standard values from the literature and empirical estimates.
    
    Attributes:
        gamma (float): Relative risk-aversion coefficient (CRRA). Higher values indicate
            greater risk aversion. Default: 2.0
        beta (float): Subjective discount factor, representing time preference.
            Must be between 0 and 1. Default: 0.997
        phi (float): Persistence parameter of the surplus-consumption ratio.
            Controls how quickly habits adjust to consumption changes. Default: 0.82
        sigma_c (float): Standard deviation of log consumption growth.
            Measures consumption volatility. Default: 0.01
        g (float): Mean log consumption growth per period.
            Represents long-term consumption growth rate. Default: 0.0013
        s_bar (float): Steady-state surplus ratio.
            Target level of the surplus-consumption ratio. Default: 0.05
        lambda_s (float): Sensitivity of surplus to consumption shocks.
            Controls how consumption shocks affect the surplus ratio. Default: 1.8
    """

    gamma: float = 2.0           # Relative risk‑aversion (CRRA)
    beta: float = 0.997          # Subjective discount factor
    phi: float = 0.82            # Persistence of surplus‑consumption ratio
    sigma_c: float = 0.01        # Std‑dev of log consumption growth
    g: float = 0.0013            # Mean log consumption growth per period
    s_bar: float = 0.05          # Steady‑state surplus ratio
    lambda_s: float = 1.8        # Sensitivity of surplus to consumption shocks

    def as_tuple(self) -> tuple[float, ...]:
        """Convert parameters to a tuple for numerical computations.
        
        Returns:
            tuple[float, ...]: A tuple containing all parameters in the order:
                (gamma, beta, phi, sigma_c, g, s_bar, lambda_s)
        """
        return (
            self.gamma,
            self.beta,
            self.phi,
            self.sigma_c,
            self.g,
            self.s_bar,
            self.lambda_s,
        )