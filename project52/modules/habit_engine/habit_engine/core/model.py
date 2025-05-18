"""Campbell‑Cochrane habit‑formation utility, marginal utility, & SDF implementation.

This module implements the core mathematical functions of the Campbell-Cochrane habit
formation model, including utility functions, marginal utilities, and the stochastic
discount factor (SDF). The implementation is purely functional and deterministic,
with no random number generation or I/O operations.
"""
from __future__ import annotations

import numpy as np

from .constants import ModelParams
from ..utils.validation import assert_finite


class HabitModel:
    """Core implementation of the Campbell-Cochrane habit formation model.
    
    This class provides the fundamental mathematical functions for the Campbell-Cochrane
    habit formation model, including utility calculations, marginal utilities, and
    stochastic discount factor computations. The implementation is immutable and
    purely functional, ensuring thread safety and reproducibility.
    
    Attributes:
        p (ModelParams): The model parameters container.
    """

    def __init__(self, params: ModelParams):
        """Initialize the habit model with parameters.
        
        Args:
            params (ModelParams): The model parameters container.
        """
        self.p = params

    # ------------------------------------------------------------------
    # Utility & marginal utility
    # ------------------------------------------------------------------
    def utility(self, c: np.ndarray, s: np.ndarray) -> np.ndarray:  # noqa: N802
        """Calculate the instantaneous utility U(C_t,S_t).
        
        The utility function is given by:
            U(C_t,S_t) = (C_t * S_t)^(1-γ)/(1-γ)
        
        where C_t is consumption and S_t is the surplus-consumption ratio.
        In the Campbell-Cochrane model, consumption relative to habit is proxied
        by the surplus-consumption ratio S_t.
        
        Args:
            c (np.ndarray): Array of consumption levels.
            s (np.ndarray): Array of surplus-consumption ratios.
            
        Returns:
            np.ndarray: Array of utility values, same shape as inputs.
            
        Raises:
            ValueError: If any utility values are non-finite.
        """
        γ = self.p.gamma
        util = np.power(c * s, 1.0 - γ) / (1.0 - γ)
        assert_finite(util, "utility")
        return util

    def mu(self, c: np.ndarray, s: np.ndarray) -> np.ndarray:  # noqa: N802
        """Calculate the marginal utility ∂U/∂C.
        
        The marginal utility is given by:
            ∂U/∂C = (C*S)^(-γ)
            
        Args:
            c (np.ndarray): Array of consumption levels.
            s (np.ndarray): Array of surplus-consumption ratios.
            
        Returns:
            np.ndarray: Array of marginal utility values, same shape as inputs.
            
        Raises:
            ValueError: If any marginal utility values are non-finite.
        """
        γ = self.p.gamma
        mu_val = np.power(c * s, -γ)
        assert_finite(mu_val, "marginal utility")
        return mu_val

    # ------------------------------------------------------------------
    # Stochastic Discount Factor (SDF)
    # ------------------------------------------------------------------
    def sdf(
        self,
        g_c: np.ndarray,
        s_t: np.ndarray,
        s_tp1: np.ndarray,
    ) -> np.ndarray:  # noqa: N802
        """Calculate the stochastic discount factor M_{t+1}/M_t.
        
        The SDF is given by:
            M_{t+1}/M_t = β * (MU_{t+1}/MU_t)
            
        where β is the subjective discount factor and MU is the marginal utility.
        
        Args:
            g_c (np.ndarray): Array of log consumption growth rates (Δlog C_{t+1}).
            s_t (np.ndarray): Array of current surplus-consumption ratios.
            s_tp1 (np.ndarray): Array of next period surplus-consumption ratios.
            
        Returns:
            np.ndarray: Array of SDF values, same shape as inputs.
            
        Raises:
            ValueError: If any SDF values are non-finite.
            
        Note:
            All input arrays are assumed to be shape-aligned with the last axis
            representing the scenario step.
        """
        β = self.p.beta
        muf = self.mu(c=np.exp(g_c), s=s_tp1)
        mub = self.mu(c=np.ones_like(g_c), s=s_t)
        m = β * muf / mub
        assert_finite(m, "sdf")
        return m