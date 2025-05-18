"""Derived metrics: risk‑free rate, equity premium, volatility, Sharpe, etc.

This module provides analytical tools for computing key financial metrics from the
Campbell-Cochrane habit formation model, including risk-free rates, equity premiums,
Sharpe ratios, and other market-relevant statistics. These metrics are computed
from simulated paths of consumption growth and surplus-consumption ratios.
"""
from __future__ import annotations

import numpy as np

from .model import HabitModel
from ..utils.validation import assert_finite

__all__ = ["Analytics"]


class Analytics:
    """Analytics computation wrapper for the Campbell-Cochrane habit formation model.
    
    This class provides methods to compute various financial metrics and statistics
    from simulated paths of the Campbell-Cochrane model. It wraps a HabitModel instance
    and provides convenient methods for analyzing model outputs.
    
    Attributes:
        model (HabitModel): The underlying habit formation model.
    """

    def __init__(self, model: HabitModel):
        """Initialize the analytics wrapper with a habit model.
        
        Args:
            model (HabitModel): The habit formation model to analyze.
        """
        self.model = model

    # ------------------------------------------------------------------
    # Risk‑free rate per period (log)
    # ------------------------------------------------------------------
    @staticmethod
    def _mean_log(x: np.ndarray, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
        """Compute the log of the mean of an array.
        
        Args:
            x (np.ndarray): Input array.
            axis (int | tuple[int, ...] | None): Axis or axes along which to compute the mean.
                If None, compute over the entire array.
                
        Returns:
            np.ndarray: Log of the mean of x along the specified axis.
        """
        return np.log(np.mean(x, axis=axis))

    def risk_free_rate(
        self,
        g_c: np.ndarray,
        s_t: np.ndarray,
        s_tp1: np.ndarray,
        *,
        annualize: bool = False,
        periods_per_year: int = 12,
    ) -> np.ndarray:
        """Compute the risk-free rate r_f = -log E[M_{t+1}].
        
        The risk-free rate is computed as the negative log of the expected
        stochastic discount factor. This represents the continuously compounded
        risk-free return.
        
        Args:
            g_c (np.ndarray): Array of log consumption growth rates.
            s_t (np.ndarray): Array of current surplus-consumption ratios.
            s_tp1 (np.ndarray): Array of next period surplus-consumption ratios.
            annualize (bool, optional): Whether to annualize the rate. Defaults to False.
            periods_per_year (int, optional): Number of periods per year for annualization.
                Defaults to 12 (monthly).
                
        Returns:
            np.ndarray: Array of risk-free rates with shape (n_paths,).
            
        Raises:
            ValueError: If any computed rates are non-finite.
        """
        m = self.model.sdf(g_c, s_t, s_tp1)
        rf = -self._mean_log(m, axis=1)
        if annualize:
            rf *= periods_per_year
        assert_finite(rf, "r_f")
        return rf

    # ------------------------------------------------------------------
    # Equity premium & Sharpe
    # ------------------------------------------------------------------
    def equity_premium(
        self,
        re: np.ndarray,
        m: np.ndarray,
        *,
        annualize: bool = False,
        periods_per_year: int = 12,
    ) -> np.ndarray:
        """Compute the equity premium E[log R_e] - r_f.
        
        The equity premium is the expected excess return of equity over the
        risk-free rate. It represents the compensation investors require for
        bearing systematic risk.
        
        Args:
            re (np.ndarray): Array of equity returns.
            m (np.ndarray): Array of stochastic discount factors.
            annualize (bool, optional): Whether to annualize the premium.
                Defaults to False.
            periods_per_year (int, optional): Number of periods per year for
                annualization. Defaults to 12 (monthly).
                
        Returns:
            np.ndarray: Array of equity premiums with shape (n_paths,).
        """
        rf = -self._mean_log(m, axis=1)
        er = self._mean_log(re, axis=1)
        prem = er - rf
        if annualize:
            prem *= periods_per_year
        return prem

    def sharpe_ratio(
        self,
        re: np.ndarray,
        m: np.ndarray,
    ) -> np.ndarray:
        """Compute the Sharpe ratio of excess returns.
        
        The Sharpe ratio measures the risk-adjusted return of an investment.
        It is computed as the mean excess return divided by the standard
        deviation of excess returns.
        
        Args:
            re (np.ndarray): Array of equity returns.
            m (np.ndarray): Array of stochastic discount factors.
            
        Returns:
            np.ndarray: Array of Sharpe ratios with shape (n_paths,).
        """
        rf = -self._mean_log(m, axis=1)
        excess = np.log(re) - rf[:, None]
        sr = excess.mean(axis=1) / excess.std(axis=1, ddof=1)
        return sr

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def summary(self, paths: dict[str, np.ndarray]) -> dict[str, float]:
        """Compute a summary of key financial metrics averaged across paths.
        
        This method computes several important financial metrics from the
        simulated paths and returns them as a dictionary. The metrics include
        the risk-free rate, equity premium, and Sharpe ratio.
        
        Args:
            paths (dict[str, np.ndarray]): Dictionary containing simulated paths.
                Must include keys:
                - "S": Surplus-consumption ratios
                - "g_c": Log consumption growth rates
                
        Returns:
            dict[str, float]: Dictionary containing:
                - "E_r_f": Average risk-free rate
                - "Equity_Premium": Average equity premium
                - "Sharpe": Average Sharpe ratio
        """
        s_t, s_tp1 = paths["S"][:, :-1], paths["S"][:, 1:]
        g_c = paths["g_c"]
        m = self.model.sdf(g_c, s_t, s_tp1)
        rf = self.risk_free_rate(g_c, s_t, s_tp1).mean()
        # Synthetic equity return: assume log R_e = g_c + SDF‑implied risk prem
        re = np.exp(g_c)  # placeholder; replace with your asset process
        prem = self.equity_premium(re, m).mean()
        sr = self.sharpe_ratio(re, m).mean()
        return {"E_r_f": rf, "Equity_Premium": prem, "Sharpe": sr}