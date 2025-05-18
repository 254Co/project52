"""
Longstaff-Schwartz Method (LSM) pricing engine for American/Bermudan options.

This module implements the core pricing algorithm for American and Bermudan options
using the Longstaff-Schwartz Method. The algorithm uses Monte Carlo simulation
combined with least-squares regression to estimate the optimal exercise strategy.
"""

# File: lsm_engine/engine/pricer.py
from __future__ import annotations
import numpy as np
import logging
from typing import Callable
from ..config import CONFIG
from .regression import PolyRegressor

class LSMPricer:
    """
    Longstaff-Schwartz pricer for American/Bermudan options.
    
    This class implements the core LSM algorithm for pricing American and Bermudan options.
    It uses Monte Carlo simulation paths and polynomial regression to estimate the
    optimal exercise strategy at each exercise date.
    
    Attributes:
        payoff_fn (Callable): Function that computes option payoffs
        discount (float): Discount factor for present value calculations
        regressor (PolyRegressor): Regression model for continuation values
        log (logging.Logger): Logger instance for tracking pricing progress
    """
    def __init__(
        self,
        payoff_fn: Callable[[np.ndarray], np.ndarray],
        discount: float,
        regressor: PolyRegressor | None = None,
        degree: int | None = None,
    ):
        """
        Initialize the LSM pricer.
        
        Args:
            payoff_fn: Function that computes option payoffs for given paths
            discount: Discount factor for present value calculations
            regressor: Optional custom regression model
            degree: Optional polynomial degree for regression (if regressor not provided)
        """
        self.payoff_fn = payoff_fn
        self.discount = discount
        self.regressor = regressor or PolyRegressor(degree or CONFIG.poly_degree)
        self.log = logging.getLogger("LSM")

    def price(self, paths: np.ndarray, exercise_idx: list[int]) -> float:
        """
        Price an American/Bermudan option using the LSM algorithm.
        
        Args:
            paths: Monte Carlo simulation paths (n_paths, n_steps)
            exercise_idx: List of indices where exercise is allowed
            
        Returns:
            float: Present value of the option
        """
        n_paths, n_steps = paths.shape
        cf = self.payoff_fn(paths)               # cash-flows matrix (n_paths, n_steps)
        values = cf[:, -1]                       # start at maturity

        for t in reversed(exercise_idx[:-1]):
            itm = cf[:, t] > 0.0                 # in-the-money paths
            x = paths[itm, t, None]              # states
            y = values[itm] * self.discount      # discounted continuation values
            if len(x) == 0:
                continue
            self.regressor.fit(x, y)             # fit regression model
            continuation = self.regressor.predict(x)  # predict continuation values
            exercise = cf[itm, t] >= continuation  # exercise if payoff > continuation
            values[itm] = np.where(exercise, cf[itm, t], values[itm] * self.discount)
            values[~itm] *= self.discount        # discount out-of-the-money paths

        pv = float(values.mean())               # average across all paths
        self.log.info("LSM completed | PV=%f", pv)
        return pv