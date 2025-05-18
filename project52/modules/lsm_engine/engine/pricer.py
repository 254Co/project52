"""
Longstaff-Schwartz Method (LSM) Pricer Implementation

This module implements the core pricing engine for American and Bermudan options using
the Longstaff-Schwartz Method. The implementation follows the algorithm described in
"Valuing American Options by Simulation: A Simple Least-Squares Approach" (2001).

The LSMPricer class handles:
- Backward induction for optimal exercise decisions
- Continuation value estimation using polynomial regression
- Present value calculation with statistical measures
- Confidence interval computation for Monte Carlo estimates
"""

# File: lsm_engine/engine/pricer.py
from __future__ import annotations
import numpy as np
import logging
from scipy.stats import norm
from typing import Callable, Union, Tuple
from ..config import CONFIG
from .regression import PolyRegressor

class LSMPricer:
    """
    Longstaff-Schwartz Method (LSM) pricer for American/Bermudan options.
    
    This class implements the core LSM algorithm for pricing American and Bermudan options.
    It uses polynomial regression to estimate continuation values and performs backward
    induction to determine optimal exercise decisions.
    
    Attributes:
        payoff_fn (Callable): Function that computes option payoffs at each time step
        discount (float): Discount factor for present value calculations
        regressor (PolyRegressor): Polynomial regressor for continuation value estimation
        log (Logger): Logger instance for tracking pricing progress
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
            payoff_fn: Function that computes option payoffs at each time step
            discount: Discount factor for present value calculations
            regressor: Optional custom polynomial regressor
            degree: Optional polynomial degree for regression (if regressor not provided)
        """
        self.payoff_fn = payoff_fn
        self.discount = discount
        self.regressor = regressor or PolyRegressor(degree or CONFIG.poly_degree)
        self.log = logging.getLogger("LSM")

    def price(self,
              paths: np.ndarray,
              exercise_idx: list[int],
              return_std_err: bool = False,
              return_ci: bool = False,
              alpha: float = 0.05
    ) -> Union[float, Tuple[float, float], Tuple[float, float, Tuple[float, float]]]:
        """
        Price the option using the Longstaff-Schwartz Method.
        
        This method implements the core LSM algorithm:
        1. Computes payoffs at each exercise date
        2. Performs backward induction to determine optimal exercise decisions
        3. Calculates present value and statistical measures
        
        Args:
            paths: Array of simulated price paths, shape (n_paths, n_steps)
            exercise_idx: List of time indices where exercise is allowed
            return_std_err: If True, return standard error with present value
            return_ci: If True, return confidence interval with present value and standard error
            alpha: Significance level for confidence interval (default: 0.05 for 95% CI)
            
        Returns:
            Union[float, Tuple[float, float], Tuple[float, float, Tuple[float, float]]]:
                - If return_ci=True: (present_value, standard_error, (ci_lower, ci_upper))
                - If return_std_err=True: (present_value, standard_error)
                - Otherwise: present_value
                
        Note:
            The standard error and confidence interval are computed using the standard
            Monte Carlo error estimation formulas, assuming normal distribution of results.
        """
        n_paths, n_steps = paths.shape
        cf = self.payoff_fn(paths)  # shape (n_paths, n_steps)
        values = cf[:, -1]

        # Backward induction
        for t in reversed(exercise_idx[:-1]):
            itm = cf[:, t] > 0.0
            x = paths[itm, t, None]
            y = values[itm] * self.discount
            if len(x) == 0:
                values[~itm] *= self.discount
                continue
            self.regressor.fit(x, y)
            continuation = self.regressor.predict(x)
            exercise = cf[itm, t] >= continuation
            values[itm] = np.where(exercise, cf[itm, t], values[itm] * self.discount)
            values[~itm] *= self.discount

        # Compute PV and statistics
        pv_per_path = values
        pv = float(pv_per_path.mean())
        stderr = float(pv_per_path.std(ddof=1) / np.sqrt(n_paths))

        if return_ci:
            z = norm.ppf(1 - alpha / 2)
            ci = (pv - z * stderr, pv + z * stderr)
            self.log.info("LSM completed | PV=%f | stderr=%f | CI=[%f, %f]", pv, stderr, ci[0], ci[1])
            return pv, stderr, ci
        if return_std_err:
            self.log.info("LSM completed | PV=%f | stderr=%f", pv, stderr)
            return pv, stderr

        self.log.info("LSM completed | PV=%f", pv)
        return pv