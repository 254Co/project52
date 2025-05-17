# -------------------- pricers/mc.py --------------------
"""
Monte Carlo pricer with Greeks support for the Chen3 package.

This module implements a Monte Carlo pricer with support for various
Greeks calculation methods, including:
- Pathwise differentiation for delta
- Likelihood ratio method for vega
- Adjoint algorithmic differentiation (AAD)
- Control variate variance reduction
"""

import numpy as np
from typing import Callable, Optional, Union, Dict, Any
from chen3.greeks import (
    delta_pathwise,
    vega_likelihood_ratio,
    greeks_aad,
    adjoint_control_variate,
)


class MonteCarloPricer:
    """
    Monte Carlo pricer with support for Greeks calculation.
    
    This class provides a flexible Monte Carlo pricer that can compute
    both option prices and Greeks using various methods. It supports
    different variance reduction techniques and Greeks calculation
    approaches.
    
    Attributes:
        payoff (Callable): Function to compute option payoffs
        discount_curve (Callable): Function to compute discount factors
        dt (float): Time increment per simulation step
        n_steps (int): Number of simulation steps
        use_cv (bool): Whether to use control variate
        use_aad (bool): Whether to use adjoint algorithmic differentiation
        cv_term (float): Known expectation of control variate
    """
    
    def __init__(
        self,
        payoff: Callable[[np.ndarray], np.ndarray],
        discount_curve: Callable[[float], float],
        dt: float,
        n_steps: int,
        use_cv: bool = False,
        use_aad: bool = False,
        cv_term: float = 0.0,
    ):
        """
        Initialize the Monte Carlo pricer.
        
        Args:
            payoff (Callable[[np.ndarray], np.ndarray]): Function that computes
                option payoffs from simulated paths. Takes an array of paths
                and returns an array of payoffs.
            discount_curve (Callable[[float], float]): Function that computes
                discount factors for given times. Takes a time and returns
                a discount factor.
            dt (float): Time increment per simulation step
            n_steps (int): Number of simulation steps
            use_cv (bool): Whether to enable control variate variance reduction
            use_aad (bool): Whether to enable adjoint algorithmic differentiation
            cv_term (float): Known expectation of control variate (if used)
        
        Note:
            The payoff function should be vectorized to handle arrays of paths
            efficiently. The discount curve should be a function of time that
            returns the appropriate discount factor.
        """
        self.payoff = payoff
        self.discount_curve = discount_curve
        self.dt = dt
        self.n_steps = n_steps
        self.use_cv = use_cv
        self.use_aad = use_aad
        self.cv_term = cv_term

    def price(
        self,
        paths: np.ndarray,
        greek: Optional[str] = None
    ) -> float:
        """
        Price the option or compute a Greek via Monte Carlo.
        
        This method can compute either the option price or various Greeks
        using different methods:
        - None: Standard Monte Carlo pricing
        - 'delta': Pathwise differentiation
        - 'vega': Likelihood ratio method
        - 'aad': Adjoint algorithmic differentiation
        - 'cv': Control variate method
        
        Args:
            paths (np.ndarray): Simulated paths array
            greek (Optional[str]): Which Greek to compute, or None for price
        
        Returns:
            float: Option price or Greek value
        
        Raises:
            ValueError: If an unsupported Greek is requested or if the
                required feature (AAD or control variate) is not enabled
        
        Note:
            The paths array should have shape (n_paths, n_steps) for
            standard pricing, or (n_paths, n_steps, n_factors) for
            multi-factor models.
        """
        T = self.dt * self.n_steps
        if greek is None:
            payoffs = self.payoff(paths)
            df = self.discount_curve(T)
            return float(np.mean(payoffs) * df)

        if greek == 'delta':
            return delta_pathwise(paths, self.payoff)
        if greek == 'vega':
            return vega_likelihood_ratio(paths, self.payoff)
        if greek == 'aad' and self.use_aad:
            return greeks_aad(paths, self.payoff)
        if greek == 'cv' and self.use_cv:
            return adjoint_control_variate(paths, self.payoff, self.cv_term)

        raise ValueError(f"Unsupported greek '{greek}' or feature not enabled.")
