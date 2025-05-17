# File: chen3/pricers/mc.py

import numpy as np
from typing import Callable, Optional
from chen3.greeks import (
    delta_pathwise,
    vega_likelihood_ratio,
    greeks_aad,
    adjoint_control_variate,
)

class MonteCarloPricer:
    def __init__(
        self,
        payoff: Callable[[np.ndarray], np.ndarray],
        discount_curve: Callable[[float], float],
        dt: float,
        n_steps: int,
    ):
        """
        payoff:   paths -> (n_paths,) array of payoffs
        discount_curve: T -> discount factor
        dt:       time increment (in years)
        n_steps:  number of time steps per path
        """
        self.payoff = payoff
        self.discount_curve = discount_curve
        self.dt = dt
        self.n_steps = n_steps

    def price(self, paths: np.ndarray, greek: Optional[str] = None) -> float:
        """
        paths: ndarray of shape (n_paths, n_steps+1, n_factors)
        greek: None or one of {"delta","vega","aad","cv"}
        """
        T = self.dt * self.n_steps
        if greek is None:
            payoffs = self.payoff(paths)
            df = self.discount_curve(T)
            return float(np.mean(payoffs) * df)

        # route to Greek calculators
        if greek == "delta":
            return float(delta_pathwise(paths, self.payoff))
        if greek == "vega":
            return float(vega_likelihood_ratio(paths, self.payoff))
        if greek == "aad":
            return float(greeks_aad(paths, self.payoff))
        if greek == "cv":
            return float(adjoint_control_variate(paths, self.payoff))

        raise ValueError(f"Unknown greek '{greek}'. Valid: delta, vega, aad, cv.")
