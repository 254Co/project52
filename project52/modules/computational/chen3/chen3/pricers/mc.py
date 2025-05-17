# -------------------- pricers/mc.py --------------------
"""Monte Carlo pricer with Greeks support."""
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
        use_cv: bool = False,
        use_aad: bool = False,
        cv_term: float = 0.0,
    ):
        """
        payoff: function paths -> (n_paths,) vector
        discount_curve: function T -> discount factor
        dt: time increment per step
        n_steps: number of steps
        use_cv: enable control variate
        use_aad: enable AAD
        cv_term: known E[control_variate]
        """
        self.payoff = payoff
        self.discount_curve = discount_curve
        self.dt = dt
        self.n_steps = n_steps
        self.use_cv = use_cv
        self.use_aad = use_aad
        self.cv_term = cv_term

    def price(self, paths: np.ndarray, greek: Optional[str] = None) -> float:
        """
        Price or compute Greek via Monte Carlo.
        greek: None, 'delta','vega','aad','cv'
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
