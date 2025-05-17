# File: chen3/payoffs/hybrid/convertible_credit.py
"""
Convertible bond payoff including credit default.
"""
import numpy as np
from typing import Any
from ..base import Payoff

class ConvertibleCreditBond(Payoff):
    def __init__(
        self,
        face_value: float,
        conversion_ratio: float,
        default_hazard: float,
        recovery_rate: float,
        dt: float
    ):
        """
        face_value: par amount
        conversion_ratio: shares per bond
        default_hazard: constant hazard rate per year
        recovery_rate: fraction of face_value recovered on default
        dt: time step size in years
        """
        self.face_value = face_value
        self.conversion_ratio = conversion_ratio
        self.hazard = default_hazard
        self.recovery = recovery_rate
        self.dt = dt

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        """
        paths: ndarray shape (n_paths, n_steps+1, factors)
        factors: [S, v, r]
        Simulate default as exponential with constant hazard.
        If default occurs before maturity, payoff = recovery * face_value.
        Else, payoff = max(face_value, conversion_ratio * S_T).
        """
        n_paths, n_steps_plus1, _ = paths.shape
        # simulate default times: draw uniform, convert to exponential
        u = np.random.rand(n_paths)
        # default time in years
        t_default = -np.log(u) / self.hazard
        # default step index
        default_step = np.floor(t_default / self.dt).astype(int)
        # maturity index
        mat_idx = n_steps_plus1 - 1

        S_T = paths[:, mat_idx, 0]
        # payoff if survives
        survive_payoff = np.maximum(self.face_value, self.conversion_ratio * S_T)
        # payoff array
        payoff = np.where(default_step <= mat_idx,
                          self.recovery * self.face_value,
                          survive_payoff)
        return payoff
