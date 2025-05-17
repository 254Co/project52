# -------------------- chen3/payoffs/exotic/callable_puttable.py --------------------
"""
Callable/Puttable structure with optional early exercise.
"""
import numpy as np
from typing import List
from ..base import Payoff

class CallablePuttable(Payoff):
    def __init__(self,
                 obs_times: List[int],
                 call_strike: float,
                 put_strike: float,
                 notional: float = 1.0):
        """
        obs_times: time indices for possible exercise
        call_strike: strike at which issuer can call (sell) the instrument
        put_strike: strike at which holder can put (sell back)
        notional: principal amount
        """
        self.obs = obs_times
        self.call_strike = call_strike
        self.put_strike = put_strike
        self.notional = notional

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        S = paths[:, :, 0]
        payoff = np.full(paths.shape[0], -np.inf)
        # at each obs, determine optimal exercise
        for t in self.obs:
            # issuer calls: holder receives call_strike
            call_pay = np.full(paths.shape[0], self.call_strike)
            # holder puts: holder receives S[t]
            put_pay = S[:, t]
            # holder chooses max, issuer's call forces that
            pay_t = np.maximum(call_pay, put_pay)
            payoff = np.maximum(payoff, pay_t)
        # if never exercised, payoff at maturity S_T
        S_T = S[:, -1]
        payoff = np.where(payoff == -np.inf, S_T, payoff)
        return self.notional * payoff
