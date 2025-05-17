# File: chen3/payoffs/hybrid/equity_rate_hybrid.py
"""
Equity-rate hybrid product payoff, e.g., equity-linked note.
"""
from typing import Any

import numpy as np

from ..base import Payoff


class EquityRateHybridProduct(Payoff):
    def __init__(
        self,
        notional: float,
        equity_strike: float,
        coupon_rate: float,
        r_obs_times: Any,  # indices for rate observations
    ):
        """
        notional: principal amount
        equity_strike: strike for equity option component
        coupon_rate: fixed rate paid if rate path stays above barrier
        r_obs_times: list of time-step indices for rate barrier observation
        """
        self.notional = notional
        self.equity_strike = equity_strike
        self.coupon_rate = coupon_rate
        self.r_obs = r_obs_times

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        """
        paths: ndarray shape (n_paths, n_steps+1, factors)
        factors: [S, v, r]
        Payoff = vanilla European call on equity + rate barrier coupon.
        The coupon pays if all observed rates exceed coupon_rate.
        """
        S = paths[:, :, 0]
        r = paths[:, :, 2]
        # equity payoff
        S_T = S[:, -1]
        eq_payoff = np.maximum(S_T - self.equity_strike, 0.0)
        # rate coupon condition
        barrier_met = np.all(r[:, self.r_obs] >= self.coupon_rate, axis=1)
        coupon_pay = self.notional * self.coupon_rate * barrier_met.astype(float)
        return self.notional * eq_payoff + coupon_pay
