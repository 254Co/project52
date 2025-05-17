# -------------------- chen3/payoffs/exotic/autocallable.py --------------------
"""
Autocallable/Snowball payoff with early redemption and coupons.
"""
import numpy as np
from typing import List
from ..base import Payoff


class Autocallable(Payoff):
    def __init__(self,
                 obs_times: List[int],
                 barrier: float,
                 coupon: float,
                 maturity: int,
                 notional: float = 1.0):
        """
        obs_times: observation indices (including maturity)
        barrier: redemption barrier level
        coupon: coupon paid per obs if not redeemed
        maturity: final time index
        notional: face notional
        """
        self.obs = obs_times
        self.barrier = barrier
        self.coupon = coupon
        self.maturity = maturity
        self.notional = notional

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        S = paths[:, :, 0]
        payoffs = np.zeros(paths.shape[0])
        redeemed = np.zeros(paths.shape[0], dtype=bool)
        for t in self.obs:
            idx = (~redeemed) & (S[:, t] >= self.barrier)
            payoffs[idx] = self.notional * (1 + self.coupon)
            redeemed[idx] = True
            # stop further observation
        # if not redeemed by maturity, pay coupon schedule
        payoffs[~redeemed] = self.notional * (1 + self.coupon * len(self.obs))
        return payoffs
