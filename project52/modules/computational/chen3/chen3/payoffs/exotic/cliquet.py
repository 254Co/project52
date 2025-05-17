# -------------------- chen3/payoffs/exotic/cliquet.py --------------------
"""
Cliquet option payoff: sum of capped periodic returns.
"""
import numpy as np
from typing import List
from ..base import Payoff

class Cliquet(Payoff):
    def __init__(self,
                 observation_times: List[int],
                 strike: float,
                 cap: float,
                 floor: float = 0.0,
                 notional: float = 1.0):
        """
        observation_times: indices of time steps to observe returns
        strike: baseline strike level
        cap: maximum return per period
        floor: minimum return per period
        notional: pay structural notional
        """
        self.obs = observation_times
        self.strike = strike
        self.cap = cap
        self.floor = floor
        self.notional = notional

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        # paths shape: (n_paths, n_steps+1, factors), factor0 = S
        S = paths[:, :, 0]
        # compute returns per period
        returns = []
        for i in range(len(self.obs)-1):
            t0, t1 = self.obs[i], self.obs[i+1]
            r = (S[:, t1] - S[:, t0]) / S[:, t0]
            returns.append(np.clip(r, self.floor, self.cap))
        total = np.sum(np.stack(returns, axis=1), axis=1)
        return self.notional * total