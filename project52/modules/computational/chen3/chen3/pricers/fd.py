# -------------------- pricers/fd.py --------------------
"""Finite-difference Pricer wrappers."""
from .pde import solve_black_scholes_pde

class FDPricer:
    def __init__(
        self,
        S_max: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
        n_S: int = 200,
        n_t: int = 200
    ):
        self.S_max = S_max
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.n_S = n_S
        self.n_t = n_t

    def price(self) -> float:
        return solve_black_scholes_pde(
            self.S_max, self.K, self.r, self.sigma, self.T,
            self.n_S, self.n_t
        )
