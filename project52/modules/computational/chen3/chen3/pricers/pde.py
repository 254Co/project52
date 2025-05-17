# -------------------- pricers/pde.py --------------------
"""PDE pricer using finite-difference methods."""
import numpy as np


def solve_black_scholes_pde(
    S_max: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_S: int = 200,
    n_t: int = 200
) -> float:
    """
    Price European call via explicit FDM.
    """
    dS = S_max / n_S
    dt = T / n_t
    # grid
    S_grid = np.linspace(0, S_max, n_S+1)
    V = np.maximum(S_grid - K, 0)

    # step backwards in time
    for j in range(n_t):
        t = j * dt
        V_new = V.copy()
        for i in range(1, n_S):
            delta = (V[i+1] - V[i-1]) / (2 * dS)
            gamma = (V[i+1] - 2*V[i] + V[i-1]) / (dS**2)
            V_new[i] = V[i] + dt * (
                0.5 * sigma**2 * S_grid[i]**2 * gamma
                + r * S_grid[i] * delta
                - r * V[i]
            )
        V = V_new
        V[0] = 0
        V[-1] = S_max - K * np.exp(-r * (T - j * dt))
    return float(np.interp(K, S_grid, V))


class PDEPricer:
    """
    PDE-based pricer for European calls via finite-difference.
    """
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
        """
        Solve the PDE and return the option price.
        """
        return solve_black_scholes_pde(
            self.S_max, self.K, self.r, self.sigma, self.T,
            self.n_S, self.n_t
        )