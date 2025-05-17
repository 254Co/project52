# File: chen3/model_enhancements/jump_diffusion.py
"""
Merton and Kou jump-diffusion processes.
"""
import numpy as np
from numpy.typing import NDArray

class MertonJump:
    def __init__(self, lam: float, mu_j: float, sigma_j: float):
        """
        lam: jump intensity (lambda)
        mu_j: mean of log-jump size
        sigma_j: std dev of log-jump size
        """
        self.lam = lam
        self.mu_j = mu_j
        self.sigma_j = sigma_j

    def simulate(self, dt: float, n_steps: int, n_paths: int, rng: np.random.Generator) -> NDArray:
        """
        Returns jump multipliers of shape (n_paths, n_steps).
        """
        # number of jumps in each interval
        N = rng.poisson(self.lam * dt, size=(n_paths, n_steps))
        # sum of lognormal jumps per interval
        jumps = np.exp(
            rng.normal(self.mu_j, self.sigma_j, size=(n_paths, n_steps)) * N
        )
        return jumps

class KouJump:
    def __init__(
        self,
        lam: float,
        p: float,
        eta1: float,
        eta2: float
    ):
        """
        lam: jump intensity
        p: probability of up-jump
        eta1: positive jump parameter (scale)
        eta2: negative jump parameter (scale)
        """
        self.lam = lam
        self.p = p
        self.eta1 = eta1
        self.eta2 = eta2

    def simulate(self, dt: float, n_steps: int, n_paths: int, rng: np.random.Generator) -> NDArray:
        """
        Returns jump multipliers of shape (n_paths, n_steps).
        """
        N = rng.poisson(self.lam * dt, size=(n_paths, n_steps))
        # for each jump, sample sizes
        U = rng.uniform(size=(n_paths, n_steps))
        jumps = np.ones((n_paths, n_steps))
        # where jumps occur
        mask = N > 0
        # total jump size = product of k i.i.d. sizes, approximate by exp(k * E[size])
        # for simplicity, approximate log-jump ~ N * E[log(Z)]
        # E[log(Z)] = p*(1/eta1) + (1-p)*(-1/eta2)
        mean_log = self.p*(1/self.eta1) + (1-self.p)*(-1/self.eta2)
        jumps[mask] = np.exp(mean_log * N[mask])
        return jumps