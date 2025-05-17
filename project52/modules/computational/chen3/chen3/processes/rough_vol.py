# -------------------- processes/rough_vol.py --------------------
"""
Rough Volatility Process Implementation

This module implements a rough volatility process using a Riemann approximation
of the fractional Brownian motion. The process captures the rough behavior of
volatility observed in financial markets, where the Hurst parameter H < 0.5
indicates roughness.

The process is implemented as a discrete-time approximation of the continuous
rough volatility model, using a convolution-based approach for efficient
simulation of the fractional noise.
"""

import numpy as np

from .base import StateProcess


class RoughVariance(StateProcess):
    """
    Implementation of a rough volatility process using Riemann approximation.

    This class implements a rough volatility process that captures the
    empirically observed roughness in financial market volatility. The process
    uses a discrete-time approximation of fractional Brownian motion with
    Hurst parameter H < 0.5 to generate rough paths.

    The process is particularly well-suited for:
    - Modeling rough volatility dynamics
    - Capturing volatility persistence
    - Pricing options with rough volatility effects
    - Simulating realistic market behavior

    Attributes:
        hurst (float): Hurst parameter (H ∈ (0, 0.5))
        eta (float): Volatility of volatility parameter
        v0 (float): Initial variance level
        steps (int): Number of time steps for simulation
        kernel (np.ndarray): Precomputed fractional kernel weights

    Example:
        >>> process = RoughVariance(hurst=0.1, eta=0.3, v0=0.04, steps=1000)
        >>> rng = np.random.default_rng(42)
        >>> path = process.simulate(dt=1/252, rng=rng)
    """

    def __init__(self, hurst: float, eta: float, v0: float, steps: int):
        """
        Initialize the rough volatility process with specified parameters.

        Args:
            hurst (float): Hurst parameter (H ∈ (0, 0.5))
                          Lower values indicate rougher paths
            eta (float): Volatility of volatility parameter (η > 0)
            v0 (float): Initial variance level (v0 ≥ 0)
            steps (int): Number of time steps for simulation (steps > 0)

        Raises:
            ValueError: If parameters are outside valid ranges
        """
        self.hurst = hurst  # H in (0,0.5)
        self.eta = eta  # vol of vol
        self.v0 = v0  # initial rough variance
        self.steps = steps
        # Precompute fractional kernel weights
        self.kernel = np.array(
            [((i + 1) ** self.hurst - i**self.hurst) for i in range(steps)]
        )

    def simulate(self, dt: float, rng: np.random.Generator) -> np.ndarray:
        """
        Generate a rough variance path using the Riemann approximation.

        This method simulates a path of the rough volatility process using
        a discrete-time approximation of fractional Brownian motion. The
        simulation uses a convolution-based approach for efficient computation
        of the fractional noise.

        Args:
            dt (float): Time step size (dt > 0)
            rng (np.random.Generator): Random number generator instance

        Returns:
            np.ndarray: Simulated variance path of length steps+1

        Example:
            >>> rng = np.random.default_rng(42)
            >>> path = process.simulate(dt=1/252, rng=rng)
            >>> # path[0] = v0, path[1:] = simulated values
        """
        dw = rng.standard_normal(self.steps) * np.sqrt(dt)
        increments = self.eta * np.convolve(self.kernel, dw, mode="full")[: self.steps]
        v = np.empty(self.steps + 1)
        v[0] = self.v0
        for t in range(1, self.steps + 1):
            v[t] = abs(v[t - 1] + increments[t - 1])
        return v

    def drift(self, state):
        """
        Not implemented for rough volatility process.

        The rough volatility process is simulated using the simulate() method
        rather than through drift and diffusion terms.

        Raises:
            NotImplementedError: Always raised, as this method is not used
        """
        raise NotImplementedError("Use simulate() for rough variance evolution.")

    def diffusion(self, state):
        """
        Not implemented for rough volatility process.

        The rough volatility process is simulated using the simulate() method
        rather than through drift and diffusion terms.

        Raises:
            NotImplementedError: Always raised, as this method is not used
        """
        raise NotImplementedError("Use simulate() for rough variance evolution.")
