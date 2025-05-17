# File: chen3/model_enhancements/stochastic_dividend.py
"""
Stochastic dividend yield model enhancement for the Chen3 model.

This module implements a stochastic dividend yield process using an
Ornstein-Uhlenbeck (OU) process, which allows for modeling mean-reverting
dividend yields with random fluctuations.

The implementation provides:
- Mean-reverting dividend yield process
- Configurable mean-reversion parameters
- Simulation of dividend yield paths
- Realistic dividend yield dynamics

This enhancement is particularly useful for modeling:
- Time-varying dividend yields
- Mean-reverting dividend behavior
- Dividend yield risk
- Dividend term structure
"""

from typing import Generator, Optional

import numpy as np
from numpy.typing import NDArray


class StochasticDividend:
    """
    Stochastic dividend yield modeled as an Ornstein-Uhlenbeck process.

    This class implements a stochastic dividend yield process where
    the yield follows a mean-reverting Ornstein-Uhlenbeck process.
    This allows for modeling realistic dividend yield dynamics with
    mean reversion and random fluctuations.

    Mathematical Formulation:
    ----------------------
    The dividend yield y(t) follows the SDE:

    dy(t) = κ(θ - y(t))dt + σdW(t)

    where:
    - κ: Mean-reversion speed
    - θ: Long-term mean dividend yield
    - σ: Volatility of dividend yield
    - W(t): Standard Brownian motion

    The process has the following properties:
    - Mean-reverting to θ
    - Stationary distribution: N(θ, σ²/(2κ))
    - Autocorrelation: exp(-κt)

    Attributes:
        kappa (float): Mean-reversion speed
        theta (float): Long-term mean dividend yield
        sigma (float): Volatility of dividend yield
        y0 (float): Initial dividend yield

    Note:
        The process ensures that dividend yields remain realistic
        through mean reversion, while allowing for temporary
        deviations from the long-term mean.
    """

    def __init__(self, kappa: float, theta: float, sigma: float, y0: float):
        """
        Initialize stochastic dividend yield process.

        Args:
            kappa (float): Mean-reversion speed (positive)
            theta (float): Long-term mean dividend yield
            sigma (float): Volatility of dividend yield (positive)
            y0 (float): Initial dividend yield

        Raises:
            ValueError: If kappa or sigma is non-positive
        """
        if kappa <= 0:
            raise ValueError("Mean-reversion speed must be positive")
        if sigma <= 0:
            raise ValueError("Volatility must be positive")

        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.y0 = y0

    def simulate(
        self, dt: float, n_steps: int, n_paths: int, rng: np.random.Generator
    ) -> NDArray:
        """
        Simulate dividend yield paths.

        This method simulates multiple paths of the dividend yield
        process using an Euler-Maruyama discretization of the
        Ornstein-Uhlenbeck SDE.

        Args:
            dt (float): Time step size
            n_steps (int): Number of time steps
            n_paths (int): Number of paths to simulate
            rng (np.random.Generator): Random number generator

        Returns:
            NDArray: Array of dividend yield paths of shape
                (n_paths, n_steps+1) where paths[i,j] is the
                dividend yield for path i at time step j

        Note:
            The simulation uses the Euler-Maruyama scheme:
            y(t+dt) = y(t) + κ(θ-y(t))dt + σ√dt * N(0,1)
        """
        # Initialize paths with initial value
        y = np.full(n_paths, self.y0)
        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = y

        # Simulate paths
        for t in range(1, n_steps + 1):
            # Compute drift and diffusion terms
            drift = self.kappa * (self.theta - y) * dt
            diffusion = self.sigma * np.sqrt(dt) * rng.standard_normal(n_paths)

            # Update yield
            y = y + drift + diffusion
            paths[:, t] = y

        return paths
