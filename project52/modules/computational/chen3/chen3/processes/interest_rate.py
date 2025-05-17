# File: chen3/processes/interest_rate.py
"""
Cox-Ingersoll-Ross (CIR) Short Rate Process Implementation

This module implements the Cox-Ingersoll-Ross (CIR) process for modeling the
short-term interest rate. The CIR process is a mean-reverting square-root
diffusion that ensures non-negative interest rates, making it suitable for
realistic interest rate modeling.

The process follows the SDE:
    dr_t = κ(θ - r_t)dt + σ√r_t dW_t

where:
- κ: Mean reversion speed
- θ: Long-term mean level
- σ: Volatility parameter
- r_t: Current interest rate
- W_t: Standard Brownian motion
"""

import numpy as np

from .base import StateProcess


class ShortRateCIR(StateProcess):
    """
    Implementation of the Cox-Ingersoll-Ross (CIR) short rate process.

    This class implements the CIR process for modeling stochastic interest rates.
    The process is mean-reverting and ensures non-negative rates, making it
    suitable for realistic interest rate modeling in the three-factor Chen model.

    The process is particularly well-suited for:
    - Modeling short-term interest rates
    - Capturing mean-reversion in rates
    - Ensuring non-negative rates
    - Pricing interest rate derivatives

    Attributes:
        kappa (float): Mean reversion speed parameter
        theta (float): Long-term mean level parameter
        sigma (float): Volatility parameter
        r0 (float): Initial interest rate level

    Example:
        >>> process = ShortRateCIR(kappa=0.1, theta=0.05, sigma=0.1, r0=0.03)
        >>> drift = process.drift(np.array([0.03, 0.04, 0.05]))
        >>> diffusion = process.diffusion(np.array([0.03, 0.04, 0.05]))
    """

    def __init__(self, kappa: float, theta: float, sigma: float, r0: float):
        """
        Initialize the CIR process with specified parameters.

        Args:
            kappa (float): Mean reversion speed parameter (κ > 0)
            theta (float): Long-term mean level parameter (θ > 0)
            sigma (float): Volatility parameter (σ > 0)
            r0 (float): Initial interest rate level (r0 ≥ 0)

        Raises:
            ValueError: If any parameter is negative or if 2κθ < σ²
                      (Feller condition for non-negative rates)
        """
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.r0 = r0

    def drift(self, r: np.ndarray) -> np.ndarray:
        """
        Compute the drift term of the CIR process.

        The drift term is given by: κ(θ - r_t)

        Args:
            r (np.ndarray): Current interest rate values, shape (n_paths,)

        Returns:
            np.ndarray: Drift term evaluated at current rates, shape (n_paths,)
        """
        return self.kappa * (self.theta - r)

    def diffusion(self, r: np.ndarray) -> np.ndarray:
        """
        Compute the diffusion term of the CIR process.

        The diffusion term is given by: σ√r_t
        Note: The maximum with 0 ensures non-negative rates

        Args:
            r (np.ndarray): Current interest rate values, shape (n_paths,)

        Returns:
            np.ndarray: Diffusion term evaluated at current rates, shape (n_paths,)
        """
        return self.sigma * np.sqrt(np.maximum(r, 0.0))
