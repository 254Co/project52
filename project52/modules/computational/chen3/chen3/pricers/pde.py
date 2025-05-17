# -------------------- pricers/pde.py --------------------
"""
PDE-based option pricing using finite-difference methods.

This module implements a finite-difference method (FDM) solver for the
Black-Scholes PDE to price European call options. It uses an explicit
time-stepping scheme with central differences for spatial derivatives.
"""

from typing import Optional, Tuple

import numpy as np


def solve_black_scholes_pde(
    S_max: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_S: int = 200,
    n_t: int = 200,
) -> float:
    """
    Price a European call option using explicit finite-difference method.

    This function solves the Black-Scholes PDE using an explicit FDM scheme:
    - Central differences for spatial derivatives (delta and gamma)
    - Forward Euler for time stepping
    - Linear boundary conditions
    - Final condition: max(S-K, 0)

    Args:
        S_max (float): Maximum stock price in the grid
        K (float): Strike price
        r (float): Risk-free interest rate
        sigma (float): Volatility
        T (float): Time to maturity
        n_S (int): Number of spatial grid points (default: 200)
        n_t (int): Number of time steps (default: 200)

    Returns:
        float: Option price at S=K

    Note:
        The method uses a uniform grid in both space and time.
        The stability condition dt <= dx^2/(2*sigma^2*S^2) should be
        satisfied for accurate results.
    """
    dS = S_max / n_S
    dt = T / n_t
    # grid
    S_grid = np.linspace(0, S_max, n_S + 1)
    V = np.maximum(S_grid - K, 0)

    # step backwards in time
    for j in range(n_t):
        t = j * dt
        V_new = V.copy()
        for i in range(1, n_S):
            delta = (V[i + 1] - V[i - 1]) / (2 * dS)
            gamma = (V[i + 1] - 2 * V[i] + V[i - 1]) / (dS**2)
            V_new[i] = V[i] + dt * (
                0.5 * sigma**2 * S_grid[i] ** 2 * gamma
                + r * S_grid[i] * delta
                - r * V[i]
            )
        V = V_new
        V[0] = 0
        V[-1] = S_max - K * np.exp(-r * (T - j * dt))
    return float(np.interp(K, S_grid, V))


class PDEPricer:
    """
    PDE-based pricer for European call options using finite-difference methods.

    This class provides a convenient interface for pricing European call
    options using the explicit finite-difference method. It encapsulates
    the grid parameters and pricing logic in a single object.

    Attributes:
        S_max (float): Maximum stock price in the grid
        K (float): Strike price
        r (float): Risk-free interest rate
        sigma (float): Volatility
        T (float): Time to maturity
        n_S (int): Number of spatial grid points
        n_t (int): Number of time steps
    """

    def __init__(
        self,
        S_max: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
        n_S: int = 200,
        n_t: int = 200,
    ):
        """
        Initialize the PDE pricer with model parameters.

        Args:
            S_max (float): Maximum stock price in the grid
            K (float): Strike price
            r (float): Risk-free interest rate
            sigma (float): Volatility
            T (float): Time to maturity
            n_S (int): Number of spatial grid points (default: 200)
            n_t (int): Number of time steps (default: 200)
        """
        self.S_max = S_max
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.n_S = n_S
        self.n_t = n_t

    def price(self) -> float:
        """
        Solve the Black-Scholes PDE and return the option price.

        Returns:
            float: The option price at the strike price

        Note:
            This method uses the explicit finite-difference scheme
            implemented in solve_black_scholes_pde.
        """
        return solve_black_scholes_pde(
            self.S_max, self.K, self.r, self.sigma, self.T, self.n_S, self.n_t
        )
