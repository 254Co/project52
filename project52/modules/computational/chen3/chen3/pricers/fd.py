# -------------------- pricers/fd.py --------------------
"""
Finite-difference Pricer wrappers for the Chen3 package.

This module provides wrapper classes for finite-difference based option pricing
methods. It serves as a high-level interface to the underlying PDE solvers
implemented in the pde module.
"""

from typing import Optional

from .pde import solve_black_scholes_pde


class FDPricer:
    """
    Finite-difference based option pricer.

    This class provides a high-level interface for pricing options using
    finite-difference methods. It wraps the underlying PDE solver with
    a more user-friendly interface and parameter management.

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
        Initialize the finite-difference pricer.

        Args:
            S_max (float): Maximum stock price in the grid
            K (float): Strike price
            r (float): Risk-free interest rate
            sigma (float): Volatility
            T (float): Time to maturity
            n_S (int): Number of spatial grid points (default: 200)
            n_t (int): Number of time steps (default: 200)

        Note:
            The grid parameters (n_S, n_t) should be chosen to ensure
            numerical stability and accuracy of the solution.
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
        Price the option using finite-difference methods.

        This method uses the explicit finite-difference solver from the
        pde module to compute the option price.

        Returns:
            float: The option price at the strike price

        Note:
            The pricing is performed using an explicit finite-difference
            scheme with central differences for spatial derivatives.
        """
        return solve_black_scholes_pde(
            self.S_max, self.K, self.r, self.sigma, self.T, self.n_S, self.n_t
        )
