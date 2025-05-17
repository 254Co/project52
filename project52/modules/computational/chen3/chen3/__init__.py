# File: chen3/__init__.py
"""
Chen3: A Comprehensive Three-Factor Chen Model Implementation

This package implements the three-factor Chen model for financial derivatives pricing,
combining interest rates, equity, and rough volatility processes. The model is particularly
suited for pricing complex financial instruments where traditional models may fall short.

Key Components:
- Interest Rate Process: CIR (Cox-Ingersoll-Ross) model for short rates
- Equity Process: Stochastic process for underlying asset prices
- Rough Volatility: Incorporates rough volatility dynamics for more realistic market behavior

Main Features:
- Multiple numerical schemes (Euler, ADF, Exact) for process simulation
- Monte Carlo pricing engine
- Support for various payoff types (Vanilla, Barrier, Asian, Convertible Bonds)
- Calibration tools for model parameters
- Correlation handling through Cholesky decomposition

Example Usage:
    >>> from chen3 import ChenModel, ModelParams
    >>> model = ChenModel(params=ModelParams(...))
    >>> price = model.price_instrument(...)

For detailed documentation and examples, please refer to the package documentation.
"""

from .config import Settings
from .datatypes import RateParams, EquityParams, ModelParams
from .model import ChenModel
from .processes.interest_rate import ShortRateCIR
from .processes.equity import EquityProcess
from .processes.rough_vol import RoughVariance
from .schemes.euler import euler_scheme
from .schemes.adf import adf_scheme
from .schemes.exact import exact_scheme
from .correlation import cholesky_correlation
from .simulators.core import PathGenerator, make_simulator
from .pricers.mc import MonteCarloPricer
from .calibration.optim import calibrate
from .payoffs import Vanilla, Barrier, Asian, ConvertibleBond

# Public API exports
__all__ = [
    # Configuration and parameter types
    "Settings", "RateParams", "EquityParams", "ModelParams",
    # Core model
    "ChenModel",
    # Stochastic processes
    "ShortRateCIR", "EquityProcess", "RoughVariance",
    # Numerical schemes
    "euler_scheme", "adf_scheme", "exact_scheme",
    # Utilities
    "cholesky_correlation", "make_simulator",
    # Pricing and calibration
    "MonteCarloPricer", "calibrate",
    # Financial instruments
    "Vanilla", "Barrier", "Asian", "ConvertibleBond"
]

# Version information
__version__ = "0.1.0"
