"""
LSM Engine - A Longstaff-Schwartz Method Implementation for American/Bermudan Option Pricing

This module provides a comprehensive implementation of the Longstaff-Schwartz Method (LSM)
for pricing American and Bermudan options. The implementation follows the algorithm
described in "Valuing American Options by Simulation: A Simple Least-Squares Approach" (2001).

Key Features:
- Path generation using Geometric Brownian Motion (GBM)
- Polynomial regression for continuation value estimation
- Configurable pricing engine with various payoff functions
- Support for American, Bermudan, and European options
- Monte Carlo simulation with variance reduction techniques
- Structured logging and performance monitoring
- Environment-based configuration system

Main Components:
1. Core Engine:
   - LSMPricer: Core pricing engine implementing the LSM algorithm
   - PolyRegressor: Polynomial regression for continuation value estimation
   - generate_gbm_paths: Path generation using GBM

2. Payoff Functions:
   - American options (put/call)
   - Bermudan options (put/call)
   - European options (put/call)
   - Custom payoff framework for exotic options

3. Utilities:
   - Structured JSON logging
   - Performance timing tools
   - Configuration management

Example Usage:
    >>> from lsm_engine import LSMPricer, generate_gbm_paths, american_put
    >>> 
    >>> # Generate price paths
    >>> paths = generate_gbm_paths(s0=100.0, mu=0.05, sigma=0.2,
    ...                           maturity=1.0, n_steps=252, n_paths=100000)
    >>> 
    >>> # Create payoff function
    >>> payoff_fn = american_put(strike=100.0)
    >>> 
    >>> # Initialize pricer
    >>> pricer = LSMPricer(payoff_fn=payoff_fn, discount=0.05)
    >>> 
    >>> # Price the option
    >>> price = pricer.price(paths, exercise_idx=list(range(252)))
    >>> print(f"Option price: {price:.4f}")
"""

# File: lsm_engine/__init__.py
from .config import CONFIG
from .engine.pricer import LSMPricer
from .engine.pathgen import generate_gbm_paths
from .engine.regression import PolyRegressor
from .payoffs.vanilla import american_put

__all__ = [
    "CONFIG",
    "LSMPricer",
    "generate_gbm_paths",
    "PolyRegressor",
    "american_put",
]
