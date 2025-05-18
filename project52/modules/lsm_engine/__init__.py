"""
LSM Engine - A Longstaff-Schwartz Method Implementation for American/Bermudan Option Pricing

This module provides a comprehensive implementation of the Longstaff-Schwartz Method (LSM)
for pricing American and Bermudan options. The implementation includes:
- Path generation using Geometric Brownian Motion (GBM)
- Polynomial regression for continuation value estimation
- Configurable pricing engine with various payoff functions

Main components:
- LSMPricer: Core pricing engine implementing the LSM algorithm
- PolyRegressor: Polynomial regression for continuation value estimation
- generate_gbm_paths: Path generation using GBM
- Various payoff functions for different option types
"""

# File: lsm_engine/__init__.py
from .config import CONFIG
from .engine.pricer import LSMPricer
from .engine.pathgen import generate_gbm_paths
from .engine.regression import PolyRegressor
from .payoffs.vanilla import american_put
