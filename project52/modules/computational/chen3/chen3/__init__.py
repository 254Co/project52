# File: chen3/__init__.py
"""
Chen3: A Three-Factor Stochastic Model for Financial Derivatives

This package implements a comprehensive three-factor stochastic model for pricing
complex financial derivatives. The model combines stochastic interest rates,
equity prices, and rough volatility to provide a realistic framework for
financial instrument valuation.

Key Features:
------------
1. Three-Factor Model:
   - Stochastic interest rates (CIR process)
   - Equity prices with stochastic volatility
   - Rough volatility dynamics

2. Advanced Correlation Structures:
   - Time-dependent correlations
   - State-dependent correlations
   - Regime-switching correlations
   - Stochastic correlations
   - Copula-based correlations

3. Numerical Methods:
   - Monte Carlo simulation
   - Adaptive time stepping
   - Variance reduction techniques
   - Parallel computation
   - GPU acceleration

4. Pricing Capabilities:
   - European options
   - American options
   - Path-dependent options
   - Complex structured products
   - Interest rate derivatives

Mathematical Formulation:
------------------------
The three-factor Chen model is defined by the following system of SDEs:

1. Interest Rate Process (CIR):
   dr_t = κ(θ - r_t)dt + σ√r_t dW^r_t

2. Equity Price Process:
   dS_t = (r_t - q)S_t dt + √v_t S_t dW^S_t

3. Variance Process (Heston):
   dv_t = κ_v(θ_v - v_t)dt + σ_v√v_t dW^v_t

where:
- r_t: Interest rate at time t
- S_t: Equity price at time t
- v_t: Variance at time t
- W^r_t, W^S_t, W^v_t: Correlated Brownian motions
- κ, θ, σ: CIR process parameters
- κ_v, θ_v, σ_v: Heston process parameters
- q: Continuous dividend yield

Example Usage:
-------------
    >>> from chen3 import ChenModel, ModelParams, RateParams, EquityParams
    >>> from chen3.correlation import TimeDependentCorrelation
    >>> import numpy as np
    >>>
    >>> # Define interest rate parameters
    >>> rate_params = RateParams(
    ...     kappa=0.1,    # Mean reversion speed
    ...     theta=0.05,   # Long-term mean
    ...     sigma=0.1,    # Volatility
    ...     r0=0.03       # Initial rate
    ... )
    >>>
    >>> # Define equity parameters
    >>> equity_params = EquityParams(
    ...     mu=0.05,      # Drift
    ...     q=0.02,       # Dividend yield
    ...     S0=100.0,     # Initial stock price
    ...     v0=0.04,      # Initial variance
    ...     kappa_v=2.0,  # Variance mean reversion
    ...     theta_v=0.04, # Long-term variance
    ...     sigma_v=0.3   # Volatility of variance
    ... )
    >>>
    >>> # Create time-dependent correlation
    >>> time_points = np.array([0.0, 1.0, 2.0])
    >>> corr_matrices = [
    ...     np.array([[1.0, 0.5, 0.3],
    ...              [0.5, 1.0, 0.2],
    ...              [0.3, 0.2, 1.0]]),
    ...     np.array([[1.0, 0.6, 0.4],
    ...              [0.6, 1.0, 0.3],
    ...              [0.4, 0.3, 1.0]]),
    ...     np.array([[1.0, 0.7, 0.5],
    ...              [0.7, 1.0, 0.4],
    ...              [0.5, 0.4, 1.0]])
    ... ]
    >>> time_corr = TimeDependentCorrelation(
    ...     time_points=time_points,
    ...     correlation_matrices=corr_matrices
    ... )
    >>>
    >>> # Create model parameters
    >>> model_params = ModelParams(
    ...     rate=rate_params,
    ...     equity=equity_params,
    ...     correlation=time_corr
    ... )
    >>>
    >>> # Create and use the model
    >>> model = ChenModel(model_params)
    >>>
    >>> # Define a payoff function (e.g., European call option)
    >>> def payoff_function(r_paths, S_paths, v_paths):
    ...     return np.maximum(S_paths[:, -1] - 100, 0)
    >>>
    >>> # Price the instrument
    >>> price = model.price_instrument(
    ...     payoff_function,
    ...     n_paths=10000,
    ...     n_steps=100,
    ...     dt=0.01
    ... )

Module Structure:
----------------
The package is organized into the following modules:

1. Core Components:
   - model.py: Main model implementation
   - datatypes.py: Parameter data structures
   - config.py: Configuration management

2. Correlation Module:
   - correlation/advanced.py: Advanced correlation structures
   - correlation/examples.py: Usage examples
   - correlation/__init__.py: Module interface

3. Numerical Methods:
   - integration/: Integration schemes
   - schemes/: Discretization methods
   - simulators/: Path simulation

4. Pricing Components:
   - payoffs/: Payoff functions
   - pricers/: Pricing engines
   - greeks/: Sensitivity calculations

5. Utilities:
   - utils/: Helper functions
   - risk/: Risk metrics
   - calibration/: Model calibration

For more detailed documentation and examples, see the individual module
docstrings and the examples directory.
"""

from .config import Settings
from .datatypes import EquityParams, ModelParams, RateParams
from .model import Chen3Model

__version__ = "1.0.0"
__author__ = "Chen3 Development Team"

__all__ = ["Chen3Model", "ModelParams", "RateParams", "EquityParams", "Settings"]
