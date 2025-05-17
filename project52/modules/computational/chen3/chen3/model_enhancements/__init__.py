# File: chen3/model_enhancements/__init__.py
"""
Model enhancement modules for the Chen3 package.

This package provides various model enhancements that can be used to
extend the base Chen3 model with additional features and dynamics.
These enhancements allow for more realistic modeling of financial
markets and instruments.

Available Enhancements:
--------------------
1. Rough Volatility (RoughVolatility)
   - Fractional Brownian motion-driven volatility
   - Captures volatility roughness and long-range dependence

2. Stochastic Dividend (StochasticDividend)
   - Ornstein-Uhlenbeck process for dividend yields
   - Models mean-reverting dividend behavior

3. Local Volatility (LocalVolatility)
   - State-dependent volatility surface
   - Calibrated to market-implied volatilities

4. Jump-Diffusion Processes
   - Merton's lognormal jumps (MertonJump)
   - Kou's double exponential jumps (KouJump)
   - Models discontinuous price movements

5. Regime Switching (RegimeSwitcher)
   - Discrete Markov chain for regime transitions
   - Regime-specific parameter sets

Usage:
-----
These enhancements can be used individually or in combination to
create more sophisticated models. For example:

>>> from chen3.model_enhancements import RoughVolatility, MertonJump
>>> vol_model = RoughVolatility(hurst=0.1, eta=0.3, v0=0.04, n_steps=100)
>>> jump_model = MertonJump(lam=0.5, mu_j=-0.1, sigma_j=0.2)
"""

from .jump_diffusion import KouJump, MertonJump
from .local_volatility import LocalVolatility
from .regime_switching import RegimeSwitcher
from .rough_volatility import RoughVolatility
from .stochastic_dividend import StochasticDividend

__all__ = [
    "RoughVolatility",
    "StochasticDividend",
    "LocalVolatility",
    "MertonJump",
    "KouJump",
    "RegimeSwitcher",
]
