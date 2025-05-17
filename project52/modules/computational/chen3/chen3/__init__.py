# File: chen3/__init__.py
"""chen3: Three-factor Chen model package."""
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

__all__ = [
    "Settings", "RateParams", "EquityParams", "ModelParams",
    "ChenModel",
    "ShortRateCIR", "EquityProcess", "RoughVariance",
    "euler_scheme", "adf_scheme", "exact_scheme",
    "cholesky_correlation", "make_simulator",
    "MonteCarloPricer", "calibrate",
]