# File: chen3/schemes/__init__.py
"""Schemes registry."""
from .euler import euler_scheme
from .adf import adf_scheme
from .exact import exact_scheme
__all__ = ["euler_scheme", "adf_scheme", "exact_scheme"]